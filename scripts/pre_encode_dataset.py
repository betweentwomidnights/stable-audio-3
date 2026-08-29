"""
Pre-encode a dataset of audio clips into latents using Stable Audio 3, saving the latents and metadata to disk.

Dataset layout:
  data_dir/
    clip1.wav   (or .flac, .mp3, .ogg)
    clip1.txt   ← text prompt for clip1
    clip2.wav
    clip2.txt
    ...

Saves .npy files for latents and .json files for metadata, compatible with train_lora.py --encoded_dir.

Usage:
  uv run python scripts/pre_encode_dataset.py --model same-s --data_dir ./my_data --output_path ./latents_out
  uv run python scripts/pre_encode_dataset.py --model same-l --data_dir ./my_data --output_path ./latents_out --batch_size 4
  uv run python scripts/pre_encode_dataset.py --model same-l --data_dir ./my_data --output_path ./latents_out --per_track_target_latent_rms 0.90
"""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from stable_audio_3 import AutoencoderModel
from stable_audio_3.model_configs import ae_models
from stable_audio_3.models.lora.loader import load_and_apply_loras
from stable_audio_3.models.lora.model import set_lora_strength
from stable_audio_3.models.lora.utils import get_lora_params
from stable_audio_3.data.dataset import (
    LocalDatasetConfig,
    SampleDataset,
    collation_fn,
)


def caption_metadata_fn(info, _audio):
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}


def _per_clip_latent_rms(z, metadata):
    """Per-clip latent RMS over the valid (non-padded) region.

    Padding is excluded on purpose. A short clip padded out to --sample_size
    would otherwise have its RMS dragged toward zero by the padding, and the
    correction below would over-drive the audio to compensate.
    """
    latent_len = z.shape[-1]
    out = []
    for i in range(z.shape[0]):
        pm = metadata[i]["padding_mask"][0]
        if not isinstance(pm, torch.Tensor):
            pm = torch.as_tensor(pm)
        mask = (
            F.interpolate(pm.reshape(1, 1, -1).float(), size=latent_len, mode="nearest")
            .reshape(-1)
            .to(z.device)
            .bool()
        )
        z_clip = z[i].float()
        valid = z_clip[..., mask] if mask.any() else z_clip
        out.append(valid.pow(2).mean().sqrt().clamp(min=1e-6).item())
    return out


def encode_with_per_track_norm(ae, audio, metadata, target_latent_rms, max_iters, tol):
    """Iterative per-track latent-RMS normalization.

    Why iterative: a single scale-the-audio-then-encode pass UNDERSHOOTS the
    target, because the encoder is nonlinear. Requesting 0.90 lands around 0.82
    in one pass. So encode at the current per-clip gain, measure the latent RMS,
    multiply the gain by (target / measured), and repeat until every clip is
    within `tol` (relative) or `max_iters` is reached.

    The latents returned are exactly the ones last measured, so the reported
    `achieved` figures describe the data actually written -- no extra encode
    afterwards that could drift from what was reported.

    Returns (latents, per_clip_gains, pre_norm_rms, achieved_rms, iters_used).
    """
    n = audio.shape[0]
    gain = torch.ones(n, device=audio.device, dtype=audio.dtype)
    pre_norm = None
    iters_used = 0
    for it in range(max_iters + 1):
        with torch.no_grad():
            z = ae.encode(audio * gain.view(-1, 1, 1), ae.sample_rate)
        rms = _per_clip_latent_rms(z, metadata)
        if it == 0:
            pre_norm = list(rms)
        iters_used = it
        max_rel_err = max(abs(r - target_latent_rms) / target_latent_rms for r in rms)
        if max_rel_err <= tol or it == max_iters:
            achieved = list(rms)
            latents = z
            break
        corr = torch.tensor(
            [target_latent_rms / r for r in rms],
            device=audio.device,
            dtype=audio.dtype,
        )
        gain = gain * corr
    return latents, gain.tolist(), pre_norm, achieved, iters_used


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = AutoencoderModel.from_pretrained(args.model, device=str(device))

    if args.encoder_lora:
        # Pre-encoding with an adapted encoder changes the TRAINING TARGETS a
        # downstream LoRA learns from, so this is not a cosmetic flag: latents
        # written with it are not interchangeable with stock ones, and mixing
        # the two in one dataset is a silent way to get an incoherent corpus.
        load_and_apply_loras(ae.autoencoder, [args.encoder_lora], "autoencoder")
        n_enc = len(list(get_lora_params(ae.autoencoder.encoder)))
        if n_enc == 0:
            raise SystemExit(
                f"{args.encoder_lora} attached no tensors to the encoder. A "
                f'checkpoint needs `target: "encoder"` in its config to land '
                f"here; a DiT or decoder adapter will load without complaint "
                f"and do nothing."
            )
        set_lora_strength(
            ae.autoencoder.encoder, args.encoder_lora_strength, lora_index=0
        )
        print(
            f"Encoder LoRA: {args.encoder_lora} -> {n_enc} tensors "
            f"@ strength {args.encoder_lora_strength}"
        )

    if args.model_half:
        ae.autoencoder = ae.autoencoder.half()

    dataset = SampleDataset(
        [
            LocalDatasetConfig(
                id="train", path=args.data_dir, custom_metadata_fn=caption_metadata_fn
            )
        ],
        sample_size=args.sample_size,
        sample_rate=ae.sample_rate,
        force_channels="stereo",
        # SampleDataset defaults to random_crop=True. For a track LONGER than
        # --sample_size that picks a random offset, so two pre-encodes of the
        # same corpus encode different audio. A one-shot pass has no use for a
        # random crop -- LoRA training does its own cropping in latent space,
        # which is where that augmentation belongs.
        random_crop=args.random_crop,
    )
    if not args.phase_flip:
        # SampleDataset hardcodes augs = Sequential(PhaseFlipper(p=0.5)), which
        # inverts polarity on a coin flip. During TRAINING that is fair: a clip
        # is seen many times, so it is seen both ways. In a ONE-SHOT pre-encode
        # each track is encoded exactly once, so the flip adds no diversity and
        # only makes the output irreproducible. Polarity is inaudible, but the
        # encoder is nonlinear, so z(-x) is essentially UNCORRELATED with z(x):
        # two stock pre-encodes of one corpus measured 78% apart on average
        # (~141% on the tracks that flipped, ~6% on those that did not) against
        # an encoder noise floor of 1.31%. That silently destroys any controlled
        # comparison between two sets of latents.
        dataset.augs = torch.nn.Identity()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collation_fn,
    )

    os.makedirs(args.output_path, exist_ok=True)

    silence_path = os.path.join(args.output_path, "silence.npy")
    if not os.path.exists(silence_path):
        print("Saving silence latent")
        silence_audio = torch.zeros(
            1, ae.autoencoder.io_channels, args.sample_size, device=device
        )
        if args.model_half:
            silence_audio = silence_audio.half()
        with torch.no_grad():
            silence_latent = ae.encode(silence_audio, ae.sample_rate)
        np.save(silence_path, silence_latent.cpu().numpy())

    per_track = args.per_track_target_latent_rms > 0
    all_gains, all_pre, all_ach = [], [], []

    for nb, (audio, metadata) in enumerate(loader):
        print(f"Processing batch {nb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        audio = audio.to(device)
        if args.model_half:
            audio = audio.half()

        if per_track:
            (
                latents,
                gains,
                pre_norm,
                achieved,
                iters_used,
            ) = encode_with_per_track_norm(
                ae,
                audio,
                metadata,
                args.per_track_target_latent_rms,
                args.norm_iters,
                args.norm_tol,
            )
            print(f"  batch {nb}: norm settled in {iters_used} iter(s)")
        else:
            latents = ae.encode(audio, ae.sample_rate)

        for i, latent in enumerate(latents):
            latent_np = latent.cpu().numpy()
            latent_id = f"{nb:06d}{i:04d}"

            md = dict(metadata[i])
            padding_mask = (
                F.interpolate(
                    md["padding_mask"][0].unsqueeze(0).unsqueeze(1).float(),
                    size=latent_np.shape[-1],
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .int()
            )
            if not args.pad:
                padding_np = padding_mask.cpu().numpy()
                valid_indices = np.where(padding_np == 1)[0]
                if len(valid_indices) > 0:
                    valid_length = valid_indices[-1] + 1
                    latent_np = latent_np[:, :valid_length]
                    padding_mask = padding_mask[:valid_length]

            np.save(os.path.join(args.output_path, f"{latent_id}.npy"), latent_np)

            md["padding_mask"] = padding_mask.cpu().numpy().tolist()
            if per_track:
                # Record what this latent actually got, so a set of latents is
                # self-describing rather than relying on the command that made
                # it being remembered correctly.
                md["audio_gain_applied"] = float(gains[i])
                md["latent_rms_pre_norm"] = float(pre_norm[i])
                md["latent_rms_achieved"] = float(achieved[i])
                all_gains.append(float(gains[i]))
                all_pre.append(float(pre_norm[i]))
                all_ach.append(float(achieved[i]))
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            with open(os.path.join(args.output_path, f"{latent_id}.json"), "w") as f:
                json.dump(md, f)

    if per_track and all_ach:
        tgt = args.per_track_target_latent_rms
        worst = max(abs(r - tgt) / tgt for r in all_ach)
        print(f"\n[per-track norm] {len(all_ach)} clips, target latent RMS = {tgt}")
        print(
            f"  pre-norm RMS  min={min(all_pre):.4f} "
            f"mean={statistics.fmean(all_pre):.4f} "
            f"median={statistics.median(all_pre):.4f} max={max(all_pre):.4f}"
        )
        print(
            f"  ACHIEVED RMS  min={min(all_ach):.4f} "
            f"mean={statistics.fmean(all_ach):.4f} "
            f"median={statistics.median(all_ach):.4f} max={max(all_ach):.4f} "
            f"std={statistics.pstdev(all_ach):.4f}"
        )
        print(
            f"  applied gain  min={min(all_gains):.4f} "
            f"mean={statistics.fmean(all_gains):.4f} max={max(all_gains):.4f}"
        )
        print(
            f"  worst clip is {worst * 100:.2f}% from target "
            f"(tol {args.norm_tol * 100:.0f}%, max_iters {args.norm_iters}). "
            "Want the achieved mean at the target with a tight std."
        )

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode audio dataset to latents")
    parser.add_argument("--model", choices=list(ae_models), default="same-l")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Folder with audio files and matching .txt captions",
    )
    parser.add_argument(
        "--output_path", required=True, help="Folder to write .npy/.json latent pairs"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=12582912,  # 380s at 44.1kHz, 2 channels
        help="Audio samples to pad/crop to (default ~380s at 44.1kHz)",
    )
    parser.add_argument(
        "--model_half", action="store_true", help="Run autoencoder in fp16"
    )
    parser.add_argument(
        "--pad", action="store_true", help="Pad audio samples to --sample_size"
    )
    parser.add_argument(
        "--encoder_lora",
        default=None,
        help="encoder-targeted LoRA (.safetensors) to encode through. Changes "
        "the latents a downstream LoRA trains on; see "
        "docs/workflows/encoder-lora.md",
    )
    parser.add_argument("--encoder_lora_strength", type=float, default=1.0)
    parser.add_argument(
        "--per_track_target_latent_rms",
        type=float,
        default=0.0,
        help="If > 0, scale each clip individually so its encoded latent RMS "
        "hits this target. Equalises latent scale across a dataset whose "
        "tracks were mastered at different levels, so an adapter trained on "
        "it does not also learn those level differences. Set it to the base "
        "model's own latent scale (~0.90 for same-l). Default 0.0 = off.",
    )
    parser.add_argument(
        "--norm_iters",
        type=int,
        default=4,
        help="Max correction rounds for --per_track_target_latent_rms. The "
        "encoder is nonlinear, so one pass undershoots; each round encodes "
        "again, so cost is up to norm_iters+1 encodes per clip. Stops early "
        "once every clip is within --norm_tol.",
    )
    parser.add_argument(
        "--norm_tol",
        type=float,
        default=0.03,
        help="Relative tolerance for --per_track_target_latent_rms "
        "convergence. Default 0.03.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="DataLoader workers; 0 loads in-process. Windows spawns rather "
        "than forks, so the workers re-import this module and re-pickle the "
        "dataset (captions arrive through a custom_metadata_fn, hence dill), "
        "and a worker that dies there surfaces only as 'DataLoader worker "
        "exited unexpectedly' with no cause. 0 both diagnoses that and works "
        "around it, at little cost here: a pre-encode is one pass over a "
        "corpus and the GPU is the bottleneck, not the decode.",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="random crop offsets for tracks longer than --sample_size "
        "(default off: a one-shot pre-encode should be reproducible)",
    )
    parser.add_argument(
        "--phase_flip",
        action="store_true",
        help="restore SampleDataset's random polarity flip (default off: it "
        "makes a one-shot pre-encode irreproducible without adding diversity)",
    )
    args = parser.parse_args()

    if not args.pad and args.batch_size > 1:
        parser.error(
            "padding is required for batch_size > 1; pass --pad or use --batch_size 1"
        )

    main(args)
