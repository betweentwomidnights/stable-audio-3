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

Loudness normalization (ported from local/pre-upstream-lora, merged onto the
Lightning-era output format):
  --per_track_target_latent_rms R  iterative per-clip normalization — encode,
    measure each clip's latent RMS over its valid (non-padded) region, correct
    the gain by (R / measured), re-encode, repeat until within tol (the VAE is
    nonlinear so a single pass undershoots ~10%). Removes per-clip loudness
    variance so the LoRA doesn't learn loudness as a feature. Set R to the
    BASE model's latent scale (~0.90, measure first) to make the LoRA
    loudness-NEUTRAL vs base — fixes the "quieter than base at strength>1"
    regression the patch dataset showed at target 0.70 (0.70 < base 0.90 ⇒
    LoRA pulled latents below base, compounding with strength).
  --audio_gain G  legacy single global gain (only used when per-track is off).
When per-track is active the run prints a latent-RMS distribution summary so
the dataset's hotness and the chosen target can be sanity-checked.

Usage:
  uv run python scripts/pre_encode_dataset.py --model same-s --data_dir ./my_data --output_path ./latents_out
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


def resize_padding_mask(padding_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize a padding mask to target_length using ceiling-based length scaling.

    Unlike F.interpolate(mode="nearest"), this ensures any target position
    that partially overlaps valid audio is marked valid (rounds up).
    """
    if padding_mask.ndim == 1:
        valid_length = padding_mask.sum()
        source_length = padding_mask.shape[0]
        valid_target_length = (
            torch.ceil(valid_length.float() * target_length / source_length)
            .long()
            .clamp(max=target_length)
        )
        positions = torch.arange(target_length, device=padding_mask.device)
        return positions < valid_target_length
    else:
        valid_lengths = padding_mask.sum(dim=-1)  # (B,)
        source_length = padding_mask.shape[-1]
        valid_target_lengths = (
            torch.ceil(valid_lengths.float() * target_length / source_length)
            .long()
            .clamp(max=target_length)
        )
        positions = torch.arange(target_length, device=padding_mask.device).unsqueeze(0)
        return positions < valid_target_lengths.unsqueeze(1)


def _per_clip_latent_rms(z, metadata):
    """Per-clip latent RMS over the valid (non-padded) region."""
    latent_len = z.shape[-1]
    out = []
    for i in range(z.shape[0]):
        # Match main's working access: md["padding_mask"][0] is the 1-D (T,)
        # audio-length mask (main does [0].unsqueeze(0).unsqueeze(1)).
        pm = metadata[i]["padding_mask"][0]
        if not isinstance(pm, torch.Tensor):
            pm = torch.as_tensor(pm)
        if pm.ndim > 1:
            pm = pm.reshape(-1)
        mask_latent = resize_padding_mask(pm, latent_len).to(z.device).bool()
        z_clip = z[i].float()
        valid = z_clip[..., mask_latent] if mask_latent.any() else z_clip
        out.append(valid.pow(2).mean().sqrt().clamp(min=1e-6).item())
    return out


def encode_with_per_track_norm(ae, audio, metadata, target_latent_rms,
                               max_iters, tol):
    """Iterative per-track latent-RMS normalization.

    A single scale-audio-then-re-encode pass UNDERSHOOTS the target because
    the VAE encoder is nonlinear (measured: requesting 0.90 lands ~0.82).
    So iterate: encode at the current per-clip gain, measure latent RMS,
    multiply the gain by (target / measured), repeat until every clip is
    within `tol` (relative) or `max_iters` is hit. The returned latents are
    exactly the ones we last measured (no extra encode, no drift between
    reported `achieved` and saved data).

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
        max_rel_err = max(abs(r - target_latent_rms) / target_latent_rms
                          for r in rms)
        if max_rel_err <= tol or it == max_iters:
            achieved = list(rms)
            latents = z
            break
        corr = torch.tensor(
            [target_latent_rms / r for r in rms],
            device=audio.device, dtype=audio.dtype,
        )
        gain = gain * corr
    return latents, gain.tolist(), pre_norm, achieved, iters_used


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = AutoencoderModel.from_pretrained(args.model, device=str(device))
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
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # LOCAL: multiprocessing DataLoader deadlocks on Spark; sync load (beta-report item)
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
    all_gains = []
    all_pre = []
    all_ach = []

    for nb, (audio, metadata) in enumerate(loader):
        print(f"Processing batch {nb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        audio = audio.to(device)
        if args.model_half:
            audio = audio.half()

        if per_track:
            (latents, gains, pre_norm, achieved,
             iters_used) = encode_with_per_track_norm(
                ae, audio, metadata, args.per_track_target_latent_rms,
                args.norm_iters, args.norm_tol,
            )
            print(f"  batch {nb}: norm settled in {iters_used} iter(s)")
        else:
            if args.audio_gain != 1.0:
                audio = audio * args.audio_gain
            latents = ae.encode(audio, ae.sample_rate)
            gains = [float(args.audio_gain)] * audio.shape[0]
            pre_norm = [None] * audio.shape[0]
            achieved = [None] * audio.shape[0]

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
            md["audio_gain_applied"] = float(gains[i])
            if pre_norm[i] is not None:
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
        n = len(all_ach)
        tgt = args.per_track_target_latent_rms
        print(f"\n[per-track norm] {n} clips, target latent RMS = {tgt}")
        print(
            f"  pre-norm  RMS  min={min(all_pre):.4f} "
            f"mean={statistics.fmean(all_pre):.4f} "
            f"median={statistics.median(all_pre):.4f} max={max(all_pre):.4f}"
        )
        print(
            f"  ACHIEVED  RMS  min={min(all_ach):.4f} "
            f"mean={statistics.fmean(all_ach):.4f} "
            f"median={statistics.median(all_ach):.4f} max={max(all_ach):.4f} "
            f"std={statistics.pstdev(all_ach):.4f}"
        )
        worst = max(abs(r - tgt) / tgt for r in all_ach)
        print(
            f"  applied gain   min={min(all_gains):.4f} "
            f"mean={statistics.fmean(all_gains):.4f} max={max(all_gains):.4f}"
        )
        print(
            f"  worst clip rel-err vs target = {worst * 100:.2f}% "
            f"(tol {args.norm_tol * 100:.0f}%, max_iters {args.norm_iters}) "
            "— want ACHIEVED mean ≈ target & tight std for a "
            "loudness-neutral LoRA"
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
        "--audio_gain",
        type=float,
        default=1.0,
        help="Legacy single global gain on audio before encode. Ignored when "
        "--per_track_target_latent_rms is set.",
    )
    parser.add_argument(
        "--per_track_target_latent_rms",
        type=float,
        default=0.0,
        help="If > 0, normalize each clip individually so its encoded latent "
        "RMS hits this target (iterative: encode, measure, correct, repeat — "
        "the VAE is nonlinear so one pass undershoots). Set to the base "
        "model's latent scale (~0.90) for a loudness-neutral LoRA. "
        "Default 0.0 = off (use --audio_gain).",
    )
    parser.add_argument(
        "--norm_iters",
        type=int,
        default=4,
        help="Max correction rounds for --per_track_target_latent_rms "
        "(encodes up to norm_iters+1 times per clip; stops early on tol).",
    )
    parser.add_argument(
        "--norm_tol",
        type=float,
        default=0.03,
        help="Relative tolerance (fraction) for per-track norm convergence; "
        "stop once every clip is within this of the target. Default 0.03.",
    )
    args = parser.parse_args()

    if not args.pad and args.batch_size > 1:
        parser.error(
            "padding is required for batch_size > 1; pass --pad or use --batch_size 1"
        )

    main(args)
