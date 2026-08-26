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
"""

import argparse
import gc
import json
import os
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

    for nb, (audio, metadata) in enumerate(loader):
        print(f"Processing batch {nb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        audio = audio.to(device)
        if args.model_half:
            audio = audio.half()

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
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            with open(os.path.join(args.output_path, f"{latent_id}.json"), "w") as f:
                json.dump(md, f)

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
