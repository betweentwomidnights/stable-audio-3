"""Find the audio gain factor that lands pre-encoded latents at a target RMS.

Loads a few representative source clips, scales by a range of gains, encodes,
and reports the latent RMS for each. Use the recommended gain as
`--audio_gain` in pre_encode_dataset.py to fix the loudness-bias issue.

Default target = 0.701, measured as the un-LoRA'd model's natural latent RMS.
"""
import argparse
import glob
import os
import statistics

import torch
import torchaudio

from stable_audio_3 import AutoencoderPipeline


def measure_latent_rms(ae, audio, sr, gain):
    audio_scaled = audio * gain
    with torch.no_grad():
        z = ae.encode(audio_scaled, sr)
    return float((z.float() ** 2).mean().sqrt().item())


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AutoencoderPipeline.from_pretrained(args.model, device=device)
    if args.model_half:
        ae.autoencoder = ae.autoencoder.half()

    wavs = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if args.n_clips and args.n_clips < len(wavs):
        # Sample evenly across the sorted list for representativeness.
        step = len(wavs) // args.n_clips
        wavs = [wavs[i * step] for i in range(args.n_clips)]
    print(f"Calibrating on {len(wavs)} clip(s) from {args.data_dir}")
    print(f"Target latent RMS: {args.target:.3f}\n")

    gains = args.gains if args.gains else [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"{'gain':>6s} | " + " | ".join(f"{os.path.basename(w)[:18]:>18s}" for w in wavs) + " |   mean   median")
    print("-" * (10 + 21 * len(wavs) + 18))

    rms_by_gain = {}
    for gain in gains:
        per_clip = []
        for wav_path in wavs:
            audio, sr = torchaudio.load(wav_path)
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            audio = audio.to(device)
            # Leave audio in float — the AE preprocess pipeline includes a sinc
            # resampler whose kernel is float, and fp16 inputs trigger a dtype clash
            # when resampling is needed (e.g. mixed 44.1 / 48 kHz datasets).
            # Crop to a manageable window so encoding is fast.
            max_samples = args.clip_seconds * sr
            if audio.shape[-1] > max_samples:
                start = (audio.shape[-1] - max_samples) // 2
                audio = audio[..., start:start + max_samples]
            rms = measure_latent_rms(ae, audio.unsqueeze(0), sr, gain)
            per_clip.append(rms)
        mean = statistics.mean(per_clip)
        median = statistics.median(per_clip)
        rms_by_gain[gain] = (mean, median)
        cells = " | ".join(f"{r:18.3f}" for r in per_clip)
        print(f"{gain:6.2f} | {cells} | {mean:6.3f}  {median:6.3f}")

    print()
    # Find the gain whose mean RMS is closest to target.
    best = min(rms_by_gain.items(), key=lambda kv: abs(kv[1][0] - args.target))
    print(f"Closest gain to target RMS={args.target:.3f}: {best[0]} (mean RMS={best[1][0]:.3f})")
    # Interpolate for a finer estimate, assuming roughly linear relationship.
    sorted_gains = sorted(rms_by_gain.keys())
    means = [rms_by_gain[g][0] for g in sorted_gains]
    if min(means) <= args.target <= max(means):
        for i in range(len(sorted_gains) - 1):
            r0, r1 = means[i], means[i + 1]
            if min(r0, r1) <= args.target <= max(r0, r1):
                g0, g1 = sorted_gains[i], sorted_gains[i + 1]
                # Linear interp on (gain, rms)
                frac = (args.target - r0) / (r1 - r0) if r1 != r0 else 0.0
                g_interp = g0 + frac * (g1 - g0)
                print(f"Interpolated gain for exact target: {g_interp:.3f}")
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="same-l")
    p.add_argument("--data_dir", default="/root/sa3_training")
    p.add_argument("--n_clips", type=int, default=5,
                   help="Number of clips to sample (evenly across the sorted list).")
    p.add_argument("--clip_seconds", type=int, default=30,
                   help="Crop each clip to a centered window of this many seconds.")
    p.add_argument("--target", type=float, default=0.701,
                   help="Target latent RMS — the un-LoRA'd model's natural output.")
    # Calibration uses fp32 — the AE preprocess sinc resampler is fp32, and forcing
    # half precision creates dtype mismatches on mixed sample-rate datasets.
    p.add_argument("--model_half", action="store_true", default=False)
    p.add_argument("--gains", type=float, nargs="*", default=None,
                   help="Custom gain factors to sweep (default 0.3..1.0).")
    main(p.parse_args())
