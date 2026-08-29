"""Measure what a decoder LoRA does as re-encode depth grows.

A continuation or a transform re-encodes previously decoded audio, so N chained
operations put the audio through N autoencoder round trips. This walks that
ladder with the decoder adapter off and on, over the same latents, so the
adapter is the only variable.

Two things make the result trustworthy, and both are easy to get wrong:

  screening   The artifact does not appear equally on all material. Every
              candidate is generated first and ranked by how far 6-16 kHz
              tonality runs away over the ladder with the adapter OFF; only the
              worst --screen_top go through the A/B. Numbers from a screened run
              are NOT comparable to numbers from an unscreened one.

  content     A count of over-threshold frames rewards an adapter that simply
              deletes high frequencies. Band energies and SI-SDR are reported
              alongside it for exactly that reason. Read them together: fewer
              tonal frames AND preserved 16-22 kHz is a fix, fewer tonal frames
              AND collapsed 16-22 kHz is damage.

Usage:
  python scripts/measure_decoder_depth.py \
      --decoder_lora out/declora/decoder_lora_step002000.safetensors \
      --model medium-base --duration 120 --depth 3 --screen_top 4 \
      --prompts "a prompt" "another prompt" --seeds 0 1 2 \
      --out_dir out/depth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from stable_audio_3 import StableAudioModel
from stable_audio_3.models.lora.utils import get_lora_params

# Sibling-module import, so this works whether it is run from the repo root or
# from scripts/. Matches what the trainers do.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _decoder_lora_eval import (  # noqa: E402
    si_sdr,
    spectral_flatness_db,
    stft,
    to_mono,
    write_wav,
)

N_FFT, HOP, EPS = 2048, 512, 1e-12


def measure(wave, sr, ref_d0=None):
    mono = to_mono(wave).float()
    mag = stft(mono, N_FFT, HOP).abs()
    _, p95, ton = spectral_flatness_db(mag, N_FFT, sr, 6000.0, 16000.0)
    b0 = int(round(6000 / (sr / N_FFT)))
    b1 = int(round(16000 / (sr / N_FFT)))
    band = mag[b0:b1].pow(2).mean(0)
    fdb = 10.0 * torch.log10(band + EPS)
    gate = fdb >= (fdb.max() - 40.0)
    pw = mag.pow(2).mean(1)

    def bd(lo, hi):
        i0 = int(round(lo / (sr / N_FFT)))
        i1 = int(round(hi / (sr / N_FFT)))
        return float(10.0 * torch.log10(pw[i0:i1].sum() + EPS))

    out = {
        "f12_6_16k": int(((ton > 12.0) & gate).sum()),
        "n_frames": int(gate.numel()),
        "p95": float(p95),
        "band_12_16k": bd(12000, 16000),
        "band_16_22k": bd(16000, 22050),
        "rms_dbfs": float(
            20.0 * torch.log10(wave.pow(2).mean().sqrt().clamp(min=1e-9))
        ),
    }
    if ref_d0 is not None:
        out["si_sdr_vs_own_d0"] = si_sdr(mono, to_mono(ref_d0).float())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_lora", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lora_dir", default="loras")
    p.add_argument(
        "--lora",
        default="none",
        help="DiT LoRA name in --lora_dir, or 'none' for the base model",
    )
    p.add_argument("--model", default="medium-base")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--screen_top", type=int, default=4)
    p.add_argument("--save_audio", type=int, default=1)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    cli = p.parse_args()

    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[dl] loading {cli.model} ...", flush=True)
    pipe = StableAudioModel.from_pretrained(cli.model, model_half=True)
    sr = pipe.model_config.get("sample_rate")

    paths = []
    if cli.lora.lower() != "none":
        for ext in (".ckpt", ".safetensors"):
            c = Path(cli.lora_dir) / f"{cli.lora}{ext}"
            if c.exists():
                paths.append(str(c))
                break
        if not paths:
            print(f"[dl] FATAL: no checkpoint for DiT LoRA {cli.lora!r}")
            return 1
    dit_index = 0 if paths else None
    dec_index = len(paths)
    paths.append(cli.decoder_lora)
    pipe.load_lora(paths)
    pre = pipe.model.pretransform
    decoder = pre.model.decoder
    n_dec = len(list(get_lora_params(decoder)))
    print(f"[dl] decoder adapter tensors: {n_dec}", flush=True)
    if n_dec == 0:
        print("[dl] FATAL: adapter did not attach")
        return 1
    if dit_index is not None:
        pipe.set_lora_strength(1.0, lora_index=dit_index)
    else:
        print("[dl] no DiT LoRA: measuring the base model")

    def set_dec(on):
        pipe.set_lora_strength(1.0 if on else 0.0, lora_index=dec_index)

    def gen(prompt, seed):
        set_dec(False)
        z = pipe.generate(
            prompt=prompt,
            negative_prompt="low quality",
            duration=cli.duration,
            steps=cli.steps,
            cfg_scale=1.0,
            seed=seed,
            return_latents=True,
        )
        dp = next(pre.parameters())
        return z.to(device=dp.device, dtype=dp.dtype)

    def ladder(z, on):
        """Decode, then re-encode/decode in place. Yields (depth, cpu wave)."""
        set_dec(on)
        with torch.inference_mode():
            cur = pre.decode(z)
            yield 0, cur[0].float().cpu()
            for d in range(1, cli.depth + 1):
                cur = pre.decode(pre.encode(cur))
                yield d, cur[0].float().cpu()

    cands = [(pr, sd) for pr in cli.prompts for sd in cli.seeds]
    print(
        f"\n[dl] screening {len(cands)} candidates at {cli.duration:.0f}s "
        f"(adapter off, depth {cli.depth}) ...",
        flush=True,
    )
    scored = []
    for pr, sd in cands:
        z = gen(pr, sd)
        t0 = t1 = None
        for d, w in ladder(z, False):
            m = measure(w, sr)
            if d == 0:
                t0 = m["p95"]
            t1 = m["p95"]
        print(
            f"[dl]   runaway {t1 - t0:+6.2f}  final {t1:6.2f}  s{sd}  {pr[:40]}",
            flush=True,
        )
        scored.append((t1 - t0, pr, sd))
        del z
        torch.cuda.empty_cache()
    scored.sort(key=lambda r: -r[0])
    keep = scored[: cli.screen_top]
    print(f"\n[dl] worst {len(keep)} kept:")
    for r, pr, sd in keep:
        print(f"[dl]   runaway {r:+6.2f}  s{sd}  {pr}")

    rows = {}
    for idx, (_, pr, sd) in enumerate(keep):
        safe = "".join(c if c.isalnum() else "_" for c in pr)[:24]
        stem = f"{idx:02d}_{safe}_s{sd}"
        z = gen(pr, sd)
        for tag, on in (("stock", False), ("v3", True)):
            ref = None
            for d, w in ladder(z, on):
                if d == 0:
                    ref = w
                rows[f"{stem}|{tag}|d{d}"] = measure(w, sr, ref_d0=ref)
                if cli.save_audio and d in (0, cli.depth):
                    write_wav(
                        out / f"{stem}__{tag}_d{d}.wav",
                        w,
                        sr,
                        normalize=True,
                        headroom_db=-1.0,
                    )
        print(f"\n[dl] {stem}")
        print(
            f"      {'depth':>6} {'stock f12':>10} {'v3 f12':>8} "
            f"{'stock p95':>10} {'v3 p95':>8} {'st 12-16k':>10} {'v3 12-16k':>10}"
        )
        for d in range(cli.depth + 1):
            a = rows[f"{stem}|stock|d{d}"]
            b = rows[f"{stem}|v3|d{d}"]
            print(
                f"      {'d' + str(d):>6} {a['f12_6_16k']:>10} {b['f12_6_16k']:>8} "
                f"{a['p95']:>10.2f} {b['p95']:>8.2f} "
                f"{a['band_12_16k']:>10.2f} {b['band_12_16k']:>10.2f}",
                flush=True,
            )
        del z
        torch.cuda.empty_cache()

    (out / "scores.json").write_text(json.dumps(rows, indent=1))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "screened": True,
                "screen_top": cli.screen_top,
                "candidates": len(cands),
                "lora": cli.lora,
                "model": cli.model,
                "duration_s": cli.duration,
                "steps": cli.steps,
                "decoder_lora": os.path.basename(cli.decoder_lora),
                "kept": [
                    {"prompt": pr, "seed": sd, "runaway_db": r} for r, pr, sd in keep
                ],
            },
            indent=1,
        )
    )

    print(f"\n[dl] ===== MEAN OF {len(keep)} (screened, {cli.lora}) =====")
    print(
        f"{'depth':>6} {'stock f12':>10} {'v3 f12':>9} {'delta':>8} "
        f"{'st 12-16k':>10} {'v3 12-16k':>10} {'st sisdr':>9} {'v3 sisdr':>9}"
    )
    for d in range(cli.depth + 1):

        def mn(tag, f):
            v = [rows[f"{k}|{tag}|d{d}"][f] for k in {kk.split("|")[0] for kk in rows}]
            return sum(v) / len(v)

        a, b = mn("stock", "f12_6_16k"), mn("v3", "f12_6_16k")
        pct = 100.0 * (b - a) / max(a, 1e-9)
        s1 = mn("stock", "si_sdr_vs_own_d0") if d else float("nan")
        s2 = mn("v3", "si_sdr_vs_own_d0") if d else float("nan")
        print(
            f"{'d' + str(d):>6} {a:>10.1f} {b:>9.1f} {pct:>+7.0f}% "
            f"{mn('stock', 'band_12_16k'):>10.2f} {mn('v3', 'band_12_16k'):>10.2f} "
            f"{s1:>9.2f} {s2:>9.2f}"
        )
    print(f"\n[dl] wrote {out}")
    return 0


sys.exit(main())
