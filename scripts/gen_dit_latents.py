"""
Sample base-DiT latents to .npy for the decoder-LoRA `dit` bucket.

Why this bucket exists: every DiT-sampled latent drifts 15-18.5% on the
round-trip self-consistency metric ||E(D(z))-z||/||z||, against 10.5% for the
encoder's own output on real audio. That gap is in the BASE model, so the decoder
needs to see generated latents during training or it will only ever be corrected
on the real-audio distribution it already handles best.

It is deliberately BASE-only. kev/koan measured drift deltas under half a
percent against base on identical prompts, so adapter-specific latents buy
nothing -- what the LoRAs change is how much HF content is present, not how
well-formed the latent is.

Usage:
  python scripts/gen_dit_latents.py --out_dir out/dit_latents --n 64 \
      --prompt_pool prompts.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_audio_3 import StableAudioModel  # noqa: E402

FALLBACK_PROMPTS = [
    "drums, percussion, snare fills, hi hats",
    "breakcore, glitch hop, intricate breakbeat",
    "post rock, electric guitar, live drums",
    "lo-fi hip hop, downtempo, jazzy",
    "synthwave, cinematic electronica",
    "neurofunk, drum and bass",
    "math rock, angular guitars",
    "ambient techno, melodic electronica",
]


def load_prompts(path):
    if not path:
        return FALLBACK_PROMPTS
    data = json.loads(Path(path).read_text())
    out = []
    for v in (data.get("dice") or {}).values():
        if isinstance(v, list):
            out.extend(str(x) for x in v)
    return out or FALLBACK_PROMPTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--model", default="medium")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--prompt_pool", default=None)
    cli = p.parse_args()

    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(cli.prompt_pool)
    print(f"[gen] {len(prompts)} prompts; generating {cli.n} latents")

    # half=True matches production, so the latents we train the decoder on are
    # the same ones it will actually be handed at serve time.
    pipe = StableAudioModel.from_pretrained(cli.model, model_half=True)

    for i in range(cli.n):
        prompt = prompts[i % len(prompts)]
        seed = cli.seed0 + i
        z = pipe.generate(
            prompt=prompt,
            negative_prompt="low quality",
            duration=cli.duration,
            steps=cli.steps,
            cfg_scale=1.0,
            seed=seed,
            return_latents=True,
        )
        arr = z[0].detach().float().cpu().numpy()
        np.save(out / f"dit_{i:04d}_s{seed}.npy", arr)
        if (i + 1) % 8 == 0 or i == 0:
            print(
                f"[gen] {i + 1}/{cli.n}  shape={arr.shape}  "
                f"std={arr.std():.4f}  | {prompt[:44]}"
            )
        del z
        torch.cuda.empty_cache()

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "model": cli.model,
                "n": cli.n,
                "duration": cli.duration,
                "steps": cli.steps,
                "seed0": cli.seed0,
                "note": "base DiT latents, no adapter; fp16 model, fp32 npy",
            },
            indent=2,
        )
    )
    print(f"[gen] wrote {cli.n} latents to {out}")


if __name__ == "__main__":
    main()
