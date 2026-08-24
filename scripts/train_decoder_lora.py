"""
Train a LoRA on SAME's decoder to stop it manufacturing high-frequency squeaks.

What this is fixing
-------------------
SAME downsamples 44.1 kHz stereo by 4096x -- one latent frame per 93 ms -- so a
snare or hi-hat attack lives entirely inside a single frame. The decoder does not
reconstruct that attack, it resynthesizes one, slightly differently every pass.
Measured over six chained encode/decode round trips -- the case a continuation
feature hits, since each continuation re-encodes previously decoded audio --
added HF energy climbs ~15 dB while lost HF stays flat, and the damage
concentrates on percussive attacks (transient-vs-sustain excess +4.2 -> +12.0 dB,
all worst frames within 35 ms of an onset). The drift is a systematic bias, not
diffusion -- consecutive drift vectors agree at cos 0.69 -- which is why it is
learnable rather than merely unavoidable.

Encoder stays frozen, so the latent space, the DiT and every existing DiT LoRA
remain bit-compatible. Checkpoints are written with `target: "decoder"` and load
through the stock loader onto either a standalone autoencoder or the full serving
model; see docs/workflows/decoder-lora.md.

Training latents come from three buckets, because the artifact shows up on more
than one distribution:

  real   E(x) for real audio. Paired audio, so all three losses apply.
  drift  E(D(E(x))) at depth >= 1 -- the chained-continuation case. Target audio
         is the ORIGINAL x, which asks the decoder to undo accumulated drift
         rather than faithfully render a degraded latent.
  dit    latents sampled from the base DiT (--dit_latent_dir). No paired audio,
         so only the cycle and tonality terms apply. Needed because generated
         latents drift 15-18.5% against real audio's 10.5% -- that gap is in the
         BASE model and is the first-generation artifact's basis. It is NOT
         LoRA-specific: kev/koan measured drift deltas under half a percent, so
         no adapter-specific bucket is required.

Eval is the round-trip ladder itself, run against a held-out clip every
--eval_every steps and compared to a baseline captured at step 0 with the
adapter switched off. The numbers to watch are transient excess and tonality p95
coming down without band energies or reconstruction regressing.

Usage (in the sa3 container, from /workspace/sa3):
  /opt/sa3-venv/bin/python scripts/train_decoder_lora.py \
      --data_dir /data --eval_audio /data/holdout.wav --out_dir /out/declora
"""

import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _decoder_lora_losses import (  # noqa: E402
    cycle_loss,
    hf_band_match_loss,
    hf_tonality,
    hf_tonality_penalty_multiband,
    multires_stft_loss,
    patch_grid_penalty,
)
from _decoder_lora_eval import (  # noqa: E402
    analyse,
    detect_onsets,
    load_audio,
    onset_locked_stats,
    stft,
    to_mono,
    write_wav,
)

from stable_audio_3 import AutoencoderModel  # noqa: E402
from stable_audio_3.models.lora.model import (  # noqa: E402
    LoRAParametrization,
    add_lora,
    set_lora_strength,
)
from stable_audio_3.models.lora.utils import (  # noqa: E402
    get_lora_params,
    get_lora_state_dict,
    save_lora_safetensors,
)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


class EvalArgs:
    """Namespace the probe's analyse()/onset helpers expect."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def list_audio(data_dir, exclude=()):
    files = []
    for p in sorted(Path(data_dir).rglob("*")):
        if p.suffix.lower() in AUDIO_EXTS and str(p) not in exclude:
            files.append(str(p))
    return files


def random_crop(path, sr, seconds, rng, multiple=None):
    """Load a random crop. Returns (1, C, T) float32 on CPU, or None.

    `multiple` (the autoencoder's downsampling ratio) trims the crop to an exact
    multiple of itself. load_audio slices at the file's ORIGINAL rate and then
    resamples, so the final length depends on the source rate and can land
    somewhere the patch/unpatch rearrange cannot factor. Normalising here means
    one code path instead of trusting every file in a 4000-file tree.
    """
    try:
        import torchaudio

        info = torchaudio.info(path)
        total = info.num_frames / float(info.sample_rate)
    except Exception:
        return None
    if total < seconds * 1.05:
        return None
    offset = rng.uniform(0.0, max(0.0, total - seconds * 1.05))
    try:
        w = load_audio(path, sr, seconds=seconds, offset=offset)
    except Exception:
        return None
    if w.dim() != 2 or w.shape[0] < 1 or w.shape[-1] < int(seconds * sr * 0.98):
        return None
    if multiple:
        n = (w.shape[-1] // multiple) * multiple
        if n < multiple:
            return None
        w = w[..., :n]
    if not torch.isfinite(w).all():
        return None
    # Skip near-silent crops: they teach nothing and make the tonality gate
    # meaningless.
    if float(w.pow(2).mean().sqrt()) < 1e-4:
        return None
    return w.unsqueeze(0)


# --------------------------------------------------------------------------
# eval: the round-trip ladder, which is the metric we actually care about
# --------------------------------------------------------------------------


@torch.no_grad()
def eval_ladder(ae, x, sr, ea, iters, onsets=None):
    """Run `iters` encode/decode round trips; report the artifact metrics."""
    # Keep the batch dim the whole way round. ae.encode() is the RAW
    # autoencoder, not the AutoencoderModel wrapper -- it does no preprocessing,
    # so handing it a (C, T) tensor makes the patch rearrange fail.
    drifts = []
    current = x  # (1, C, T)
    z_prev = None
    last = None
    for _ in range(iters):
        z = ae.encode(current)
        if z_prev is not None:
            m = min(z.shape[-1], z_prev.shape[-1])
            drifts.append(
                float(
                    (z[..., :m] - z_prev[..., :m]).float().norm()
                    / z_prev[..., :m].float().norm().clamp(min=1e-9)
                )
            )
        z_prev = z
        current = ae.decode(z).float()
        last = current[0].cpu()

    src_mono = to_mono(x[0].float().cpu())
    if onsets is None:
        onsets = detect_onsets(stft(src_mono, ea.n_fft, ea.hop).abs(), sr, ea.hop)
    stats, _, _, fr = analyse(last, x[0].float().cpu(), sr, ea)
    on_m, off_m, excess, _ = onset_locked_stats(
        fr["ratio_db"],
        fr["valid"],
        onsets,
        fr["ratio_db"].numel(),
        sr,
        ea.hop,
        window_s=ea.onset_window,
    )
    return {
        "tonality_p95": stats["hf_tonality_db_p95"],
        "tonality_mean": stats["hf_tonality_db_mean"],
        "a2s_median": stats["artifact_to_signal_db_median"],
        # Report the two SIDES, not just their difference. `excess` is
        # onset-minus-steady, so it widens when sustained material cleans up
        # faster than transients even though both improved -- without the
        # absolute numbers that reads as "transients got worse".
        "onset_artifact_db": on_m,
        "steady_artifact_db": off_m,
        "transient_excess_db": excess,
        "band_8_12k": stats["band_db"]["brill_8k_12k"],
        "band_12_16k": stats["band_db"]["air_12k_16k"],
        "step_drift_pct": 100 * (sum(drifts) / len(drifts)) if drifts else float("nan"),
        "_final_audio": last,
        "_onsets": onsets,
    }


def fmt_eval(tag, e, base=None):
    def d(key):
        if base is None or base[key] != base[key] or e[key] != e[key]:
            return ""
        return f" ({e[key] - base[key]:+.2f})"

    return (
        f"  {tag:>10}  tonal_p95 {e['tonality_p95']:6.2f}{d('tonality_p95')}"
        f"   onset {e['onset_artifact_db']:6.2f}{d('onset_artifact_db')}"
        f"   steady {e['steady_artifact_db']:6.2f}{d('steady_artifact_db')}"
        f"   excess {e['transient_excess_db']:5.2f}{d('transient_excess_db')}"
        f"   8-12k {e['band_8_12k']:6.1f}{d('band_8_12k')}"
        f"   drift% {e['step_drift_pct']:5.2f}{d('step_drift_pct')}"
    )


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data_dir", required=True, help="dir of real audio (recursive)")
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--eval_audio",
        default=None,
        help="held-out clip for the ladder eval (default: first file)",
    )
    p.add_argument(
        "--dit_latent_dir",
        default=None,
        help="dir of .npy base-DiT latents (see gen_dit_latents.py)",
    )
    p.add_argument("--model", default="same-l", choices=["same-l", "same-s"])

    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha; default = rank (scaling 1.0)",
    )
    p.add_argument(
        "--adapter_type", default="lora", help="lora | dora-rows | bora | ..."
    )

    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--crop_seconds", type=float, default=10.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Bucket sampling weights (real, drift, dit).
    p.add_argument("--w_real", type=float, default=0.5)
    p.add_argument("--w_drift", type=float, default=0.35)
    p.add_argument("--w_dit", type=float, default=0.15)
    p.add_argument("--drift_depth_max", type=int, default=3)

    # Loss weights. lambda_cycle is deliberately large: at init the raw cycle
    # term is ~100x smaller than reconstruction, and it is the term aimed at the
    # mechanism we diagnosed, so it should lead rather than garnish.
    p.add_argument("--lambda_rec", type=float, default=1.0)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_tonal", type=float, default=0.3)
    # v2: match HF band energy to the target. The pair the ear picked out as
    # clearly better moved almost nothing on tonality but gained +2.2 dB at
    # 16-22 kHz, so restored air may be what is actually audible. Only applies
    # to buckets with paired audio.
    p.add_argument("--lambda_hfband", type=float, default=1.0)
    # Default 0 = inert, so this changes nothing until asked for. v1 shipped
    # without it and put a 172 Hz comb on the patch grid that no other term
    # could see; 30 puts the penalty near 0.03 on clean audio (negligible
    # against rec ~= 1.0) and ~0.56 at v1's artifact level. Start LOW: v2's
    # band-match term failed by competing with reconstruction for rank-16
    # capacity, and this one applies to every bucket including dit.
    p.add_argument(
        "--lambda_patch",
        type=float,
        default=0.0,
        help="weight on patch_grid_penalty (0 = off; try 30)",
    )
    # WHERE the tonality penalty looks. The default three sub-bands all sit at
    # 6 kHz and above, which is right for SAME-L -- its invented tonality is
    # genuinely high. It is WRONG for SAME-S, whose artifact ("the banshee":
    # wandering, time-incoherent spectral structure, no stable resonances) lives
    # at 1-8 kHz. Measured stock vs adapted, the p95 tonality gap is +4.26 dB at
    # 4-8 kHz and +0.77 dB at 8-16 kHz, so a 6 kHz floor can barely see it.
    # Whatever this is set to also drives the eval band and the scalar reference
    # measured for the dit bucket, so one flag keeps all three consistent.
    p.add_argument(
        "--tonality_bands",
        default="6000-10000,10000-14000,14000-18000",
        help="comma-separated lo-hi Hz sub-bands for the tonality "
        "penalty (default is tuned for SAME-L; for SAME-S try "
        "1000-2000,2000-4000,4000-8000)",
    )
    # Kept as explicit flags rather than derived from --tonality_bands: the eval
    # ladder's tonal_p95 is the number compared ACROSS runs, and silently moving
    # its band would make new runs incomparable to v1/v3's recorded baselines.
    # Move it deliberately when you move the penalty.
    p.add_argument("--eval_tonality_lo", type=float, default=6000.0)
    p.add_argument("--eval_tonality_hi", type=float, default=16000.0)
    p.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="un-patch grid; must match the model's pretransform",
    )

    p.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="round-trip eval ladder every N steps; 0 = never, which\nalso skips the step-0 baseline ladder there is nothing left to compare against",
    )
    p.add_argument("--eval_iters", type=int, default=4)
    p.add_argument("--eval_seconds", type=float, default=20.0)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    cli = p.parse_args()

    alpha = cli.alpha if cli.alpha is not None else float(cli.rank)
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cli.seed)
    torch.manual_seed(cli.seed)

    try:
        tonality_bands = tuple(
            (float(b.split("-")[0]), float(b.split("-")[1]))
            for b in cli.tonality_bands.split(",")
            if b.strip()
        )
        if not tonality_bands:
            raise ValueError("empty")
    except (ValueError, IndexError):
        print(
            f"[train] bad --tonality_bands {cli.tonality_bands!r}; "
            f"expected e.g. 1000-2000,2000-4000"
        )
        return 1
    ton_lo = min(lo for lo, _ in tonality_bands)
    ton_hi = max(hi for _, hi in tonality_bands)
    print(
        "[train] tonality penalty bands: "
        + ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in tonality_bands)
        + f"  (dit reference measured over {ton_lo:.0f}-{ton_hi:.0f} Hz)"
    )
    print(
        f"[train] eval tonality band: {cli.eval_tonality_lo:.0f}-"
        f"{cli.eval_tonality_hi:.0f} Hz"
    )

    ea = EvalArgs(
        n_fft=2048,
        hop=512,
        hf_lo=6000.0,
        hf_hi=20000.0,
        tonality_lo=cli.eval_tonality_lo,
        tonality_hi=cli.eval_tonality_hi,
        onset_window=0.12,
    )

    # ---- model ----------------------------------------------------------
    print(f"[train] loading {cli.model} on {device} (fp32) ...")
    ae_wrap = AutoencoderModel.from_pretrained(cli.model, device=device)
    ae = ae_wrap.autoencoder
    sr = ae_wrap.sample_rate
    ds_ratio = int(ae.downsampling_ratio)

    ae.requires_grad_(False)
    add_lora(
        ae.decoder,
        lora_config={
            nn.Linear: {
                "weight": partial(
                    LoRAParametrization.from_linear,
                    rank=cli.rank,
                    lora_alpha=alpha,
                    adapter_type=cli.adapter_type,
                )
            }
        },
        include=[""],
    )
    lora_params = list(get_lora_params(ae))
    for prm in lora_params:
        prm.requires_grad_(True)
    n_lora = sum(p.numel() for p in lora_params)
    print(
        f"[train] decoder LoRA: rank={cli.rank} alpha={alpha} "
        f"type={cli.adapter_type} -> {len(lora_params)} tensors, "
        f"{n_lora / 1e6:.2f}M trainable"
    )
    assert sum(p.numel() for p in ae.parameters() if p.requires_grad) == n_lora, (
        "something other than the LoRA is trainable"
    )

    # Without this the patch/unpatch runs under torch.no_grad() and BOTH the
    # reconstruction and cycle terms silently stop reaching the LoRA params --
    # the loss still prints plausible numbers and trains nothing.
    if ae.pretransform is not None:
        ae.pretransform.enable_grad = True
    ae.encoder.eval()
    ae.decoder.train()

    # ---- data -----------------------------------------------------------
    eval_audio = cli.eval_audio
    files = list_audio(cli.data_dir, exclude=(eval_audio,) if eval_audio else ())
    if not files:
        print(f"[train] no audio under {cli.data_dir}")
        return 1
    if eval_audio is None:
        eval_audio = files[0]
        files = files[1:]
    print(f"[train] {len(files)} training files; eval on {eval_audio}")

    dit_latents = []
    if cli.dit_latent_dir:
        dit_latents = sorted(str(p) for p in Path(cli.dit_latent_dir).glob("*.npy"))
        print(f"[train] {len(dit_latents)} base-DiT latents")
    weights = [cli.w_real, cli.w_drift, cli.w_dit if dit_latents else 0.0]
    if not dit_latents and cli.w_dit > 0:
        print(
            "[train] NOTE: --w_dit set but no --dit_latent_dir; "
            "dropping the dit bucket (real/drift only)"
        )
    buckets = ["real", "drift", "dit"]
    tot = sum(weights)
    weights = [w / tot for w in weights]
    print(
        "[train] bucket mix: "
        + ", ".join(f"{b}={w:.2f}" for b, w in zip(buckets, weights))
    )

    # ---- reference tonality for the dit bucket ---------------------------
    # DiT latents have no paired audio, so the one-sided tonality penalty needs
    # a scalar reference. Measure it from the real set rather than guessing.
    ref_tonality = None
    if weights[2] > 0:
        vals = []
        for f in files[:12]:
            w = random_crop(f, sr, min(cli.crop_seconds, 10.0), rng, multiple=ds_ratio)
            if w is None:
                continue
            vals.append(
                float(hf_tonality(to_mono(w[0]).to(device), sr, lo=ton_lo, hi=ton_hi))
            )
        if vals:
            ref_tonality = float(np.median(vals))
            print(
                f"[train] real-audio reference HF tonality = "
                f"{ref_tonality:.2f} dB (median of {len(vals)} crops)"
            )

    # ---- baseline eval (adapter off) ------------------------------------
    # Every eval prints against this, so with --eval_every 0 there is nothing to
    # compare to and the ladder is pure cost -- skip it with the rest of eval.
    x_eval = base_eval = eval_onsets = None
    if cli.eval_every > 0:
        x_eval = load_audio(eval_audio, sr, seconds=cli.eval_seconds, offset=20.0)
        x_eval = x_eval.unsqueeze(0).to(device)
        set_lora_strength(ae.decoder, 0.0)
        ae.decoder.eval()
        base_eval = eval_ladder(ae, x_eval, sr, ea, cli.eval_iters)
        eval_onsets = base_eval["_onsets"]
        write_wav(out_dir / "eval_baseline.wav", base_eval["_final_audio"], sr)
        set_lora_strength(ae.decoder, 1.0)
        ae.decoder.train()
        print(f"\n[train] baseline ladder ({cli.eval_iters} round trips, adapter off):")
        print(fmt_eval("BASE", base_eval))
        print()

    # ---- train ----------------------------------------------------------
    opt = torch.optim.AdamW(lora_params, lr=cli.lr, weight_decay=0.0)

    def lr_at(step):
        if step < cli.warmup:
            return cli.lr * (step + 1) / cli.warmup
        t = (step - cli.warmup) / max(1, cli.steps - cli.warmup)
        return cli.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    history = []
    running = {}
    t0 = time.time()
    step = 0
    skipped = 0
    failed = 0

    while step < cli.steps:
        bucket = rng.choices(buckets, weights=weights, k=1)[0]

        # One unusable file out of thousands must not end a multi-hour run, but a
        # silent skip would hide a systematic problem -- so log, count, and bail
        # if failures dominate rather than grinding on producing nothing.
        src_path = None
        try:
            # ---- build (z, target_audio) --------------------------------
            target = None
            if bucket == "dit":
                src_path = rng.choice(dit_latents)
                arr = np.load(src_path)
                z = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                if z.dim() == 2:
                    z = z.unsqueeze(0)
                n_lat = max(1, int(cli.crop_seconds * sr / ds_ratio))
                if z.shape[-1] > n_lat:
                    s = rng.randrange(0, z.shape[-1] - n_lat + 1)
                    z = z[..., s : s + n_lat]
            else:
                src_path = rng.choice(files)
                w = random_crop(src_path, sr, cli.crop_seconds, rng, multiple=ds_ratio)
                if w is None:
                    skipped += 1
                    if step == 0 and skipped > 200:
                        print(
                            "[train] 200+ unusable crops before a single step; "
                            "check --data_dir and --crop_seconds"
                        )
                        return 1
                    continue
                target = w.to(device)
                with torch.no_grad():
                    z = ae.encode(target)
                    if bucket == "drift":
                        # Walk the ladder to a random depth so the decoder sees
                        # the chained-continuation distribution, not just depth 1.
                        for _ in range(rng.randint(1, cli.drift_depth_max)):
                            z = ae.encode(ae.decode(z))

            # ---- forward / backward -------------------------------------
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)

            y = ae.decode(z)

            l_rec = y.new_zeros(())
            if target is not None:
                l_rec = multires_stft_loss(y, target, sr=sr)

            z_rt = ae.encode(y)
            l_cyc = cycle_loss(z_rt, z)

            y_mono = to_mono(y[0]).unsqueeze(0)
            t_mono = to_mono(target[0]).unsqueeze(0) if target is not None else None
            l_ton = hf_tonality_penalty_multiband(
                y_mono,
                sr,
                target_audio=t_mono,
                target_db=ref_tonality if target is None else None,
                bands=tonality_bands,
            )
            # No paired audio for DiT latents, so there is nothing to match band
            # energy against -- skip rather than invent a target.
            l_hf = y.new_zeros(())
            if target is not None:
                l_hf = hf_band_match_loss(y_mono, t_mono, sr)

            # Applies to EVERY bucket, deliberately: the artifact is in the
            # waveform and needs no paired audio to detect, so this is the only
            # waveform-domain term the dit bucket gets. Measured on stock, dit
            # decodes already carry 6-9x the grid structure of a real-audio
            # round trip, so that bucket is where it matters most.
            l_patch = y.new_zeros(())
            if cli.lambda_patch > 0:
                l_patch = patch_grid_penalty(y, patch=cli.patch_size)

            loss = (
                cli.lambda_rec * l_rec
                + cli.lambda_cycle * l_cyc
                + cli.lambda_tonal * l_ton
                + cli.lambda_hfband * l_hf
                + cli.lambda_patch * l_patch
            )
            if not torch.isfinite(loss):
                print(
                    f"[train] step {step}: non-finite loss "
                    f"({bucket}, {src_path}), skipping"
                )
                failed += 1
                step += 1
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(lora_params, cli.grad_clip)
            opt.step()

        except Exception as exc:
            failed += 1
            print(
                f"[train] step {step}: {type(exc).__name__} on {bucket} "
                f"{src_path}: {str(exc)[:180]}"
            )
            if failed > 50 and failed > step:
                print("[train] failures outnumber successful steps; aborting")
                return 1
            opt.zero_grad(set_to_none=True)
            continue

        for k, v in (
            ("loss", loss),
            ("rec", l_rec),
            ("cyc", l_cyc),
            ("ton", l_ton),
            ("hf", l_hf),
            ("patch", l_patch),
            ("gnorm", gnorm),
        ):
            running.setdefault(k, []).append(float(v))
        running.setdefault(f"n_{bucket}", []).append(1.0)

        step += 1

        if step % cli.log_every == 0:
            m = {
                k: (sum(v) / len(v))
                for k, v in running.items()
                if not k.startswith("n_")
            }
            counts = {b: len(running.get(f"n_{b}", [])) for b in buckets}
            el = time.time() - t0
            print(
                f"[train] {step:>6}/{cli.steps}  loss {m['loss']:.4f}  "
                f"rec {m['rec']:.4f}  cyc {m['cyc']:.4f}  ton {m['ton']:.3f}  "
                f"hf {m['hf']:.3f}  patch {m['patch']:.2e}  "
                f"|g| {m['gnorm']:.4f}  lr {lr_at(step):.2e}  "
                f"[{counts['real']}r/{counts['drift']}d/{counts['dit']}g]  "
                f"{el / step:.2f}s/step"
            )
            running = {}

        if cli.eval_every > 0 and (step % cli.eval_every == 0 or step == cli.steps):
            ae.decoder.eval()
            ev = eval_ladder(ae, x_eval, sr, ea, cli.eval_iters, onsets=eval_onsets)
            ae.decoder.train()
            print(f"[train] eval @ {step}")
            print(fmt_eval("BASE", base_eval))
            print(fmt_eval(f"step{step}", ev, base_eval))
            write_wav(out_dir / f"eval_step{step:06d}.wav", ev["_final_audio"], sr)
            history.append(
                {
                    "step": step,
                    **{k: v for k, v in ev.items() if not k.startswith("_")},
                }
            )
            (out_dir / "history.json").write_text(
                json.dumps(
                    {
                        "baseline": {
                            k: v for k, v in base_eval.items() if not k.startswith("_")
                        },
                        "config": vars(cli) | {"alpha": alpha},
                        "history": history,
                    },
                    indent=2,
                    default=str,
                )
            )

        if step % cli.save_every == 0 or step == cli.steps:
            ckpt = out_dir / f"decoder_lora_step{step:06d}.safetensors"
            save_lora_safetensors(
                get_lora_state_dict(ae.decoder),
                {
                    "rank": cli.rank,
                    "alpha": alpha,
                    "adapter_type": cli.adapter_type,
                    # The loader resolves this to model.pretransform.model.decoder
                    # (full model) or model.decoder (standalone AE).
                    "target": "decoder",
                    # Provenance. The loader ignores these -- it reads only rank,
                    # alpha, adapter_type, include, exclude and target -- but
                    # without them two checkpoints from different runs are
                    # byte-identical in metadata and there is no way to tell,
                    # from the file alone, which autoencoder it belongs to or
                    # which recipe produced it. An adapter loaded onto the wrong
                    # base model does not error; it just sounds wrong.
                    "base_model": cli.model,
                    "step": step,
                    "trained_with": {
                        "lambda_rec": cli.lambda_rec,
                        "lambda_cycle": cli.lambda_cycle,
                        "lambda_tonal": cli.lambda_tonal,
                        "lambda_patch": cli.lambda_patch,
                        "patch_size": cli.patch_size,
                        "tonality_bands": cli.tonality_bands,
                        "buckets": {
                            "real": cli.w_real,
                            "drift": cli.w_drift,
                            "dit": cli.w_dit,
                        },
                        "crop_seconds": cli.crop_seconds,
                        "lr": cli.lr,
                        "seed": cli.seed,
                    },
                },
                ckpt,
            )
            print(f"[train] wrote {ckpt.name}")

    print(f"\n[train] done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
