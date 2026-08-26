"""
Train a LoRA on SAME's ENCODER, with the decoder frozen, so it finds latents the
decoder can already render.

Why the encoder and not more decoder work
-----------------------------------------
Freeze the decoder and optimise the LATENT itself, and D(z*) is the best audio
that decoder can produce from ANY latent. Run on a DiT-generated clip -- the one
case where a perfect latent provably exists, namely the z_dit that made the
audio -- that reports:

    z_dit reconstructs at        43.2 dB SI-SDR
    E(audio) reconstructs at     14.7 dB SI-SDR, 21.3% away from z_dit

The decoder can render it. The encoder does not find it. On 24 real SAOS chunks
the same probe found recoverable headroom everywhere: min 12.4%, median 21.7%,
max 42.7%. So this is not a wall, it is a gap.

What the training mix follows from
----------------------------------
Recoverable headroom correlates with HF TONALITY at r = +0.91, with HF energy
NEGATIVELY (-0.65 at 16-22k), and with onset density not at all (r = -0.19).
Low-tonality half: 17.9% headroom. High-tonality half: 29.7%.

That inverts the material guidance from the decoder LoRA, where percussion was
the barometer. The decoder INVENTS spurious tonality; the encoder DROPS real
tonality -- harmonics, sustained pitched HF, resonances. Same metric, opposite
defects. Hence --tonality_percentile, which keeps only the more tonal half of
sampled crops by default rather than trusting a folder to be the right material.

Two buckets, because only one of them has ground truth
------------------------------------------------------
  real  Real audio. Reconstruction against the audio itself, plus an ANCHOR to
        the stock encoder's latent, because no correct latent is known here and
        an unanchored audio loss will walk 21-30% (see anchor_loss).

  gen   Audio decoded from a base-DiT latent (--dit_latent_dir). Here the target
        latent is KNOWN: z_dit made the audio, so we can supervise the encoder
        directly onto it rather than merely fence it in. This is the bucket that
        attacks the measured 21.3% gap head-on, and it is also the realistic
        distribution for /continue, which re-encodes audio the model just made.

Both halves of every A/B here are computed under the SAME RNG state. The SAME
encoder adds `mask_noise` = 1e-3 gaussian to its query tokens on every forward
(ungated by train/eval), which is where the "encoder is nondeterministic at 1.5%
relative" measurement comes from. Replaying the draw makes the anchor term and
the eval deltas exact instead of sitting on a 1.5% noise floor -- which matters
when the whole anchor budget is 5%. The startup probe confirms this rather than
assuming it: it reports the floor both ways, and on same-l it is 1.52% unpaired
against 0.000% paired. The nondeterminism is entirely that one draw.

Pick --eval_audio from material the model has NOT trained on and ideally from a
different source than --data_dir; the eval is what decides which checkpoint
ships, and the decoder work showed a single clip cannot resolve a 3.5 dB gap.

The model stays in eval() throughout. The softnorm bottleneck injects noise at
50x the eval scale in train mode, and the decoder's own mask_noise is 0.1; none
of that helps a rank-16 adapter and all of it hides the signal.

Why --data_dir takes several directories
----------------------------------------
An early run trained on a single source (70 guitar/band songs) and learned that
distribution rather than the defect. Measured over 23 clips: on held-out clips of
ITS OWN material it moved steady artifact -2.22 dB, and on five unrelated
electronic and rock sources it moved -0.10. Latent drift said the same -- ~5% on
familiar material against 10-11.7% on unfamiliar, over a 5% budget. The one thing
that did generalise was the gen bucket: the known-z gap fell 17.35% -> 11.78%
regardless of material, because z_dit supervision does not care what the audio
sounds like.

So pass several sources. Note the gate is calibrated per source for the same
reason -- see --tonality_percentile.

Usage:
  uv run python scripts/train_encoder_lora.py \
      --data_dir /patch /saos --max_depth 1 --out_dir /out/enclora \
      --eval_audio /patch/holdout.wav /saos/holdout.wav \
      --dit_latent_dir /out/dit_latents

Keep one source out of training entirely if you want an honest generalisation
probe afterwards; held-out FILES from a trained source are a weaker test than
a source the run never saw.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from fnmatch import fnmatch
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _decoder_lora_losses import (  # noqa: E402
    anchor_loss,
    hf_tonality,
    multires_stft_loss,
    relative_latent_error,
)
from _decoder_lora_eval import (  # noqa: E402
    describe,
    detect_onsets,
    load_audio,
    stft,
    to_mono,
    write_wav,
)
from train_decoder_lora import EvalArgs, list_audio, random_crop  # noqa: E402

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


# --------------------------------------------------------------------------
# paired randomness
# --------------------------------------------------------------------------


# How many crops to draw looking for one that passes the gate before giving up
# and taking the last. Only reached when a source sits almost entirely under its
# own threshold, which a self-calibrating threshold makes unlikely.
_GATE_MAX_TRIES = 12


class TonalityGate:
    """Keep the more tonal share of each source's crops, per source.

    The threshold is RE-DERIVED from the crops training actually draws, rather
    than fixed once from a probe. Two reasons, and the second is the one that
    bites across datasets:

      * The probe sampled every file in a source, including the clips later held
        out for eval, while training draws only from the rest. On the ratatat
        set that gap turned a requested p50 into a measured 66% rejection rate.
      * A fixed dB threshold means a different acceptance RATE on every dataset,
        because it is the material's own tonality distribution that decides how
        much sits above it. `--tonality_percentile 50` should mean "the top half
        of THIS dataset" whatever the dataset is; a number in dB cannot.

    Values are recorded before the accept/reject decision, so the window holds
    the material's true distribution and not just the part that already passed.
    """

    def __init__(self, percentile, seed_values=None, window=512):
        self.pct = percentile
        self.window = window
        self.vals = {k: list(v)[-window:] for k, v in (seed_values or {}).items()}
        self.floor = {
            k: float(np.percentile(v, percentile)) for k, v in self.vals.items() if v
        }
        self.seen = 0
        self.passed = 0

    def observe(self, source, value):
        if self.pct <= 0 or source is None:
            return
        v = self.vals.setdefault(source, [])
        v.append(value)
        if len(v) > self.window:
            del v[: len(v) - self.window]
        # Recomputed periodically rather than per draw: a percentile over the
        # whole window every step is pure overhead beside an encode, and the
        # threshold moves slowly once the window is populated.
        if len(v) >= 16 and len(v) % 32 == 0:
            self.floor[source] = float(np.percentile(v, self.pct))

    def accepts(self, source, value):
        self.seen += 1
        f = self.floor.get(source) if self.pct > 0 else None
        ok = f is None or value >= f
        self.passed += bool(ok)
        return ok

    def acceptance(self):
        return self.passed / self.seen if self.seen else float("nan")

    def describe(self):
        return ", ".join(
            f"{Path(k).name} {v:.2f}dB" for k, v in sorted(self.floor.items())
        )


def rng_snapshot(device):
    """Capture enough RNG state to replay a forward pass exactly."""
    cuda = (
        torch.cuda.get_rng_state(device)
        if torch.device(device).type == "cuda"
        else None
    )
    return torch.get_rng_state(), cuda


def rng_restore(state, device):
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


# --------------------------------------------------------------------------
# adapter sanity
# --------------------------------------------------------------------------


@torch.no_grad()
def assert_adapter_live(ae, x, lora_params, device):
    """Prove the encoder LoRA CHANGES OUTPUT, and measure the noise floor.

    LoRA B is zero-initialised, so at step 0 the adapter is a mathematical
    no-op and "does the output change?" cannot be asked directly. Perturb the
    parameters, measure, restore. The same probe run WITHOUT perturbing gives
    the encoder's own noise floor -- twice, once with the mask noise replayed
    and once without -- which is the number every later delta has to beat.

    Silent no-ops are the recurring failure mode in this project (four in one
    session: enable_grad defaulting False, strict=False swallowing a key
    mismatch, lora_configs not setting strength, the API worker zeroing
    indices). Asserting that an adapter attached is not the same as asserting
    it does anything.
    """
    snap = rng_snapshot(device)
    z_a = ae.encode(x)

    rng_restore(snap, device)
    z_paired = ae.encode(x)
    floor_paired = float(relative_latent_error(z_paired, z_a))

    z_free = ae.encode(x)
    floor_free = float(relative_latent_error(z_free, z_a))

    saved = [p.detach().clone() for p in lora_params]
    for p in lora_params:
        p.add_(torch.randn_like(p) * 0.02)
    rng_restore(snap, device)
    z_pert = ae.encode(x)
    moved = float(relative_latent_error(z_pert, z_a))
    for p, s in zip(lora_params, saved):
        p.copy_(s)

    print(
        f"[train] adapter probe: perturbed LoRA moves the latent "
        f"{100 * moved:.2f}%; noise floor {100 * floor_paired:.3f}% paired / "
        f"{100 * floor_free:.2f}% unpaired"
    )
    if moved < 10 * max(floor_paired, 1e-6):
        raise SystemExit(
            "[train] ABORT: perturbing every LoRA parameter barely moved the "
            "latent. The adapter is attached but not in the forward path."
        )
    if floor_paired > 0.2 * floor_free and floor_free > 1e-4:
        print(
            "[train] WARNING: replaying the RNG did not remove the encoder's "
            "run-to-run variation, so something else in the encode path is "
            "nondeterministic. The anchor term will carry that as noise."
        )
    return floor_paired, floor_free


# --------------------------------------------------------------------------
# eval: paired A/B on held-out material
# --------------------------------------------------------------------------


@torch.no_grad()
def eval_clip(ae, x, sr, ea, onsets, device):
    """Reconstruct one held-out clip with the adapter off and on, same noise.

    Both passes see identical encoder mask noise and identical decoder mask
    noise, so every difference reported is the adapter and nothing else.
    """
    enc_snap = rng_snapshot(device)
    set_lora_strength(ae.encoder, 0.0)
    z_off = ae.encode(x)
    dec_snap = rng_snapshot(device)
    y_off = ae.decode(z_off).float()

    rng_restore(enc_snap, device)
    set_lora_strength(ae.encoder, 1.0)
    z_on = ae.encode(x)
    rng_restore(dec_snap, device)
    y_on = ae.decode(z_on).float()

    tgt = x[0].float().cpu()
    off = describe(y_off[0].float().cpu(), tgt, sr, ea, onsets)
    on = describe(y_on[0].float().cpu(), tgt, sr, ea, onsets)
    return {
        "off": off,
        "on": on,
        "latent_drift_pct": 100 * float(relative_latent_error(z_on, z_off)),
        "_audio_on": y_on[0].float().cpu(),
        "_audio_off": y_off[0].float().cpu(),
    }


@torch.no_grad()
def eval_known_latent(ae, z_dit, device):
    """The headline number: how far is E(D(z_dit)) from the latent that made it?

    This is the one place in the whole pipeline with ground truth. Stock encoder
    measured 21.3%. Anything the adapter does here is unambiguous -- there is no
    argument about whether the target is the right target.
    """
    set_lora_strength(ae.encoder, 0.0)
    x = ae.decode(z_dit).float()
    snap = rng_snapshot(device)
    z_off = ae.encode(x)
    rng_restore(snap, device)
    set_lora_strength(ae.encoder, 1.0)
    z_on = ae.encode(x)
    return (
        100 * float(relative_latent_error(z_off, z_dit)),
        100 * float(relative_latent_error(z_on, z_dit)),
    )


def aggregate(clips, tags=None):
    """Mean over held-out clips, plus the per-clip deltas that produced it.

    The mean alone cannot say whether an adapter improved the material it
    trained on while degrading everything else, which is the whole reason the
    eval set spans two distributions. Keeping the per-clip rows means that
    question is answerable from history.json afterwards instead of needing the
    checkpoints re-scored.
    """
    keys = (
        "stft_loss",
        "si_sdr_db",
        "tonality_p95",
        "onset_artifact_db",
        "steady_artifact_db",
        "band_8_12k",
        "band_12_16k",
        "band_16_22k",
    )
    out = {}
    for side in ("off", "on"):
        for k in keys:
            out[f"{side}_{k}"] = float(np.mean([c[side][k] for c in clips]))
    out["latent_drift_pct"] = float(np.mean([c["latent_drift_pct"] for c in clips]))
    out["per_clip"] = [
        {
            "clip": (tags[i] if tags else str(i)),
            "drift_pct": c["latent_drift_pct"],
            **{k: c["on"][k] - c["off"][k] for k in keys},
        }
        for i, c in enumerate(clips)
    ]
    return out


def fmt_eval(tag, a):
    # BOTH absolute artifact sides. Printing onset alone was enough to make a
    # healthy run look like a regression at step 500 of an early run: onset had moved
    # -0.20 dB while steady had moved -0.49, and only the JSON showed it.
    return (
        f"  {tag:>10}  stft {a['on_stft_loss']:.4f} "
        f"({a['on_stft_loss'] - a['off_stft_loss']:+.4f})"
        f"   si-sdr {a['on_si_sdr_db']:6.2f} "
        f"({a['on_si_sdr_db'] - a['off_si_sdr_db']:+.2f})"
        f"   onset {a['on_onset_artifact_db']:6.2f} "
        f"({a['on_onset_artifact_db'] - a['off_onset_artifact_db']:+.2f})"
        f"   steady {a['on_steady_artifact_db']:6.2f} "
        f"({a['on_steady_artifact_db'] - a['off_steady_artifact_db']:+.2f})"
        f"   16-22k {a['on_band_16_22k']:6.1f} "
        f"({a['on_band_16_22k'] - a['off_band_16_22k']:+.1f})"
        f"   drift {a['latent_drift_pct']:5.2f}%"
    )


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Several sources, because an early run learned its training distribution instead of
    # the defect: on held-out clips from its own material it moved steady
    # artifact -2.22 dB, and on five unrelated electronic and rock sources it
    # moved -0.10. Latent drift told the same story -- ~5% on familiar material,
    # 10-11.7% on unfamiliar, against a 5% budget.
    p.add_argument(
        "--data_dir",
        required=True,
        nargs="+",
        help="dirs of real audio (recursive; see --max_depth). "
        "Files are sampled uniformly across the union.",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--eval_audio",
        nargs="+",
        default=None,
        help="held-out clips; several, because a single clip cannot "
        "resolve a small gap (default: first 3 files)",
    )
    # A dataset directory is not a dataset. ~/the_patch_dataset holds 70 real
    # songs at the top level and 4300 derived files underneath: infinigram
    # expansions (generated, tonality median 5.02 -- the worst material in
    # either folder) and __source.wav files, which are NOT sources but
    # pyrubberband time-stretches of the top-level songs, made as varied input
    # for melodyflow. Phase-vocoder stretching leaves its own signature in the
    # exact band this training targets. Sampling the tree uniformly would spend
    # the run on generated and resampled audio, so say which files you mean.
    p.add_argument(
        "--max_depth",
        type=int,
        default=0,
        help="only files at most this many path components below "
        "--data_dir; 1 = the directory itself, 0 = no limit",
    )
    p.add_argument(
        "--include_glob",
        default=None,
        help="keep only training files whose BASENAME matches this "
        "glob, e.g. '*__source.wav'",
    )
    p.add_argument(
        "--exclude_glob",
        default=None,
        help="drop training files whose basename matches this glob",
    )
    p.add_argument(
        "--dit_latent_dir",
        default=None,
        help="dir of .npy base-DiT latents (see gen_dit_latents.py). "
        "Without it the gen bucket -- the only one with a known "
        "correct latent -- is dropped.",
    )
    p.add_argument("--model", default="same-l", choices=["same-l", "same-s"])

    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha; default = rank (scaling 1.0)",
    )
    p.add_argument("--adapter_type", default="lora")

    # Both decoder-LoRA runs peaked at roughly a third of their cosine schedule
    # and degraded after, so the schedule is set to ~3x the expected useful
    # training rather than to a step count anyone believes in.
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--crop_seconds", type=float, default=10.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--w_real", type=float, default=0.5)
    p.add_argument("--w_gen", type=float, default=0.5)

    # Keep only crops in the top (100 - percentile)% by HF tonality, because
    # recoverable headroom tracks tonality at r = +0.91. The threshold is
    # measured at startup rather than assumed, and measured PER SOURCE DIR:
    # tonality is not comparable across material types, so one global threshold
    # over a mixed pool would keep the tonal source and reject the others --
    # which is that distribution skew wearing a different hat. 0 disables.
    p.add_argument("--tonality_percentile", type=float, default=50.0)
    p.add_argument("--tonality_probe_crops", type=int, default=64)

    p.add_argument("--lambda_rec", type=float, default=1.0)
    p.add_argument("--lambda_anchor", type=float, default=10.0)
    p.add_argument(
        "--anchor_budget",
        type=float,
        default=0.05,
        help="relative latent movement allowed on the real bucket "
        "before the anchor pushes back. The DiT samples its "
        "latents from a space E(x) is supposed to land in, and "
        "the optimal latents sit 21-30%% out, so an unanchored "
        "run drifts far enough to break init_audio/continue "
        "while looking fine on reconstruction.",
    )
    p.add_argument(
        "--lambda_sup",
        type=float,
        default=10.0,
        help="weight on latent supervision toward z_dit (gen bucket)",
    )

    p.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="run the eval every N steps; 0 = never, which also skips the "
        "step-0 baseline there is then nothing to compare against",
    )
    p.add_argument("--eval_seconds", type=float, default=20.0)
    p.add_argument(
        "--eval_offset",
        type=float,
        default=20.0,
        help="seconds into each eval file to start; clamped down "
        "for files that are shorter than that",
    )
    p.add_argument("--save_every", type=int, default=500)
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

    ea = EvalArgs(
        n_fft=2048,
        hop=512,
        hf_lo=6000.0,
        hf_hi=20000.0,
        tonality_lo=6000.0,
        tonality_hi=16000.0,
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
        ae.encoder,
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
    lora_params = list(get_lora_params(ae.encoder))
    for prm in lora_params:
        prm.requires_grad_(True)
    n_lora = sum(p.numel() for p in lora_params)
    n_enc = sum(p.numel() for p in ae.encoder.parameters())
    print(
        f"[train] encoder LoRA: rank={cli.rank} alpha={alpha} "
        f"type={cli.adapter_type} -> {len(lora_params)} tensors, "
        f"{n_lora / 1e6:.2f}M trainable ({100 * n_lora / max(n_enc, 1):.2f}% of the "
        f"encoder)"
    )
    assert sum(p.numel() for p in ae.parameters() if p.requires_grad) == n_lora, (
        "something other than the encoder LoRA is trainable"
    )

    # Without this, pretransform.decode() runs under no_grad and the audio loss
    # never reaches the encoder at all -- the loss prints plausible numbers and
    # trains nothing.
    if ae.pretransform is not None:
        ae.pretransform.enable_grad = True
    # eval() everywhere, deliberately: the softnorm bottleneck injects noise at
    # 5e-2 in train mode against 1e-3 in eval, and its running_std update is
    # gated on training too. Neither belongs in a frozen-base LoRA run.
    ae.eval()

    # ---- data -----------------------------------------------------------
    eval_audio = list(cli.eval_audio or [])
    # Keep the files grouped by source dir: the tonality gate is calibrated per
    # source, and the startup summary should show the mix rather than one total
    # that could be 95% one thing.
    by_source = {}
    for d in cli.data_dir:
        root = Path(d)
        got = list_audio(d, exclude=tuple(eval_audio))
        if cli.max_depth:
            got = [
                f for f in got if len(Path(f).relative_to(root).parts) <= cli.max_depth
            ]
        if cli.include_glob:
            got = [f for f in got if fnmatch(os.path.basename(f), cli.include_glob)]
        if cli.exclude_glob:
            got = [f for f in got if not fnmatch(os.path.basename(f), cli.exclude_glob)]
        if got:
            by_source[d] = got
        else:
            print(
                f"[train] WARNING: {d} contributed no files "
                f"(max_depth={cli.max_depth} include={cli.include_glob!r})"
            )
    files = [f for got in by_source.values() for f in got]
    if not files:
        print(f"[train] no usable audio under {cli.data_dir}")
        return 1
    if not eval_audio:
        eval_audio, files = files[:3], files[3:]
    print(
        f"[train] {len(files)} training files across {len(by_source)} "
        f"source(s); eval on {len(eval_audio)} held-out clips"
    )
    for d, got in by_source.items():
        print(f"[train]   {len(got):>5} from {d}")

    dit_latents, dit_holdout = [], None
    if cli.dit_latent_dir:
        dit_latents = sorted(str(q) for q in Path(cli.dit_latent_dir).glob("*.npy"))
        if dit_latents:
            # Hold one out. The known-latent gap is the headline metric, so it
            # cannot be measured on a latent the encoder trained against.
            dit_holdout, dit_latents = dit_latents[-1], dit_latents[:-1]
        print(
            f"[train] {len(dit_latents)} base-DiT latents "
            f"(+1 held out: {Path(dit_holdout).name if dit_holdout else '-'})"
        )
    if not dit_latents and cli.w_gen > 0:
        print(
            "[train] NOTE: --w_gen set but no usable --dit_latent_dir; "
            "dropping the gen bucket. The run then has NO ground-truth latent "
            "supervision and leans entirely on the anchor."
        )

    buckets = ["real", "gen"]
    weights = [cli.w_real, cli.w_gen if dit_latents else 0.0]
    tot = sum(weights)
    if tot <= 0:
        print("[train] both bucket weights are zero")
        return 1
    weights = [w / tot for w in weights]
    print(
        "[train] bucket mix: "
        + ", ".join(f"{b}={w:.2f}" for b, w in zip(buckets, weights))
    )

    # ---- tonality gate ---------------------------------------------------
    # Recoverable headroom tracks HF tonality at r = +0.91, so which crops we
    # train on matters more than how many. Calibrated PER SOURCE: tonality is
    # not comparable across material types (patch songs sit ~1 dB above SAOS
    # chunks on the same measure), so a single threshold over a mixed pool
    # would keep most of the tonal source and reject most of the others. That
    # is the same distribution skew again, arriving through the gate instead of
    # through the folder.
    source_of = {f: d for d, got in by_source.items() for f in got}
    # Probe the TRAINING pool only. by_source still lists every file in each
    # source, but the eval clips were carved out of `files` above, so probing
    # `got` measured material training never sees -- which is one of the two
    # reasons a requested p50 realised as 66% rejection. The gate re-derives its
    # threshold from real draws afterwards; this only has to start it close.
    train_files = set(files)
    probe_source = {
        d: [f for f in got if f in train_files] for d, got in by_source.items()
    }
    probe_source = {d: got for d, got in probe_source.items() if got}
    seed_vals = {}
    if cli.tonality_percentile > 0 and probe_source:
        per_source_probe = max(8, cli.tonality_probe_crops // len(probe_source))
        for d, got in probe_source.items():
            vals = []
            tries = 0
            while len(vals) < per_source_probe and tries < 10 * per_source_probe:
                tries += 1
                w = random_crop(
                    rng.choice(got), sr, cli.crop_seconds, rng, multiple=ds_ratio
                )
                if w is None:
                    continue
                vals.append(float(hf_tonality(to_mono(w[0]).to(device), sr)))
            if vals:
                seed_vals[d] = vals
                print(
                    f"[train] tonality gate [{Path(d).name}]: {len(vals)} "
                    f"crops, min {min(vals):.2f} / med {np.median(vals):.2f} "
                    f"/ max {max(vals):.2f} dB -> start above "
                    f"{np.percentile(vals, cli.tonality_percentile):.2f} "
                    f"(p{cli.tonality_percentile:g}, re-derived as it runs)"
                )
    gate = TonalityGate(cli.tonality_percentile, seed_vals)

    # ---- held-out eval material -----------------------------------------
    eval_clips = []
    for path in eval_audio:
        try:
            # Skip the intro if the file is long enough to have one, otherwise
            # take it from the top. A fixed 20 s offset silently yields nothing
            # on a dataset of 12 s chunks, which reads as "no usable eval clips"
            # rather than as an offset problem.
            import torchaudio

            info = torchaudio.info(path)
            total = info.num_frames / float(info.sample_rate)
            offset = min(cli.eval_offset, max(0.0, total - cli.eval_seconds))
            xw = load_audio(path, sr, seconds=cli.eval_seconds, offset=offset)
        except Exception as exc:
            print(f"[train] eval clip {path}: {type(exc).__name__} {exc}")
            continue
        n = (xw.shape[-1] // ds_ratio) * ds_ratio
        if n < ds_ratio:
            print(
                f"[train] eval clip {path}: too short "
                f"({xw.shape[-1]} samples) -- skipped"
            )
            continue
        xw = xw[:, :n].unsqueeze(0).to(device)
        onsets = detect_onsets(
            stft(to_mono(xw[0].float().cpu()), ea.n_fft, ea.hop).abs(), sr, ea.hop
        )
        eval_clips.append((Path(path).stem[:40], xw, onsets))
    if not eval_clips:
        print("[train] no usable eval clips")
        return 1

    z_holdout = None
    if dit_holdout:
        arr = np.load(dit_holdout)
        z_holdout = torch.from_numpy(arr).float().to(device)
        if z_holdout.dim() == 2:
            z_holdout = z_holdout.unsqueeze(0)
        n_lat = max(1, int(cli.eval_seconds * sr / ds_ratio))
        z_holdout = z_holdout[..., :n_lat].contiguous()

    # ---- adapter sanity + noise floor ------------------------------------
    set_lora_strength(ae.encoder, 1.0)
    floor_paired, floor_free = assert_adapter_live(
        ae, eval_clips[0][1], lora_params, device
    )

    def run_eval(step):
        # Same noise for every eval in the run, not just within one A/B. The
        # off-row is then bit-stable across steps, so a checkpoint comparison
        # is a comparison and not a re-roll of the decoder's mask noise. The
        # training stream is put back afterwards so eval does not perturb it.
        train_state = rng_snapshot(device)
        torch.manual_seed(cli.seed + 9973)
        clips = [
            eval_clip(ae, xw, sr, ea, onsets, device) for _, xw, onsets in eval_clips
        ]
        agg = aggregate(clips, tags=[t for t, _, _ in eval_clips])
        if z_holdout is not None:
            g_off, g_on = eval_known_latent(ae, z_holdout, device)
            agg["known_gap_off_pct"], agg["known_gap_on_pct"] = g_off, g_on
        set_lora_strength(ae.encoder, 1.0)
        for (tag, _, _), c in zip(eval_clips, clips):
            write_wav(out_dir / f"eval_{tag}_step{step:06d}.wav", c["_audio_on"], sr)
            if step == 0:
                write_wav(out_dir / f"eval_{tag}_stock.wav", c["_audio_off"], sr)
        rng_restore(train_state, device)
        return agg

    # Every eval prints against this, so with --eval_every 0 there is nothing to
    # compare to and the baseline pass is pure cost -- skip it with the rest.
    base = None
    if cli.eval_every > 0:
        base = run_eval(0)
        print(
            "\n[train] baseline (adapter off vs off is a no-op; this is the reference row):"
        )
        print(
            f"  {'STOCK':>10}  stft {base['off_stft_loss']:.4f}"
            f"   si-sdr {base['off_si_sdr_db']:6.2f}"
            f"   onset {base['off_onset_artifact_db']:6.2f}"
            f"   16-22k {base['off_band_16_22k']:6.1f}"
        )
        if "known_gap_off_pct" in base:
            print(
                f"  {'KNOWN':>10}  E(D(z_dit)) sits {base['known_gap_off_pct']:.2f}% "
                f"from z_dit -- this is the number to move"
            )
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
    rejected = 0
    gate_exhausted = 0
    failed = 0
    grad_checked = False

    while step < cli.steps:
        bucket = rng.choices(buckets, weights=weights, k=1)[0]
        src_path = None
        try:
            # ---- build (input audio, reference latent) -------------------
            if bucket == "gen":
                src_path = rng.choice(dit_latents)
                arr = np.load(src_path)
                z_ref = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                if z_ref.dim() == 2:
                    z_ref = z_ref.unsqueeze(0)
                n_lat = max(1, int(cli.crop_seconds * sr / ds_ratio))
                if z_ref.shape[-1] > n_lat:
                    s = rng.randrange(0, z_ref.shape[-1] - n_lat + 1)
                    z_ref = z_ref[..., s : s + n_lat]
                z_ref = z_ref.contiguous()
                with torch.no_grad():
                    # Decoded fresh every visit: the decoder's own mask_noise
                    # makes each decode of z_dit a different rendering, and
                    # mapping ALL of them back to z_dit is exactly the
                    # invariance /continue needs.
                    x = ae.decode(z_ref).float()
                snap = None
            else:
                # Retry inside the REAL bucket, never by falling back to the
                # loop head. A `continue` there re-draws the BUCKET, and since
                # the gen bucket is never gated, every rejection became a fresh
                # coin flip that gen was likelier to survive -- so a requested
                # 50/50 mix arrived as 25/75 on this dataset, and as something
                # else on the next one. That silently made two datasets two
                # different experiments.
                x = None
                for _ in range(_GATE_MAX_TRIES):
                    src_path = rng.choice(files)
                    w = random_crop(
                        src_path, sr, cli.crop_seconds, rng, multiple=ds_ratio
                    )
                    if w is None:
                        skipped += 1
                        if step == 0 and skipped > 200:
                            print(
                                "[train] 200+ unusable crops before a single "
                                "step; check --data_dir and --crop_seconds"
                            )
                            return 1
                        continue
                    cand = w.to(device)
                    src = source_of.get(src_path)
                    ton = float(hf_tonality(to_mono(cand[0]), sr))
                    # Measured on EVERY drawn crop, before accept/reject, so the
                    # window sees the material's true distribution rather than
                    # the part that already passed. That is what lets the
                    # threshold be re-derived from it without censoring bias.
                    gate.observe(src, ton)
                    if gate.accepts(src, ton):
                        x = cand
                        break
                    rejected += 1
                if x is None:
                    # Every try in a row was gated out. Taking the last one beats
                    # dropping the step, which would put us back to skewing the
                    # mix by exactly the route this loop exists to close. A
                    # self-calibrating threshold makes this rare by
                    # construction; it stays as a floor for pathological input.
                    x = cand
                    gate_exhausted += 1
                with torch.no_grad():
                    snap = rng_snapshot(device)
                    set_lora_strength(ae.encoder, 0.0)
                    z_ref = ae.encode(x)
                    set_lora_strength(ae.encoder, 1.0)

            # ---- forward / backward -------------------------------------
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)

            if snap is not None:
                # Same mask noise as the reference encode, so the anchor
                # measures the WEIGHTS and not a 1.5% random draw.
                rng_restore(snap, device)
            z_new = ae.encode(x)
            y = ae.decode(z_new)

            l_rec = multires_stft_loss(y, x, sr=sr)
            l_anc = y.new_zeros(())
            l_sup = y.new_zeros(())
            if bucket == "gen":
                l_sup = anchor_loss(z_new, z_ref, budget=0.0)
            else:
                l_anc = anchor_loss(z_new, z_ref, budget=cli.anchor_budget)

            loss = (
                cli.lambda_rec * l_rec
                + cli.lambda_anchor * l_anc
                + cli.lambda_sup * l_sup
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

            if not grad_checked:
                gsum = sum(
                    float(p.grad.abs().sum()) for p in lora_params if p.grad is not None
                )
                if gsum == 0.0:
                    print(
                        "[train] ABORT: no gradient reached the LoRA "
                        "parameters on the first step. The audio loss is "
                        "detached from the encoder (check "
                        "pretransform.enable_grad)."
                    )
                    return 1
                grad_checked = True

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
            set_lora_strength(ae.encoder, 1.0)
            opt.zero_grad(set_to_none=True)
            continue

        # Per-bucket, not per-step: averaging the anchor over gen steps (where it
        # is structurally 0) and the supervision over real steps would halve both
        # numbers and make them mean nothing.
        for k, v in (("loss", loss), ("rec", l_rec), ("gnorm", gnorm)):
            running.setdefault(k, []).append(float(v))
        if bucket == "gen":
            running.setdefault("sup", []).append(float(l_sup))
        else:
            running.setdefault("anc", []).append(float(l_anc))
            # Where the latent actually sits, not just the hinge. Below the
            # budget the anchor reads 0.0000 forever, which is
            # indistinguishable from a term that was never wired up.
            with torch.no_grad():
                running.setdefault("drift", []).append(
                    100 * float(relative_latent_error(z_new, z_ref))
                )
        running.setdefault(f"n_{bucket}", []).append(1.0)

        step += 1

        if step % cli.log_every == 0:
            nan = float("nan")
            m = {
                k: (sum(v) / len(v))
                for k, v in running.items()
                if not k.startswith("n_")
            }
            counts = {b: len(running.get(f"n_{b}", [])) for b in buckets}
            el = time.time() - t0
            print(
                f"[train] {step:>6}/{cli.steps}  loss {m['loss']:.4f}  "
                f"rec {m['rec']:.4f}  anchor {m.get('anc', nan):.4f}  "
                f"sup {m.get('sup', nan):.4f}  "
                f"drift {m.get('drift', nan):5.2f}%  "
                f"|g| {m['gnorm']:.4f}  "
                f"lr {lr_at(step):.2e}  "
                f"[{counts['real']}r/{counts['gen']}g]  "
                # The realised bucket mix and gate acceptance, because both
                # used to be inferable only by squinting at the r/g counter.
                f"acc {100 * gate.acceptance():.0f}%  "
                f"{el / step:.2f}s/step"
            )
            running = {}

        if cli.eval_every > 0 and (step % cli.eval_every == 0 or step == cli.steps):
            ev = run_eval(step)
            print(
                f"[train] eval @ {step}   (stock: stft "
                f"{ev['off_stft_loss']:.4f}  si-sdr {ev['off_si_sdr_db']:.2f})"
            )
            print(fmt_eval(f"step{step}", ev))
            if "known_gap_on_pct" in ev:
                print(
                    f"  {'known z':>10}  E(D(z_dit)) -> z_dit  "
                    f"{ev['known_gap_off_pct']:.2f}% stock  ->  "
                    f"{ev['known_gap_on_pct']:.2f}% adapted  "
                    f"({ev['known_gap_on_pct'] - ev['known_gap_off_pct']:+.2f})"
                )
            history.append({"step": step, **ev})
            (out_dir / "history.json").write_text(
                json.dumps(
                    {
                        "config": vars(cli) | {"alpha": alpha},
                        "noise_floor_paired_pct": 100 * floor_paired,
                        "noise_floor_unpaired_pct": 100 * floor_free,
                        "baseline": base,
                        "history": history,
                    },
                    indent=2,
                    default=str,
                )
            )

        if step % cli.save_every == 0 or step == cli.steps:
            ckpt = out_dir / f"encoder_lora_step{step:06d}.safetensors"
            save_lora_safetensors(
                get_lora_state_dict(ae.encoder),
                {
                    "rank": cli.rank,
                    "alpha": alpha,
                    "adapter_type": cli.adapter_type,
                    # Resolves to model.pretransform.model.encoder (full model)
                    # or model.encoder (standalone AE).
                    "target": "encoder",
                    # Provenance. The loader ignores these, but without them a
                    # checkpoint cannot say which autoencoder it belongs to or
                    # which recipe produced it -- and an adapter loaded onto the
                    # wrong base model does not error, it just encodes wrong.
                    "base_model": cli.model,
                    "name": Path(cli.out_dir).name,
                    "step": step,
                    "trained_with": {
                        "anchor_budget": cli.anchor_budget,
                        "tonality_percentile": cli.tonality_percentile,
                        "crop_seconds": cli.crop_seconds,
                        "lr": cli.lr,
                        "seed": cli.seed,
                        "steps_configured": cli.steps,
                    },
                },
                ckpt,
                # fp32, against the library's fp16 default. Rounding a random
                # rank-8 encoder adapter to fp16 moves the latent by 25-45% of
                # the adapter's own effect (measured in memory, no save or load
                # involved). A trained rank-16 DECODER adapter only loses 0.8%
                # the same way, so that is a worst case rather than a
                # prediction -- but the eval below is what picks a checkpoint,
                # and it is worth 11 MB instead of 5.6 to know the file ships
                # the adapter that was ranked.
                dtype=torch.float32,
            )
            print(f"[train] wrote {ckpt.name}")

    print(
        f"\n[train] done in {(time.time() - t0) / 60:.1f} min "
        f"(gate: {100 * gate.acceptance():.1f}% of drawn crops accepted vs "
        f"p{cli.tonality_percentile:g} requested; floors {gate.describe()}; "
        f"{rejected} rejected, {gate_exhausted} steps took an unpassed crop; "
        f"{skipped} unusable, {failed} failed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
