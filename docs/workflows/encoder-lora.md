# Encoder LoRA

A LoRA on the autoencoder's **encoder**, with the decoder frozen. It changes how
audio becomes latents, not how latents become audio.

That makes it the mirror of the [decoder LoRA](decoder-lora.md), and it applies
to a disjoint set of paths. An encoder adapter is **inert during plain
generation** — the DiT produces latents directly and the encoder is uninvolved.
It affects only paths that encode existing audio: `init_audio`, continuation,
transform, and **pre-encoding a dataset for LoRA training**.

That last one is the workflow this document is mainly about.

> **Scope.** Everything measured here is SAME-L. The mechanism is not specific to
> it, but no number below has been reproduced on SAME-S. See
> [On SAME-S](#on-same-s).

---

## The gap it targets

Freeze the decoder and optimise the *latent* directly, and `D(z*)` is the best
audio that decoder can produce from any latent. Run that on a DiT-generated clip
— the one case where a perfect latent provably exists, namely the `z_dit` that
made the audio — and:

| | SI-SDR |
|---|---|
| `z_dit` (the latent that made the clip) | **43.2 dB** |
| `E(audio)` (what the encoder finds) | **14.7 dB**, 21.3 % away from `z_dit` |

The decoder can render it. The encoder does not find it. On 24 real audio chunks
the same probe found recoverable headroom everywhere: min 12.4 %, median 21.7 %,
max 42.7 %.

**So this is a gap, not a wall** — and it is the encoder's half of it.

### What material to train on

Recoverable headroom correlates with **HF tonality** at r = +0.91, negatively
with HF energy (−0.65 at 16–22 kHz), and not at all with onset density
(r = −0.19). The low-tonality half of a sample averages 17.9 % headroom; the
high-tonality half averages 29.7 %.

That **inverts** the material guidance from the decoder LoRA, where percussion is
the barometer. The decoder *invents* spurious tonality; the encoder *drops* real
tonality — harmonics, sustained pitched HF, resonances. Same metric, opposite
defects. Hence `--tonality_percentile`, which keeps only the more tonal half of
sampled crops rather than trusting a folder to be the right material.

### Use several sources

An early run trained on a single source (70 guitar/band songs) and learned that
distribution rather than the defect. Over 23 clips it moved steady artifact
−2.22 dB on held-out clips of *its own* material and −0.10 dB on five unrelated
sources. Latent drift agreed: ~5 % on familiar material against 10–11.7 % on
unfamiliar, against a 5 % budget.

`--data_dir` takes several directories for this reason, and the tonality gate is
calibrated per source so one loud source cannot dominate it.

Keep one source out of training entirely if you want an honest generalisation
probe. Held-out *files* from a trained source are a weaker test than a source the
run never saw.

---

## The pre-encoding workflow

This is the part worth understanding, because it changes what "overfitting"
means.

A DiT LoRA trains on **pre-encoded latents**. Those latents are the training
targets — whatever the encoder produces is what the LoRA learns to reproduce. If
the encoder is dropping recoverable tonal detail, every target carries that loss
and no amount of DiT training gets it back.

Pre-encoding is a **one-time pass over a fixed corpus**. That is the key
property. An encoder adapter trained on the very corpus you are about to
pre-encode is not overfitting in any way that costs you, because there is no
other data it will ever be asked to handle — the corpus is closed, the pass runs
once, and the result is a set of latents. In-distribution is the *whole job*.

```
      your corpus  ──┬─────────────────────────────►  train_encoder_lora.py
                     │                                        │
                     │                                        ▼
                     │                                  encoder.safetensors
                     │                                        │
                     ▼                                        ▼
              pre_encode_dataset.py  ◄────── --encoder_lora ───┘
                     │
                     ▼
                  latents  ──────────────────────►  train_lora.py (DiT)
```

### 1. Train the encoder adapter on the corpus you are about to encode

```bash
uv run python scripts/train_encoder_lora.py \
    --model same-l \
    --data_dir /path/to/corpus --max_depth 1 \
    --eval_audio /path/to/holdout.wav \
    --out_dir out/enclora
```

### 2. Pre-encode through it

```bash
uv run python scripts/pre_encode_dataset.py \
    --data_dir /path/to/corpus \
    --output_path out/latents \
    --encoder_lora out/enclora/encoder_lora_step002000.safetensors
```

### 3. Train the DiT LoRA on those latents, unchanged

Nothing downstream needs to know. The latents are ordinary latents.

### Latents from different encoders do not mix

Latents written with an adapter are not interchangeable with stock ones. Mixing
them in one dataset gives an incoherent corpus, and nothing will tell you — the
files are the same shape and the training loss looks normal. Pre-encode a corpus
one way or the other, and re-encode from scratch when you change the adapter.

**Never compare the flow-matching losses of two DiT LoRAs trained on differently
pre-encoded latents.** They are computed against different targets, so the
comparison is meaningless in a way that looks entirely reasonable on a chart.
Compare renders.

---

## Reproducibility of the pre-encode pass

Two `SampleDataset` defaults make a one-shot pre-encode non-deterministic, and
both are now off by default in `pre_encode_dataset.py`.

**Polarity.** `SampleDataset` hardcodes a `PhaseFlipper(p=0.5)` that inverts
polarity on a coin flip. During training that is a fair augmentation — a clip is
seen many times, so it is seen both ways. In a one-shot pre-encode each track is
encoded exactly once, so the flip adds no diversity and only makes the output
irreproducible. Polarity is inaudible, but the encoder is **nonlinear**, so
`z(-x)` is essentially uncorrelated with `z(x)`:

| | latent difference between two pre-encodes of the same corpus |
|---|---|
| tracks that happened to flip | ~141 % |
| tracks that did not | ~6 % |
| **corpus average** | **78 %** |
| encoder noise floor | 1.31 % |

That silently destroys any controlled comparison between two sets of latents.
`--phase_flip` restores the old behaviour if you want it.

**Crop offset.** `random_crop` defaults on, so any track longer than
`--sample_size` is cropped from a random offset and two passes encode different
audio. A one-shot pre-encode has no use for that — DiT LoRA training does its own
cropping in latent space, which is where the augmentation belongs. `--random_crop`
restores it.

---

## Keeping the encoder inside the DiT's latent space

This is the constraint that makes the whole thing safe, and it is why
`anchor_loss` exists.

Latent inversion showed the optimal latent sits 21–30 % away from `E(x)` (mean
25.8 %). A pure audio loss against a frozen decoder will happily walk the encoder
that far. **Generation would survive it** — the DiT feeds the decoder directly
and the encoder is uninvolved. But `init_audio`, continuation and transform all
encode real audio and hand the result to the DiT, which is exactly the
distribution shift that would break the paths this work is for.

`--anchor_budget` makes the term a hinge rather than a spring: below that
relative error there is no gradient at all, so reconstruction is free to use the
whole allowance, and above it the term pushes back. That lets the budget be
stated as something defensible — "stay within 5 % of the stock encoder" — rather
than emerging from a weight.

One subtlety: the reference latent must be computed under the **same RNG state**
as the adapted one. The SAME encoder adds `mask_noise` to its learned query
tokens on every forward, ungated by train/eval, so drawing the reference
separately puts ~1.5 % of pure noise into the anchor term — a third of a typical
budget.

---

## On SAME-S

Untested. The hypothesis is that the benefit should be **larger** on SAME-S, not
smaller: it is the more compressed model, so it has more to lose in the encode
step and more recoverable headroom to find. Nothing here has measured that.

Two things to hold in mind if you try it:

- **There is far less adapter to work with.** At rank 16 the SAME-L decoder
  exposes 49 `nn.Linear` layers / 5.63M params; SAME-S exposes 25 / 1.42M. The
  encoder side is smaller again. Loss weights tuned on SAME-L are not defaults.
- **Do not screen results on spectral flatness.** The tonality detector is
  inverted on SAME-S material — the clip that sounds squeakiest scores lowest.
  Judge by ear.

Since an encoder adapter is inert during plain generation, the honest test is a
transform or a continuation, not a text-to-audio render.

---

## Storage dtype

Encoder adapters are written **fp32**, against the library's fp16 default for
LoRA checkpoints. fp16's ~5e-4 relative precision can be amplified by the
encoder's layer stack: a *random* rank-8 encoder adapter rounded to fp16 — in
memory, no save or load involved — moved the latent by 25–45 % of its own effect.

That is a worst case rather than a prediction; a trained rank-16 *decoder*
adapter re-rounds at 0.8 % the same way. But the eval is what picks a checkpoint,
and 11 MB instead of 5.6 MB is a cheap way to know the file ships the adapter that
was ranked.
