# Decoder LoRA

A LoRA on the autoencoder's **decoder** rather than on the DiT. It changes how
latents are rendered to audio, not how latents are produced.

> **Scope.** Everything measured in this document is **SAME-L**. The mechanism
> and the trainer are not specific to it, but no number here has been reproduced
> on SAME-S, and there is reason to expect it to behave differently — see
> [SAME-S](#same-s) below.

The encoder stays frozen throughout, so the latent space, the DiT and every
existing DiT LoRA remain bit-compatible. A decoder adapter stacks with them
freely.

One adapter trained this way is published at
[`thepatch/same-l-decoder-lora`](https://huggingface.co/thepatch/same-l-decoder-lora),
if hearing one is useful before training your own. Numbers quoted throughout this
document come from that checkpoint and its ablations; they are illustrations of
what the measurements look like, not targets to hit.

---

## What it is for

SAME downsamples 44.1 kHz stereo by 4096×, so there is one latent frame per
93 ms and a snare attack lives entirely inside a single frame. The decoder does
not reconstruct that attack — it resynthesizes one, slightly differently every
pass.

That is invisible in a single generation and compounds badly under **iteration**.
A continuation re-encodes previously decoded audio, so N chained continuations
put the audio through N encode/decode round trips. Running that ladder
deliberately, with no DiT involved:

| iter | tonality p95 | added-vs-real HF (median) | on transients | on sustain | **excess** | 8–12k |
|---|---|---|---|---|---|---|
| 0 | 14.2 | — | — | — | — | −17.4 |
| 1 | 11.8 | −6.9 | −7.3 | −11.5 | **+4.2** | −17.4 |
| 2 | 13.4 | −6.9 | −7.2 | −13.5 | **+6.3** | −17.0 |
| 4 | 17.7 | −7.5 | −6.1 | −16.4 | **+10.3** | −14.8 |
| 6 | 22.8 | −9.0 | −5.6 | −17.6 | **+12.0** | −11.3 |

Reading the table:

1. **It manufactures, it does not merely dull.** Added-HF climbs ~15 dB across
   the ladder while lost-HF stays flat at ≈−30 dB.
2. **It is narrowband and transient-locked.** Energy piles into one band while
   the rest stays dead, and every one of the twelve worst frames at iteration 6
   lands within 35 ms of a detected onset. Sustained material gets relatively
   *cleaner*; percussive attacks get worse.
3. **Iteration 1 is fine.** Tonality actually drops on the first pass — the
   autoencoder smooths. Divergence starts at iteration 2. Single generations are
   not the problem; feedback is.

No DiT LoRA can reach this. The error is downstream of the DiT entirely.

### Why it happens

From SAME's training objective: every loss is `real audio → latent → real audio`.
**Nothing ever asked that `encode(decode(z)) ≈ z`.** Round-trip idempotence was
never an objective, so there is no reason for it to hold, and it does not.

### What does not fix it

All measured on the same clip over 6 iterations, against the untreated baseline
(excess 4.2 → 12.0):

| treatment | excess 1→6 | verdict |
|---|---|---|
| lowpass 16 kHz before re-encode | 4.0 → 10.8 | no effect |
| lowpass 12 kHz before re-encode | 3.9 → 9.8 | no effect |
| latent clamp ±4.0 | 4.2 → 12.1 | no effect |
| dither 5e-4 before re-encode | −1.8 → −2.4 | de-concentrates, adds its own noise |
| chunked encode/decode | 4.3 → 12.0 | not a contributor |

The lowpass result is the important one. **Stripping the hallucinated HF before
re-encoding does not arrest the runaway.** The invented high end is a symptom;
the damage is in the latent's representation of the transient itself. Audio-domain
pre-treatment is a dead end, and so is latent clamping.

### Why it is nevertheless learnable

Perturbing a clean latent with *random* noise the same size as a real round trip
(10.5 % of latent RMS) produces +0.4 dB of tonality and leaves transient excess
negative. The real round trip produces +1.7 dB and flips excess positive. Even
84 % random noise never reproduces the transient-locked signature — so
off-manifold-ness alone is not the mechanism.

Walking the ladder and keeping every latent:

| step | drift % | coherence w/ prev | cos(Δ, z) |
|---|---|---|---|
| z1→z2 | 10.52 | — | 0.161 |
| z3→z4 | 9.94 | 0.690 | 0.121 |
| z6→z7 | 9.95 | 0.687 | 0.142 |

Path straightness **0.735**, against a random-walk expectation of 0.408.
Consecutive drift vectors agree at cos ≈ 0.69 at every step: the encoder/decoder
pair makes the same mistake the same way every pass, so it accumulates
coherently. **A consistent bias is learnable, and therefore cancellable.**

`cos(Δ, z) ≈ 0.13` — the drift is nearly orthogonal to the latent itself, which
is why clamping and rescaling did nothing.

---

## Training

```bash
uv run python scripts/train_decoder_lora.py \
    --model same-l \
    --data_dir /path/to/audio \
    --eval_audio /path/to/holdout.wav \
    --out_dir out/declora \
    --lambda_patch 30
```

Needs only audio — no captions, no paired data. The objective is self-supervised.

Rank 16 on SAME-L is 98 tensors / 5.63M trainable params, about 1.30 % of the
decoder; ~2.9 s/step at 10 s crops, under 10 GB. The decoder is 426M params and
**99.8 % of them are `nn.Linear`**, so the stock `add_lora` covers it without new
injection machinery.

### A worked starting recipe

```bash
uv run python scripts/train_decoder_lora.py \
    --model same-l --rank 16 --steps 8000 --lr 1e-4 \
    --w_real 0.4 --w_drift 0.3 --w_dit 0.3 \
    --lambda_cycle 10.0 --lambda_tonal 0.3 --lambda_patch 30 \
    --patch_size 256 --crop_seconds 10 \
    --dit_latent_dir out/dit_latents \
    --data_dir /path/to/audio --eval_audio /path/to/holdout.wav \
    --out_dir out/declora
```

That run was configured for 8000 steps and the checkpoint that shipped was
**step 2000**, not the last one — see "Picking a checkpoint" below, because the
reason is not obvious.

### Training buckets

The artifact appears on more than one distribution, so latents come from three
places, mixed by `--w_real` / `--w_drift` / `--w_dit`:

| bucket | latents | target audio | which losses |
|---|---|---|---|
| `real` | `E(x)` for real audio | `x` | all three |
| `drift` | `E(D(E(x)))` at depth ≥ 1 | the **original** `x` | all three |
| `dit` | sampled from the base DiT (`--dit_latent_dir`) | none | cycle + tonality + patch |

The `drift` bucket's target is the original audio rather than the degraded input,
which asks the decoder to *undo* accumulated drift rather than faithfully render
a degraded latent.

The `dit` bucket earns its place: generated latents drift 15–18.5 % against real
audio's 10.5 %, and that gap is in the **base model**, not in any adapter — DiT
LoRAs measured drift deltas under half a percent. Generate its latents with
`scripts/gen_dit_latents.py`.

### The losses

| flag | term | what it buys |
|---|---|---|
| `--lambda_rec` | multi-resolution log-STFT L1, K-weighted | keeps the decoder honest, stops the other terms finding degenerate wins |
| `--lambda_cycle` | `‖E(D(z)) − z‖ / ‖z‖`, encoder frozen | **the load-bearing term** — trains the decoder to emit audio the frozen encoder maps back to the same latent |
| `--lambda_tonal` | one-sided HF tonality penalty | pushes invented narrowband HF toward noise |
| `--lambda_patch` | patch-grid structure penalty | see below — not optional if you intend to ship the result |
| `--lambda_hfband` | two-sided HF band-energy match | **off by default** — see below |

Two deliberate choices worth knowing about:

**K-weighting is kept.** Decoder fine-tunes that target the *opposite* defect (a
VAE dull above 6 kHz) remove the perceptual HF de-emphasis to force more high end
out. This decoder over-generates HF under iteration, so removing it would push
the wrong way.

**The tonality penalty is one-sided.** A symmetric version would punish the
decoder for legitimate tonal HF — cymbal ring, sibilance, guitar harmonics. It
only ever pushes toward noise, never away from it.

**`--lambda_hfband` is off because it lost its own trial.** It tests a real
hypothesis — that what the ear rewards is the top octave coming back rather than
peakiness going down, which is what the listening evidence pointed at. But the
run that introduced it, at weight 1.0 on a rank-16 SAME-L adapter, was rejected:
the term competed with reconstruction for the adapter's capacity and
reconstruction lost. It is kept, at 0, because the hypothesis has not been
settled — not because it is a garnish you can leave on.

**There is no discriminator.** Adversarial training is how you make a decoder
invent *more* detail. The failure mode of plain reconstruction losses — smoother,
less confident HF — is the direction this wants.

### `--lambda_patch`, and why it defaults to 0 but you want it on

The first adapter trained here scored clean on every metric it was trained on and
still put an audible **172.27 Hz harmonic comb** into its output. It was found by
ear, in a stem-separated drum track — separation strips the masking content that
hides it in a mix. Against a matched pair (same seed, prompt and DiT LoRA, the
decoder adapter the only difference), stock measured +0.14 dB of comb excess over
a meaningless control comb, and that adapter measured **+9.26 dB**. It
manufactures the comb; it is not amplifying something already there.

The cause is architectural rather than a fault of one checkpoint.
`PatchedPretransform.decode` is a bare reshape,
`rearrange(x, "b (c h) l -> b c (l h)", h=256)`, and the 512 output channels split
as `(c=2, h=256)` — so **output channel index and time-position-within-patch are
the same axis**. The layer feeding it is `WNConv1d(1536, 512, kernel_size=1)`:
per-position, no cross-patch context. Any channel-wise bias the adapter learns is
therefore stamped identically into every patch. `sr/256 = 172.2656 Hz`, with an
F3 fundamental, which is why it reads as a wrong note rather than as noise.

None of the other losses can see it. Multi-res STFT spreads a comb over 60+
harmonics so it is negligible in any single bin; the cycle loss lives in latent
space; the tonality terms only measure flatness in a band. **Any decoder LoRA
trained without this term will do some version of it.**

`patch_grid_penalty` measures both shapes the artifact takes: a coherent additive
stamp that survives averaging over patches, and a *normalised* per-position
roughness profile — normalised so the term penalises SHAPE and cannot be
satisfied by the adapter simply going quieter. Calibration on 20 s crops:

| source | penalty |
|---|---|
| real audio (grid-agnostic by construction — this is the floor) | 2–4e-4 |
| stock decode | 9e-4 – 1.4e-3 |
| adapter trained without the term, strength 1 | 1.9e-2 |
| the same adapter at strength 2 | 1.1e-1 |

~20× separation between stock and the artifact. Start at `30`: against the table
above that is ~0.03–0.04 on clean audio — negligible beside a reconstruction term
of ~1.0 — and ~0.57 at the artifact level, where it becomes the thing the
optimiser has to answer for.

Two things to know when reading the number:

- The coherent half has a **hard statistical floor of ~1/P** for P patches per
  crop — ~5.8e-4 at the 10 s default, ~2.9e-4 at 20 s. It cannot reach zero.
  Calibrate at the crop length you actually use, and treat convergence toward the
  **stock** value as the target. Below stock means the reduction is being bought
  with something else.
- The term applies to every bucket including `dit`, so the reported value tracks
  bucket composition within a run. A rise that follows the `g` count in the log
  is the mix, not the adapter degrading.

#### What it buys, and where to measure it

Comb excess in dB over a control comb at meaningless spacing, mean of 4
generations. The same latents were decoded through each adapter, so the decoder
is the only variable:

| variant | comb excess | vs stock |
|---|---|---|
| stock | 2.80 | — |
| `--lambda_patch 0`, strength 1 | 13.70 | **+10.90** |
| `--lambda_patch 30`, strength 1 | 1.97 | **−0.83** |
| `--lambda_patch 0`, +3 round trips | 13.34 | +10.54 |
| `--lambda_patch 30`, +3 round trips | 1.37 | **−1.43** |
| `--lambda_patch 0`, strength 2 | 18.52 | +15.72 |
| `--lambda_patch 30`, strength 2 | 3.49 | **+0.68** |

Without the term the adapter adds ~11 dB of comb and holds it under re-encoding.
With it, the adapter sits at or slightly below stock, and is still within 0.7 dB
of stock at strength 2 — double what it ships at, where every effect is 2–4×
clearer.

**Measure this on audio the model generated, not on real audio pushed round the
autoencoder.** They are different distributions and the artifact does not behave
the same on both: on real music the penalised adapter measures +1.5 to +4 dB of
comb excess where on model output it is at or below stock. Model output is what
the decoder is actually asked to render, and it is where the artifact is audible
under stem separation — which is how it was found in the first place.

### `--tonality_bands`

Where the tonality penalty looks. The default three sub-bands all sit at 6 kHz and
above, which is right for SAME-L, whose invented tonality is genuinely high.

<a name="same-s"></a>
### On SAME-S

Everything above was developed and measured on SAME-L. SAME-S is a different
proposition and this document does not claim to cover it.

Four things that are different, from exploratory runs on it:

- **There is far less adapter to work with.** At rank 16 the SAME-L decoder
  exposes 49 `nn.Linear` layers / 5.63M trainable params; SAME-S exposes 25 /
  1.42M — under a quarter. Terms that merely compete for capacity on SAME-L can
  crowd reconstruction out entirely there, so every weight above should be
  treated as a SAME-L value, not a default.
- **The patch grid transfers; the weight for it does not.** `patch_size` is 256
  on both, so `--lambda_patch` measures the same thing. But 30 — the SAME-L
  value — was too weak on SAME-S and left the comb roughly where it started.
  Around **90** is what brought it down.
- **The tonality term did not earn its place.** Turning it off entirely
  (`--lambda_tonal 0`) made no measurable difference against an otherwise
  identical run, so on SAME-S the sensible starting point is off, not retuned.
  The artifact there does sit lower — around 1–8 kHz, where a p95 tonality gap
  of +4.26 dB at 4–8 kHz against +0.77 dB at 8–16 kHz means the default 6 kHz
  floor can barely see it — so `--tonality_bands 1000-2000,2000-4000,4000-8000`
  is where to look *if* you want to revisit it. Moving the band is not by itself
  the adaptation required.
- **Do not screen SAME-S results on spectral flatness.** The tonality detector
  is inverted on that material: the clip that sounds squeakiest scores lowest.
  Judge by ear, and use `--lambda_patch` and the comb measurement for the part
  that is measurable.

Expect the rest to need its own tuning pass and its own listening.

---

## Picking a checkpoint

Every `--eval_every` steps the trainer runs the round-trip ladder against a
held-out clip and prints it beside a baseline captured at step 0 with the adapter
switched off. The numbers to watch are transient excess and tonality p95 coming
down **without** band energies or reconstruction regressing.

Onset and steady artifact levels are printed separately as well as their
difference, and that matters: `excess` is onset-minus-steady, so it *widens* when
sustained material cleans up faster than transients even though both improved.
Without both sides you cannot distinguish that from transients actually
degrading.

The in-loop ladder is a single-pass measurement, and it will generally keep
improving past the point where the checkpoint is best for real use. For the
published checkpoint the ladder favoured step 3000–4000; step 2000 was chosen on
re-encode behaviour and on ears. **Screen candidates on re-encode depth and on
percussion-dense material, not on the ladder alone.**

### The metric gap at low re-encode depth

What a decoder adapter buys scales with re-encode depth, and that shapes how you
should evaluate one. Example measurements from the published checkpoint — tonal
frames (>12 dB tonality, 6–16 kHz) on 2-minute generations, mean of 4, the
adapter the only variable:

| depth | stock | adapted | | 12–16 kHz (stock → adapted) | SI-SDR vs own d0 |
|---|---|---|---|---|---|
| d0 — fresh generation | 364 | 422 | −16% | −5.6 → −5.3 | — |
| d1 — one transform or continuation | 432 | 445 | −3% | −6.2 → −5.5 | 14.7 → 18.3 |
| d2 | 821 | 519 | **37% fewer** | −6.6 → −5.6 | 10.1 → 13.7 |
| d3 | 1665 | 619 | **63% fewer** | −7.1 → −5.8 | 7.8 → 11.2 |

**Read the last two columns before believing the first.** A tonal-frame count can
fall because the artifact went away *or* because the content did, and an adapter
that quietly sands the top end will post an excellent frame count. Here it does
not: at every depth the adapted render holds more 12–16 kHz energy than stock,
sits marginally louder in RMS, and stays closer to its own depth-0 render
(SI-SDR 11.2 against stock's 7.8 by d3) — so it is drifting less with depth, not
erasing what drifts. The 8–12 kHz band does drop (+2.7 → +1.1 by d3), which is
the band the artifact piles into.

Always pair a frame count at depth with a content measure. Reading the count
alone once produced a confident "the benefit grows with depth" on an adapter that
was, at that depth, deleting the audio.

**The squeak is audible at d0 by ear — hi hats and snares especially — and none
of the metrics here show it.** Flat or slightly negative at d0: tonality p95 in
6–16 kHz and 1–8 kHz, frame counts over 10/12/15 dB, onset-locked tonality
excess, fp32 and fp16, two DiT LoRAs, 30 s and 120 s material.

The measurement has no headroom there. About 640 of ~10,300 frames already read
as tonal before any re-encoding, because cymbals and distorted guitar genuinely
are tonal, so a few dozen added squeak frames cannot move a count that size.
Separating artifact from content works at depth but not at d0 without a clean
reference, and generated audio has none.

Combined with the tonality detector's habit of pointing the wrong way — on SAME-S
the clip that sounds squeakiest scores lowest — the practical rule is: **judge
low-depth behaviour by ear, and use the metrics for depth.** If you can devise a
reference-free measure of HF transient artifacts on generated audio, it would
improve every screen in this document.

### Measurement floors — read before trusting small deltas

Neither half of the autoencoder is bit-deterministic:

| quantity | floor (on one clip) |
|---|---|
| encoder: two encodes of the same audio | **1.50 %** relative |
| decoder: two decodes of the same latent | **7.0e-3** max abs |

**Measure your own.** These are not constants. The decoder's forward pass carries
`mask_noise`, so its floor is material-dependent — the same measurement on a
transient-heavy clip gives **1.9e-2**, nearly 3x the number above. Re-measure on
the material you are testing, then express your deltas as a multiple of that.

Consequences:

- The round-trip drift numbers stand: 10.5 % for real audio is 7× the encoder
  floor.
- **The cycle loss has an irreducible floor of ~1.5 %.** Training cannot drive
  `‖E(D(z))−z‖/‖z‖` below it. That is the target, not zero.
- Any audio A/B must be expressed as a multiple of the decoder floor **you
  measured on that material**. Absolute tolerances are meaningless, and so is
  reusing someone else's floor.

---

## Inference

Decoder checkpoints load through the stock loader with no special handling — the
`target: "decoder"` in the checkpoint config is what routes them:

```bash
uv run python run_gradio.py --model medium-base \
    --lora-ckpt-path style.safetensors decoder_lora.safetensors
```

They stack with DiT LoRAs in any order, and the same file loads onto either the
full serving model (`model.pretransform.model.decoder`) or a standalone
`AutoencoderModel` (`model.decoder`), so a training or eval harness can use it
directly.

Because the encoder is untouched, a decoder adapter changes **only** the render
step. It applies equally to plain generation, continuation and transform.

### Verifying an adapter is actually doing something

Attaching is not the same as loading. `load_state_dict(strict=False)` will accept
a state dict whose every key misses, leaving a correctly-shaped, correctly-counted
adapter full of zeros that reports as a clean load and does nothing. Assert on
**output**, against a nondeterminism floor you measure on the same audio:

```python
z = ae.encode(x)
a = ae.decode(z).clone()
floor = (ae.decode(z) - a).abs().max()      # two decodes of the SAME latent
load_and_apply_loras(ae, [ckpt], "autoencoder")
assert (ae.decode(z) - a).abs().max() > 10 * floor
```

Strength control reaches the pretransform, so `set_lora_strength(ae.decoder, 0.0)`
should render back at the stock decode within that same floor. If it does not, the
adapter never attached where you think it did.

---

## Traps

Each of these produces plausible-looking numbers while being wrong.

- **`pretransform.enable_grad` defaults to `False`**, wrapping patch/unpatch in
  `torch.no_grad()`. Left alone, the reconstruction and cycle terms silently stop
  reaching the LoRA params.
- **The eval ladder must keep the batch dim.** `ae.encode` here is the raw
  autoencoder, which does no preprocessing, so a `(C, T)` tensor breaks the patch
  rearrange.
- **Crops must be trimmed to a multiple of the downsampling ratio.** `load_audio`
  slices at the source rate before resampling, so the length is not guaranteed to
  factor.
- **`lora_configs` at inference only gates the sigma interval and layer filter.**
  Strength is a separate `set_lora_strength` call, or the adapter silently does
  nothing.
- **LoRA `alpha` must match between build and checkpoint** (`scaling = alpha/rank`),
  or a reload is `rank`× stronger than what was saved.
- **The eval window has to exist inside the clip.** `--eval_offset` skips the
  head of `--eval_audio` so a fade-in does not dominate the ladder, and it
  defaults to 20 s. On a clip shorter than `--eval_offset + --eval_seconds` the
  trainer now moves the window earlier and says so; a clip shorter than
  `--eval_seconds` is evaluated whole with a warning that its ladder numbers are
  not comparable to a full-window run. Both used to happen silently, or die
  inside the encoder with a shape error.
- **Sigma-interval and layer-filter controls do not reach autoencoder adapters.**
  That gating lives in the DiT's forward pass, so for a `decoder` or `encoder`
  target only *strength* applies. The controls are inert rather than wrong -- they
  cannot affect the adapter at all -- but nothing currently hides them.
- **`--data_dir` walks all the way down unless you stop it.** Datasets often
  keep derived audio in subdirectories — time-stretched variants, stem splits,
  per-track working files — and an unbounded walk trains on those as though they
  were independent material, silently over-representing whatever happened to
  have been expanded. `--max_depth 1` restricts it to files sitting directly in
  the directory.
- **Tonality is a detector, not an annoyance predictor.** It is what found the
  artifact, but a clip can score well and still sound wrong. Screen on ears and on
  percussion-dense material.
