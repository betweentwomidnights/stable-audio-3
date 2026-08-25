"""Losses for the SAME decoder LoRA.

Five terms. The first three are the core of the objective and aimed at the two
things the probes actually measured; the last two exist because of artifacts the
first three provably cannot see.

  reconstruction   multi-resolution log-STFT L1, K-weighted. Keeps the decoder
                   honest and stops the other terms finding degenerate wins.

  round-trip       ||E(D(z)) - z|| / ||z||, encoder frozen. This is THE term:
  consistency      the probes showed the encoder/decoder pair carries a
                   systematic bias (consecutive drift vectors agree at cos 0.69,
                   path straightness 0.735 vs 0.408 for a random walk) that
                   lands on percussive attacks. A consistent bias is learnable,
                   so training the decoder to emit audio the frozen encoder maps
                   back to the same latent attacks it directly. Expressed as
                   RELATIVE drift so it shares units with the eval metric
                   (10.5% for real audio, 15-18.5% for generated) and so its
                   scale does not depend on latent magnitude.

  HF tonality      one-sided penalty on spectral peakiness, over the sub-bands
                   given by --tonality_bands (default 6-18 kHz for SAME-L). The
                   differentiable version of dragging an erosion effect across
                   the snare: real hats and cymbals are noise-like (flat, low
                   tonality), the artifact is narrowband (peaky, high tonality).
                   ONE-SIDED matters -- a symmetric version would punish the
                   decoder for legitimate tonal HF (cymbal ring, guitar
                   harmonics, sibilance), so we only ever push toward noise,
                   never away from it.

  patch grid       penalty on structure locked to the un-patch grid. NOT
                   optional if you intend to ship the result: the pretransform's
                   un-patch makes output channel index and time-position-within-
                   patch the same axis, so any channel-wise bias the adapter
                   learns is stamped into every patch as a comb at sr/patch.
                   None of the three terms above can see it -- multi-res STFT
                   spreads a comb over 60+ harmonics, the cycle loss lives in
                   latent space, and the tonality terms only measure flatness
                   inside their band. Off by default; see --lambda_patch.

  HF band match    two-sided match of HF band ENERGY to the target. Off by
                   default -- it competes with reconstruction for a rank-16
                   adapter's capacity and reconstruction loses. Kept because the
                   hypothesis it tests (that restored top octave, not reduced
                   peakiness, is what the ear rewards) has not been settled.

Note on K-weighting: decoder fine-tunes that target the OPPOSITE defect -- a VAE
dull above 6 kHz -- remove the perceptual HF de-emphasis to force more high end
out of the decoder. This one over-generates HF under iteration, so removing it
would push the wrong way. Kept deliberately.
"""

import torch

EPS = 1e-10


def cmag(Z):
    """Magnitude of a complex tensor WITHOUT complex .abs().

    On some CUDA setups (observed on sm_121 / GB10) complex abs() dispatches to
    an nvrtc-JIT'd kernel and fails with "invalid value for --gpu-architecture".
    sqrt(re^2+im^2) stays on ordinary CUDA kernels, is differentiable, and costs
    nothing elsewhere, so it is used unconditionally.
    """
    return torch.sqrt(Z.real.pow(2) + Z.imag.pow(2) + 1e-12)


def _stft(sig, n_fft, hop):
    win = torch.hann_window(n_fft, device=sig.device, dtype=torch.float32)
    return torch.stft(
        sig.float().reshape(-1, sig.shape[-1]),
        n_fft,
        hop,
        window=win,
        return_complex=True,
        center=True,
    )


def _bins(lo_hz, hi_hz, n_fft, sr):
    import math

    b0 = max(int(math.floor(lo_hz * n_fft / sr)), 0)
    b1 = min(int(math.ceil(hi_hz * n_fft / sr)), n_fft // 2 + 1)
    return b0, b1


def multires_stft_loss(y, x, ffts=(512, 1024, 2048), sr=44100, hf_tilt=1.5):
    """Log-magnitude L1 across resolutions with a gentle high-shelf."""
    total = y.new_zeros(())
    n = min(y.shape[-1], x.shape[-1])
    y, x = y[..., :n], x[..., :n]
    for n_fft in ffts:
        hop = n_fft // 4
        Y, X = _stft(y, n_fft, hop), _stft(x, n_fft, hop)
        freqs = torch.linspace(0, sr / 2, Y.shape[-2], device=y.device)
        w = (1.0 + hf_tilt * (freqs / (sr / 2)).clamp(0, 1)).unsqueeze(-1)
        total = (
            total
            + ((torch.log(cmag(Y) + 1e-5) - torch.log(cmag(X) + 1e-5)).abs() * w).mean()
        )
    return total / len(ffts)


def relative_latent_error(z, z_ref):
    """||z - z_ref|| / ||z_ref||, cropped to the shorter of the two.

    Everything latent-side in this project is expressed this way -- round-trip
    drift, inversion distance, encoder-vs-known-latent gap -- so the losses share
    units with the probes and none of them scale with latent magnitude.
    """
    m = min(z.shape[-1], z_ref.shape[-1])
    a, b = z_ref[..., :m].float(), z[..., :m].float()
    return (b - a).norm() / (a.norm() + EPS)


def cycle_loss(z_rt, z):
    """Relative round-trip drift, the same quantity the probes report.

    Floor is ~1.5% -- the encoder's own nondeterminism -- so this cannot be
    driven to zero and should not be expected to. (That nondeterminism is not
    hardware noise: the SAME encoder adds `mask_noise` = 1e-3 gaussian to its
    learned query tokens on every forward, ungated by train/eval. Draw the same
    noise for both passes and the floor disappears.)
    """
    return relative_latent_error(z_rt, z)


def patch_grid_penalty(y, patch=256, eps=1e-8):
    """Structure locked to the decoder's un-patch grid.

    SAME-L's pretransform is `patched` at patch_size 256, and its decode is a
    bare reshape -- rearrange(x, "b (c h) l -> b c (l h)", h=256) -- with the
    512 output channels split as (c=2, h=256). So OUTPUT CHANNEL INDEX AND
    TIME-POSITION-WITHIN-PATCH ARE THE SAME AXIS, and the layer feeding it is
    WNConv1d(1536, 512, kernel_size=1): per-position, no cross-patch context.
    Any channel-wise bias the adapter learns is therefore stamped identically
    into every patch, which is a comb at sr/256 = 172.27 Hz.

    None of the other losses can see that. multires_stft spreads a comb over 60+
    harmonics so it is negligible in any single bin, cycle_loss lives in latent
    space, and the tonality terms only measure 6-16 kHz flatness. A rank-16 adapter
    trained without this term scored ~20x this penalty against stock at strength
    1 while every metric it WAS trained on looked clean. It was found by ear, in
    a stem-separated drum track -- separation strips the masking content that
    hides it in a mix.

    Real audio has no reason to correlate with the grid, so flat is the correct
    answer and any deviation is artifact. Measured on 20 s crops: real audio
    2-4e-4, stock decode 9e-4..1.4e-3, and an adapter trained WITHOUT this term
    1.9e-2 at strength 1 -- roughly 20x the stock value.

    Two shapes, because the artifact uses both:
      coherent   a fixed additive stamp, which survives averaging over patches
      modulated  a fixed roughness envelope -- normalised, so only the SHAPE is
                 penalised and the term cannot be satisfied by going quieter

    NOTE the coherent term has a hard statistical floor of ~1/P for P patches
    per crop (incoherent audio averages down as 1/sqrt(P)), so it cannot reach
    zero: ~5.8e-4 at a 10 s crop, ~2.9e-4 at 20 s. Calibrate any weight at the
    crop length actually used, and treat convergence toward the STOCK value as
    the goal -- landing below it means the reduction is being bought with
    something else.
    """
    if y.dim() == 2:
        y = y.unsqueeze(0)
    b, c, t = y.shape
    n = ((t - 1) // patch) * patch
    if n < patch:
        return y.new_zeros(())

    blocks = y[..., :n].reshape(b, c, -1, patch)
    coherent = blocks.mean(dim=2).pow(2).mean() / (y.pow(2).mean() + eps)

    step = (y[..., 1 : n + 1] - y[..., :n]).abs().reshape(b, c, -1, patch)
    prof = step.mean(dim=2)
    prof = prof / (prof.mean(dim=-1, keepdim=True) + eps)
    modulated = (prof - 1.0).pow(2).mean()

    return coherent + modulated


def hf_tonality(sig, sr, lo=6000.0, hi=16000.0, n_fft=2048):
    """Mean HF tonality in dB, reduced over items and frames.

    Higher = peakier = whistlier. Scalar, not per-item: the mean is taken here
    so the callers can treat it as a loss term directly.
    """
    hop = n_fft // 4
    S = cmag(_stft(sig, n_fft, hop))
    b0, b1 = _bins(lo, hi, n_fft, sr)
    p = S[:, b0:b1].pow(2) + EPS
    geo = torch.exp(torch.log(p).mean(dim=1))
    ari = p.mean(dim=1)
    flat = (geo / (ari + EPS)).clamp(EPS, 1.0)
    return (-10.0 * torch.log10(flat)).mean()


def hf_band_match_loss(
    y, x, sr, bands=((12000.0, 16000.0), (16000.0, 22050.0)), n_fft=2048
):
    """Match HF band ENERGY to the target. Two-sided, unlike the tonality term.

    Added on listening evidence. The case the ear picked out as clearly better --
    a breakcore/drum-and-bass generation, single pass -- moved only -0.69 dB on
    tonality and not at all at 8-12 kHz, but gained +2.2 dB at 16-22 kHz. Across
    every A/B run, 16-22 kHz went UP with the adapter (+0.74 to +7.95). So the
    audible improvement may be the top octave coming back rather than peakiness
    going down, which earlier adapters achieved without ever optimising for it.

    Two-sided on purpose: losing air and inventing air are both wrong. The
    round-trip probes showed the failure does BOTH -- it drains 16-22 kHz while
    inflating 8-12 kHz -- so a one-sided version would only fix half of it.
    """
    hop = n_fft // 4
    Y, X = _stft(y, n_fft, hop), _stft(x, n_fft, hop)
    total = y.new_zeros(())
    for lo, hi in bands:
        b0, b1 = _bins(lo, hi, n_fft, sr)
        if b1 <= b0:
            continue
        ey = Y[:, b0:b1].real.pow(2).add(Y[:, b0:b1].imag.pow(2)).mean() + EPS
        ex = X[:, b0:b1].real.pow(2).add(X[:, b0:b1].imag.pow(2)).mean() + EPS
        # dB-domain error, so it is scale-free and symmetric in ratio terms.
        total = total + (10.0 * torch.log10(ey / ex)).abs()
    return total / max(len(bands), 1)


def hf_tonality_penalty_multiband(
    y,
    sr,
    target_audio=None,
    target_db=None,
    bands=((6000.0, 10000.0), (10000.0, 14000.0), (14000.0, 18000.0)),
    n_fft=2048,
):
    """Per-sub-band one-sided tonality penalty.

    An earlier version used one broadband 6-16 kHz penalty, which the decoder
    satisfied within ~150 steps and which then contributed no gradient for the
    rest of the run -- the erosion effect shaped only the opening couple of
    percent of training. Requiring the
    constraint to hold in EACH sub-band is strictly harder to satisfy, so it
    stays engaged, without pushing tonality below the target (which would erode
    real cymbal and string detail rather than squeaks).
    """
    total = y.new_zeros(())
    for lo, hi in bands:
        t_y = hf_tonality(y, sr, lo, hi, n_fft)
        if target_audio is not None:
            ref = hf_tonality(target_audio, sr, lo, hi, n_fft).detach()
        elif target_db is not None:
            ref = torch.as_tensor(float(target_db), device=y.device)
        else:
            ref = torch.zeros((), device=y.device)
        total = total + (t_y - ref).clamp(min=0.0)
    return total / max(len(bands), 1)
