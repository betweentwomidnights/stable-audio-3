"""Round-trip eval helpers shared by the decoder-LoRA trainer.

Extracted from the iterated VAE round-trip probe that established the artifact
these adapters target, so the trainer's periodic eval reports the same numbers,
computed the same way, as the measurements the objective was designed against.

Glossary, because the metric names are not self-explanatory:

  a2s        added HF energy relative to the real HF energy at that instant, dB.
             The decoder INVENTS high frequency under iteration; this measures
             how much, against how much is legitimately there.
  excess     a2s on transient frames minus a2s on sustained frames. Positive
             means the artifact rides percussion, which is its signature.
  tonality   HF spectral peakiness. Real hats and cymbals are noise-like and
             score low; a manufactured whistle scores high.

Onset and steady artifact levels are reported SEPARATELY as well as their
difference. `excess` widens when sustained material cleans up faster than
transients even though both improved, and without the two sides you cannot tell
that from transients actually degrading.
"""

import math

import torch
import torchaudio

EPS = 1e-10
BANDS = [
    ("sub_0_120", 0, 120),
    ("low_120_500", 120, 500),
    ("mid_500_2k", 500, 2000),
    ("umid_2k_4k", 2000, 4000),
    ("pres_4k_8k", 4000, 8000),
    ("brill_8k_12k", 8000, 12000),
    ("air_12k_16k", 12000, 16000),
    ("top_16k_22k", 16000, 22050),
]


def stft(x, n_fft, hop):
    """Complex STFT of a mono float32 tensor -> (freq, frames)."""
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )


def istft(X, n_fft, hop, length):
    window = torch.hann_window(n_fft, device=X.device, dtype=torch.float32)
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=length,
    )


def bin_range(lo_hz, hi_hz, n_fft, sr):
    """Inclusive-exclusive FFT bin indices covering [lo_hz, hi_hz)."""
    lo = int(math.floor(lo_hz * n_fft / sr))
    hi = int(math.ceil(hi_hz * n_fft / sr))
    return max(lo, 0), min(hi, n_fft // 2 + 1)


def db(x):
    return 10.0 * math.log10(float(x) + EPS)


def band_energies_db(mag, n_fft, sr):
    """Mean power per band, in dB. mag is (freq, frames)."""
    power = mag.pow(2)
    out = {}
    for name, lo, hi in BANDS:
        b0, b1 = bin_range(lo, hi, n_fft, sr)
        if b1 <= b0:
            out[name] = float("nan")
            continue
        out[name] = db(power[b0:b1].mean().item())
    return out


def spectral_flatness_db(mag, n_fft, sr, lo_hz, hi_hz):
    """Per-frame tonality = -10*log10(geomean/arithmean) over a band.

    Noise-like (hats, cymbals, air)  -> flatness near 1 -> tonality near 0 dB.
    Tonal (whistles, chirps, rings)  -> flatness << 1   -> tonality large.
    Only frames with meaningful band energy are counted, so silence does not
    dilute the statistic.
    """
    b0, b1 = bin_range(lo_hz, hi_hz, n_fft, sr)
    power = mag[b0:b1].pow(2) + EPS  # (bins, frames)
    geo = torch.exp(torch.log(power).mean(dim=0))
    ari = power.mean(dim=0)
    flatness = (geo / (ari + EPS)).clamp(EPS, 1.0)
    tonality = -10.0 * torch.log10(flatness)

    # Gate on frames that actually carry band energy (within 40 dB of the
    # loudest frame) — otherwise near-silent frames dominate.
    frame_db = 10.0 * torch.log10(ari + EPS)
    if frame_db.numel() == 0:
        return float("nan"), float("nan"), tonality
    gate = frame_db >= (frame_db.max() - 40.0)
    if gate.sum() < 4:
        gate = torch.ones_like(gate, dtype=torch.bool)
    active = tonality[gate]
    return (
        float(active.mean()),
        float(torch.quantile(active.float(), 0.95)),
        tonality,
    )


def hf_flux_std(mag, n_fft, sr, lo_hz, hi_hz):
    """Std-dev of frame-to-frame log-magnitude change in a band."""
    b0, b1 = bin_range(lo_hz, hi_hz, n_fft, sr)
    logmag = 20.0 * torch.log10(mag[b0:b1] + EPS)
    if logmag.shape[1] < 2:
        return float("nan")
    d = logmag[:, 1:] - logmag[:, :-1]
    return float(d.std())


def si_sdr(est, ref):
    """Scale-invariant SDR in dB. Both 1-D, same length."""
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = (est @ ref) / (ref @ ref + EPS)
    target = alpha * ref
    noise = est - target
    return db((target @ target).item() / ((noise @ noise).item() + EPS))


def highpass_via_stft(x, sr, n_fft, hop, cutoff_hz):
    X = stft(x, n_fft, hop)
    # bin_range(0, f) -> (0, bin_of_f); the CUTOFF bin is the second element.
    _, cut = bin_range(0, cutoff_hz, n_fft, sr)
    X[:cut] = 0
    return istft(X, n_fft, hop, x.shape[-1])


def spectral_residuals(x_test, x_ref, n_fft, hop):
    """Split x_test vs x_ref into 'added' and 'lost' spectral energy.

    Returns (added_wave, lost_wave, added_mag, X_test_mag, X_ref_mag).
    Phase-insensitive by construction: we compare magnitudes and resynthesize
    each part with the phase of the generation it came from.
    """
    Xt = stft(x_test, n_fft, hop)
    Xr = stft(x_ref, n_fft, hop)
    frames = min(Xt.shape[1], Xr.shape[1])
    Xt, Xr = Xt[:, :frames], Xr[:, :frames]
    mt, mr = Xt.abs(), Xr.abs()

    added_mag = (mt - mr).clamp(min=0.0)
    lost_mag = (mr - mt).clamp(min=0.0)

    length = min(x_test.shape[-1], x_ref.shape[-1])
    added = istft(added_mag * torch.exp(1j * torch.angle(Xt)), n_fft, hop, length)
    lost = istft(lost_mag * torch.exp(1j * torch.angle(Xr)), n_fft, hop, length)
    return added, lost, added_mag, mt, mr


def write_wav(path, wave, sr, normalize=False, headroom_db=-1.0):
    """wave: (C, T) float tensor on any device."""
    w = wave.detach().float().cpu()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if normalize:
        peak = w.abs().max().item()
        if peak > EPS:
            w = w * (10.0 ** (headroom_db / 20.0)) / peak
    else:
        peak = w.abs().max().item()
        if peak > 1.0:
            w = w / peak
    torchaudio.save(str(path), w, sr)


def load_audio(path, target_sr, seconds=None, offset=0.0):
    wave, sr = torchaudio.load(str(path))
    if offset > 0:
        start = int(offset * sr)
        wave = wave[:, start:]
    if seconds:
        wave = wave[:, : int(seconds * sr)]
    if sr != target_sr:
        wave = torchaudio.transforms.Resample(sr, target_sr)(wave)
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    elif wave.shape[0] > 2:
        wave = wave[:2]
    return wave


def to_mono(wave):
    return wave.mean(dim=0) if wave.dim() > 1 else wave


def analyse(x_test, x_ref, sr, args):
    """All metrics for one generation against one reference. Mono-sum basis."""
    n_fft, hop = args.n_fft, args.hop
    mt = to_mono(x_test).float().cpu()
    mr = to_mono(x_ref).float().cpu()
    length = min(mt.shape[-1], mr.shape[-1])
    mt, mr = mt[:length], mr[:length]

    Xt = stft(mt, n_fft, hop).abs()

    out = {}
    out["band_db"] = band_energies_db(Xt, n_fft, sr)
    tmean, t95, _ = spectral_flatness_db(
        Xt, n_fft, sr, args.tonality_lo, args.tonality_hi
    )
    out["hf_tonality_db_mean"] = tmean
    out["hf_tonality_db_p95"] = t95
    out["hf_flux_std"] = hf_flux_std(Xt, n_fft, sr, args.hf_lo, args.hf_hi)

    # crest factor of the HF band — transient sharpness of hats/snares
    hf = highpass_via_stft(mt, sr, n_fft, hop, args.hf_lo)
    rms = float(hf.pow(2).mean().sqrt())
    out["hf_crest_db"] = db((float(hf.abs().max()) ** 2) / (rms**2 + EPS))

    # fidelity vs reference
    out["si_sdr_db"] = si_sdr(mt, mr)
    hf_ref = highpass_via_stft(mr, sr, n_fft, hop, args.hf_lo)
    out["hf_si_sdr_db"] = si_sdr(hf, hf_ref)

    # spectral residuals
    added, lost, added_mag, magt, magr = spectral_residuals(mt, mr, n_fft, hop)
    b0, b1 = bin_range(args.hf_lo, args.hf_hi, n_fft, sr)
    out["added_total_db"] = db(added_mag.pow(2).mean().item())
    out["added_hf_db"] = db(added_mag[b0:b1].pow(2).mean().item())
    out["lost_hf_db"] = db((magr - magt).clamp(min=0)[b0:b1].pow(2).mean().item())
    # log-magnitude L1 restricted to HF — the "how different does the top sound"
    out["hf_logmag_l1_db"] = float(
        (20 * torch.log10(magt[b0:b1] + EPS) - 20 * torch.log10(magr[b0:b1] + EPS))
        .abs()
        .mean()
    )

    # Per-frame artifact-to-signal ratio in the HF band.
    #
    # Ranking frames by ABSOLUTE added energy just finds the loudest moments
    # in the track (a crash wins on level alone) — which is not where the ear
    # notices the artifact. What the ear tracks is how much invented energy
    # there is RELATIVE to the real HF content at that instant, so that is
    # what we rank and report.
    frame_added_hf = added_mag[b0:b1].pow(2).mean(dim=0)
    frame_ref_hf = magr[b0:b1].pow(2).mean(dim=0)
    ratio_db = 10.0 * torch.log10((frame_added_hf + EPS) / (frame_ref_hf + EPS))

    ref_level_db = 10.0 * torch.log10(frame_ref_hf + EPS)
    valid = ref_level_db >= (ref_level_db.max() - 45.0)
    if valid.sum() < 8:
        valid = torch.ones_like(valid)

    out["artifact_to_signal_db_median"] = float(ratio_db[valid].median())
    out["artifact_to_signal_db_p90"] = float(
        torch.quantile(ratio_db[valid].float(), 0.90)
    )
    return (
        out,
        added,
        lost,
        {
            "frame_added_hf": frame_added_hf,
            "frame_ref_hf": frame_ref_hf,
            "ratio_db": ratio_db,
            "valid": valid,
            "ref_mag": magr,
        },
    )


def detect_onsets(mag, sr, hop, min_gap_s=0.08, thresh_k=1.5):
    """Onset frames via half-wave-rectified spectral flux on a magnitude STFT.

    Used to test whether the round-trip artifact is transient-locked — i.e.
    whether it lands on snares and percussion hits rather than spreading
    evenly. That is what the ear reports, so it is what we measure.
    """
    logmag = 20.0 * torch.log10(mag + EPS)
    flux = (logmag[:, 1:] - logmag[:, :-1]).clamp(min=0).sum(dim=0)
    if flux.numel() < 3:
        return torch.zeros(0, dtype=torch.long)
    med = flux.median()
    mad = (flux - med).abs().median() + EPS
    peaks = []
    min_gap = max(int(min_gap_s * sr / hop), 1)
    last = -min_gap * 2
    for i in range(1, flux.numel() - 1):
        v = flux[i]
        if v < med + thresh_k * mad:
            continue
        if v >= flux[i - 1] and v >= flux[i + 1] and (i - last) >= min_gap:
            peaks.append(i + 1)  # +1: flux index i is the step into frame i+1
            last = i
    return torch.tensor(peaks, dtype=torch.long)


def onset_locked_stats(ratio_db, valid, onsets, n_frames, sr, hop, window_s=0.12):
    """Compare artifact-to-signal ratio on transients vs between them.

    Returns (onset_mean_db, steady_mean_db, excess_db, n_onsets). A positive
    excess means the round trip damages percussive attacks more than it
    damages sustained material.
    """
    if onsets.numel() == 0:
        return float("nan"), float("nan"), float("nan"), 0
    win = max(int(window_s * sr / hop), 1)
    mask = torch.zeros(n_frames, dtype=torch.bool)
    for o in onsets.tolist():
        mask[o : min(o + win, n_frames)] = True
    on = ratio_db[mask & valid]
    off = ratio_db[(~mask) & valid]
    if on.numel() < 2 or off.numel() < 2:
        return float("nan"), float("nan"), float("nan"), int(onsets.numel())
    on_m, off_m = float(on.mean()), float(off.mean())
    return on_m, off_m, on_m - off_m, int(onsets.numel())
