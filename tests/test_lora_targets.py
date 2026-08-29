"""Tests for LoRA `target` resolution — dit / decoder / encoder.

These run on synthetic modules rather than a downloaded checkpoint. The things
worth pinning here are structural (which module an adapter attaches to, which
parametrization slot its keys land in, what happens to a layer that cannot be
parametrized), and all of them reproduce on a handful of nn.Linear layers
arranged in the same shape as the real model.

The assertions deliberately check that adapter WEIGHTS ARRIVED, not that an
adapter attached. Attaching is not the failure mode: load_state_dict(strict=False)
will happily accept a state dict whose every key misses, leaving a
correctly-shaped, correctly-counted, all-zero adapter that does nothing.
"""

import sys
from functools import partial

import pytest
import torch
import torch.nn as nn

from stable_audio_3 import StableAudioModel
from stable_audio_3.models.lora import (
    LoRAParametrization,
    add_lora,
    get_lora_state_dict,
    save_lora_safetensors,
)
from stable_audio_3.models.lora.loader import (
    _resolve_lora_targets,
    load_and_apply_loras,
)

_RANK = 4
_ALPHA = 4.0


def _lora_cfg(rank=_RANK, alpha=_ALPHA, lora_index=0):
    return {
        nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=rank,
                lora_alpha=alpha,
                adapter_type="lora",
                lora_index=lora_index,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Synthetic stand-ins, shaped like the real thing
# ---------------------------------------------------------------------------


class _Half(nn.Module):
    """Stands in for an autoencoder encoder or decoder."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

    def forward(self, x):
        return self.layers(x)


class _AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Half()
        self.decoder = _Half()


class _Pretransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _AE()


class _FullModel(nn.Module):
    """model.model (DiT) + model.conditioner + model.pretransform.model.{enc,dec}."""

    def __init__(self):
        super().__init__()
        self.model = _Half()
        self.conditioner = _Half()
        self.pretransform = _Pretransform()


def _decoder_ckpt(tmp_path, name="dec.safetensors", seed=0):
    """A decoder-targeted checkpoint with non-zero lora_B, saved at slot 0.

    lora_B initialises to zeros, so it is filled here: a checkpoint whose B is
    all zeros is indistinguishable from one that failed to load.
    """
    half = _Half()
    add_lora(half, _lora_cfg())
    sd = get_lora_state_dict(half)
    g = torch.Generator().manual_seed(seed)
    for k, v in sd.items():
        if "lora_B" in k:
            sd[k] = torch.randn(v.shape, generator=g, dtype=v.dtype)
    path = tmp_path / name
    # fp32 so the reload compares exactly; the fp16 default is pinned by
    # test_save_dtype_default_is_fp16 below.
    save_lora_safetensors(
        sd,
        {"rank": _RANK, "alpha": _ALPHA, "adapter_type": "lora", "target": "decoder"},
        path,
        dtype=torch.float32,
    )
    return path, sd


def _dit_ckpt(tmp_path, name="dit.safetensors"):
    model = _Half()
    cond = _Half()
    add_lora(model, _lora_cfg())
    add_lora(cond, _lora_cfg())
    sd = {**get_lora_state_dict(model), **get_lora_state_dict(cond)}
    path = tmp_path / name
    save_lora_safetensors(
        sd, {"rank": _RANK, "alpha": _ALPHA, "adapter_type": "lora"}, path
    )
    return path


# ---------------------------------------------------------------------------
# target resolution
# ---------------------------------------------------------------------------


def test_absent_target_means_dit():
    """Back-compat: every checkpoint published before `target` existed."""
    m = _FullModel()
    mods, prefixes = _resolve_lora_targets(m, "diffusion_cond", "dit")

    assert mods == [m.model, m.conditioner]
    assert prefixes == ["model.", "conditioner."]


def test_dit_target_uncond_model():
    m = _FullModel()
    mods, prefixes = _resolve_lora_targets(m, "diffusion_uncond", "dit")

    assert mods == [m]
    assert prefixes == ["model."]


@pytest.mark.parametrize("half", ["decoder", "encoder"])
def test_autoencoder_target_on_full_model(half):
    """Resolves through the pretransform, not the DiT."""
    m = _FullModel()
    mods, prefixes = _resolve_lora_targets(m, "diffusion_cond", half)

    assert mods == [getattr(m.pretransform.model, half)]
    assert prefixes == [f"{half}."]


@pytest.mark.parametrize("half", ["decoder", "encoder"])
def test_autoencoder_target_on_standalone_ae(half):
    """The same checkpoint has to load in the training/eval harness too."""
    ae = _AE()
    mods, prefixes = _resolve_lora_targets(ae, "autoencoder", half)

    assert mods == [getattr(ae, half)]
    assert prefixes == [f"{half}."]


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="unknown LoRA target"):
        _resolve_lora_targets(_FullModel(), "diffusion_cond", "bogus")


def test_autoencoder_target_without_autoencoder_raises():
    """A decoder LoRA pointed at a model that has no autoencoder must not
    silently fall through to the DiT."""

    class _NoAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Half()
            self.conditioner = _Half()

    with pytest.raises(ValueError, match="needs either an autoencoder"):
        _resolve_lora_targets(_NoAE(), "diffusion_cond", "decoder")


# ---------------------------------------------------------------------------
# the silent-miss regression
# ---------------------------------------------------------------------------


def test_decoder_lora_loads_onto_full_model(tmp_path):
    path, saved = _decoder_ckpt(tmp_path)
    m = _FullModel()

    load_and_apply_loras(m, [str(path)], "diffusion_cond")

    loaded = get_lora_state_dict(m.pretransform.model.decoder)
    assert loaded, "no adapter on the decoder"
    for k, v in saved.items():
        assert k in loaded, f"{k} missing after load"
        torch.testing.assert_close(loaded[k], v.to(loaded[k].dtype))

    # ...and the DiT was left alone.
    assert not get_lora_state_dict(m.model)
    assert not get_lora_state_dict(m.conditioner)


def test_decoder_lora_stacked_after_dit_loras(tmp_path):
    """The regression: parametrization slot is per-target, not the global index.

    A decoder adapter loaded third sits at DECODER slot 0 while its global
    index is 2. Keying off the global index makes every key miss, and
    strict=False turns that into a successful-looking no-op.
    """
    dit_a = _dit_ckpt(tmp_path, "a.safetensors")
    dit_b = _dit_ckpt(tmp_path, "b.safetensors")
    dec, saved = _decoder_ckpt(tmp_path, "c.safetensors", seed=7)
    m = _FullModel()

    load_and_apply_loras(m, [str(dit_a), str(dit_b), str(dec)], "diffusion_cond")

    loaded = get_lora_state_dict(m.pretransform.model.decoder)
    # Slot 0, because it is the first adapter on the DECODER.
    assert any(".parametrizations.weight.0.lora_B" in k for k in loaded)
    assert not any(".parametrizations.weight.2." in k for k in loaded)

    for k, v in saved.items():
        assert k in loaded, f"{k} missing — remapped to the global index?"
        torch.testing.assert_close(loaded[k], v.to(loaded[k].dtype))

    # The two DiT adapters still stacked normally, at slots 0 and 1.
    dit_keys = get_lora_state_dict(m.model)
    assert any(".parametrizations.weight.0." in k for k in dit_keys)
    assert any(".parametrizations.weight.1." in k for k in dit_keys)


def test_decoder_lora_changes_output(tmp_path):
    """The end that matters: the adapter has to move the audio path."""
    path, _ = _decoder_ckpt(tmp_path, seed=3)
    m = _FullModel()
    x = torch.randn(2, 8)

    before = m.pretransform.model.decoder(x).clone()
    load_and_apply_loras(m, [str(path)], "diffusion_cond")
    after = m.pretransform.model.decoder(x)

    assert not torch.allclose(before, after), "adapter attached but inert"


# ---------------------------------------------------------------------------
# weight_norm
# ---------------------------------------------------------------------------


def test_weight_norm_layer_is_skipped_not_fatal():
    """Deprecated weight_norm makes `weight` a hook product — not a Parameter,
    not a buffer, not parametrized — so register_parametrization raises. The
    autoencoder contains such layers; skipping them is what lets a LoRA aimed
    at that tree load at all."""

    class _Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = nn.Linear(8, 8)
            self.normed = nn.utils.weight_norm(nn.Linear(8, 8))

    m = _Mixed()
    add_lora(m, _lora_cfg())  # must not raise

    keys = get_lora_state_dict(m)
    assert any(k.startswith("plain.") for k in keys), "plain layer got no adapter"
    assert not any(k.startswith("normed.") for k in keys), (
        "weight_norm'd layer should have been skipped"
    )


# ---------------------------------------------------------------------------
# storage dtype
# ---------------------------------------------------------------------------


def test_save_dtype_default_is_fp16(tmp_path):
    """Unchanged default, because every published checkpoint is fp16."""
    from stable_audio_3.models.lora.utils import load_lora_checkpoint

    half = _Half()
    add_lora(half, _lora_cfg())
    path = tmp_path / "default.safetensors"
    save_lora_safetensors(get_lora_state_dict(half), {"rank": _RANK}, path)

    sd, _ = load_lora_checkpoint(path)
    assert all(v.dtype == torch.float16 for v in sd.values())


def test_save_dtype_fp32_round_trips_exactly(tmp_path):
    """fp32 is the option that matters for encoder adapters, where fp16's
    ~5e-4 relative precision gets amplified by the layer stack."""
    from stable_audio_3.models.lora.utils import load_lora_checkpoint

    half = _Half()
    add_lora(half, _lora_cfg())
    sd = get_lora_state_dict(half)
    g = torch.Generator().manual_seed(11)
    sd = {
        k: torch.randn(v.shape, generator=g, dtype=torch.float32) for k, v in sd.items()
    }

    path = tmp_path / "fp32.safetensors"
    save_lora_safetensors(sd, {"rank": _RANK}, path, dtype=torch.float32)

    reloaded, _ = load_lora_checkpoint(path)
    for k, v in sd.items():
        assert reloaded[k].dtype == torch.float32
        torch.testing.assert_close(reloaded[k], v, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# shipped scripts import
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        "train_decoder_lora.py",
        "train_encoder_lora.py",
        "gen_dit_latents.py",
        "pre_encode_dataset.py",
        "_decoder_lora_losses.py",
        "_decoder_lora_eval.py",
    ],
)
def test_decoder_lora_scripts_import(script):
    """Every shipped script must import on a clean checkout.

    Ruff does not resolve imports, so a module that exists only in a working
    tree -- a local shim, a probe left behind -- passes lint and then fails at
    the first line for anyone else. That is exactly how `gen_dit_latents.py`
    shipped importing a machine-specific helper that was never committed.
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / script
    assert path.exists(), f"{script} missing"

    r = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy, sys; sys.argv=['{script}', '--help']; "
            f"runpy.run_path({str(path)!r}, run_name='not_main')",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert r.returncode == 0, f"{script} failed to import:\n{r.stderr}"


def test_decoder_lora_docs_reference_real_files():
    """Every repo path named by this work's files must actually exist.

    Prose gets moved between branches more freely than code does, and a citation
    of a script that lives only in someone's working tree is invisible to lint,
    to the type checker and to every other test -- it is found by a reader who
    goes looking for the file and cannot find it. Both of the ones this catches
    shipped that way.

    Scoped to the decoder-LoRA files on purpose. Elsewhere in the repo, some
    paths are written relative to their own subproject rather than to the repo
    root and resolve fine from there, so a repo-wide version of this check would
    report dozens of things that are not broken.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    owned = [
        "stable_audio_3/models/lora/loader.py",
        "stable_audio_3/models/lora/model.py",
        "stable_audio_3/models/lora/utils.py",
        "scripts/train_decoder_lora.py",
        "scripts/_decoder_lora_losses.py",
        "scripts/_decoder_lora_eval.py",
        "scripts/gen_dit_latents.py",
        "scripts/train_encoder_lora.py",
        "scripts/pre_encode_dataset.py",
        "docs/workflows/decoder-lora.md",
        "docs/workflows/encoder-lora.md",
        "tests/test_lora_targets.py",
    ]
    pattern = re.compile(
        r"\b(?:scripts|tests|docs|stable_audio_3|pipelines)/[A-Za-z0-9_./-]+"
        r"\.(?:py|md|json|sh|toml)\b"
    )

    missing = []
    for rel in owned:
        f = root / rel
        assert f.exists(), f"{rel} is listed here but not in the repo"
        # Explicit encoding: read_text() defaults to the locale's, which is
        # cp1252 on Windows, and several of these files are UTF-8 (the dataset
        # layout in pre_encode_dataset.py's docstring uses U+2190). That made
        # this test pass on Linux and die on Windows with a UnicodeDecodeError
        # from inside pathlib -- an encoding failure wearing a missing-file
        # test's name.
        for ref in sorted(set(pattern.findall(f.read_text(encoding="utf-8")))):
            if not (root / ref).exists():
                missing.append(f"{rel} -> {ref}")

    assert not missing, "references to files that do not exist:\n  " + "\n  ".join(
        missing
    )


# ---------------------------------------------------------------------------
# strength control reaching the pretransform
# ---------------------------------------------------------------------------


def _strengths(module):
    """Every lora_strength currently set on a module tree.

    Uses the library's own parametrization walk rather than a second
    implementation of it, so the test cannot pass by agreeing with itself.
    """
    from stable_audio_3.models.lora.model import _iter_lora_params

    return [float(p.lora_strength) for p in _iter_lora_params(module)]


class _FakeStableAudioModel:
    """The attribute shape StableAudioModel.set_lora_strength walks.

    Bound to the real method rather than reimplementing it, so this exercises the
    shipped code and not a copy of it.
    """

    def __init__(self, with_pretransform=True):
        self.model = _FullModel() if with_pretransform else _DiTOnly()

    set_lora_strength = StableAudioModel.set_lora_strength


class _DiTOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Half()
        self.conditioner = _Half()


def test_set_lora_strength_reaches_the_decoder():
    """The line that makes "strength 0 renders at the stock decode" true.

    Without it an autoencoder adapter is stuck at whatever strength it was built
    with. Since the built-in default is 1.0, the adapter looks fine and only
    turning it DOWN silently fails -- so nothing louder than this would notice.
    """
    m = _FakeStableAudioModel()
    dec = m.model.pretransform.model.decoder
    add_lora(dec, _lora_cfg())
    add_lora(m.model.model, _lora_cfg())

    assert _strengths(dec), "fixture attached no adapter to the decoder"

    m.set_lora_strength(0.0)
    assert all(s == 0.0 for s in _strengths(dec)), "decoder never got the update"
    assert all(s == 0.0 for s in _strengths(m.model.model)), "DiT never got it"

    m.set_lora_strength(0.5)
    assert all(s == 0.5 for s in _strengths(dec))


def test_set_lora_strength_is_index_selective_on_the_decoder():
    """Stacked adapters have to stay independently controllable on the AE too."""
    m = _FakeStableAudioModel()
    dec = m.model.pretransform.model.decoder
    add_lora(dec, _lora_cfg(lora_index=0))
    add_lora(dec, _lora_cfg(lora_index=1))

    m.set_lora_strength(1.0)
    m.set_lora_strength(0.0, lora_index=0)

    seen = sorted(set(_strengths(dec)))
    assert seen == [0.0, 1.0], f"expected one adapter off and one on, got {seen}"


def test_set_lora_strength_without_a_pretransform():
    """An uncond/DiT-only model must not trip over the autoencoder lookup."""
    m = _FakeStableAudioModel(with_pretransform=False)
    add_lora(m.model.model, _lora_cfg())

    m.set_lora_strength(0.0)  # must not raise

    assert all(s == 0.0 for s in _strengths(m.model.model))
