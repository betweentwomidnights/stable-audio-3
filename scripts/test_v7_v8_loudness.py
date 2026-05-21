"""Measure generation loudness for base / v7 / v8 LoRAs across multiple prompts and seeds.

Loads v7 (r16 step5000) and v8 (r32 step5000) onto medium-ARC at the same time and
toggles strengths between configs. Generates 3 prompts x 2 seeds per config (18 gens),
prints per-generation peak/RMS/dBFS, and aggregates mean +/- std per config so a real
loudness bias is distinguishable from sample-to-sample variation.

Run inside the sa3:spark container with the venv python and PYTHONPATH set.
"""
import os
import statistics

import torch
import torchaudio

from stable_audio_3.pipeline import StableAudioPipeline

LORA_V6 = "/workspace/sa3/lora_out_saos_v6_r16_g026/lora_step3000.safetensors"
LORA_V7 = "/workspace/sa3/lora_out_patch_v7_r16_g046/lora_step5000.safetensors"
LORA_V8 = "/workspace/sa3/lora_out_patch_v8_r32_lnm-neg1/lora_step5000.safetensors"
OUT_DIR = "/workspace/sa3/loudness_test_v8"

PROMPTS = [
    "Math rock, 100 bpm, D minor",
    "Breakbeat, 174 bpm, B minor",
    "Lo-fi hip hop, 85 bpm, E major",
]
SEEDS = [42, 1337]

DURATION = 10
STEPS = 8
CFG = 1.0
SR = 44100

# Strengths are [v6, v7, v8] in load order.
CONFIGS = [
    ("no_lora", [0.0, 0.0, 0.0]),
    ("v6_s1",   [1.0, 0.0, 0.0]),
    ("v6_s2",   [2.0, 0.0, 0.0]),
    ("v7_s2",   [0.0, 2.0, 0.0]),
    ("v8_s2",   [0.0, 0.0, 2.0]),
]

os.makedirs(OUT_DIR, exist_ok=True)

pipe = StableAudioPipeline.from_pretrained("medium", model_half=True)
pipe.load_lora([LORA_V6, LORA_V7, LORA_V8])


def stats(audio):
    a = audio.float().detach().cpu()
    rms = a.pow(2).mean().sqrt().item()
    dbfs = 20 * torch.log10(torch.tensor(rms + 1e-10)).item()
    peak = a.abs().max().item()
    peak_dbfs = 20 * torch.log10(torch.tensor(peak + 1e-10)).item()
    over = (a.abs() > 1.0).float().mean().item() * 100
    return rms, dbfs, peak, peak_dbfs, over


def run_one(tag, strengths, prompt, seed):
    for i, s in enumerate(strengths):
        pipe.set_lora_strength(s, lora_index=i)

    torch.manual_seed(seed)
    audio = pipe.generate(
        prompt=prompt, duration=DURATION, steps=STEPS, cfg_scale=CFG, seed=seed,
    )
    if isinstance(audio, list):
        audio = audio[0]
    if audio.dim() == 3:
        audio = audio[0]

    rms, dbfs, peak, peak_dbfs, over = stats(audio)
    wav = audio.clamp(-1, 1).float().cpu()
    safe_prompt = prompt.split(",")[0].replace(" ", "_").lower()
    fname = f"{tag}__{safe_prompt}__seed{seed}.wav"
    torchaudio.save(os.path.join(OUT_DIR, fname), wav, SR)
    return rms, dbfs, peak, peak_dbfs, over


print(f"STEPS={STEPS}  CFG={CFG}  DURATION={DURATION}s  MODEL=medium(ARC)")
print(f"Prompts: {len(PROMPTS)}   Seeds: {SEEDS}   Configs: {[c[0] for c in CONFIGS]}\n")

header = f"{'config':10s} {'seed':>5s}  {'prompt':32s}  {'rms':>7s} {'dBFS':>8s}  {'peak':>6s} {'peak_dBFS':>10s}  {'clip%':>6s}"
print(header)
print("-" * len(header))

by_config = {tag: [] for tag, _ in CONFIGS}
for prompt in PROMPTS:
    for seed in SEEDS:
        for tag, strengths in CONFIGS:
            rms, dbfs, peak, peak_dbfs, over = run_one(tag, strengths, prompt, seed)
            by_config[tag].append((rms, dbfs, peak, peak_dbfs))
            print(
                f"{tag:10s} {seed:5d}  {prompt[:32]:32s}  "
                f"{rms:7.4f} {dbfs:+8.2f}  {peak:6.3f} {peak_dbfs:+10.2f}  {over:6.2f}"
            )
        print()

print("=" * len(header))
print(f"{'config':10s}  {'rms (mean +/- std)':>22s}  {'dBFS (mean +/- std)':>24s}  {'peak (mean)':>14s}")
print("-" * 80)
for tag, _ in CONFIGS:
    rmss   = [r[0] for r in by_config[tag]]
    dbfss  = [r[1] for r in by_config[tag]]
    peaks  = [r[2] for r in by_config[tag]]
    rms_m, rms_s     = statistics.mean(rmss),  statistics.stdev(rmss)  if len(rmss)  > 1 else 0.0
    dbfs_m, dbfs_s   = statistics.mean(dbfss), statistics.stdev(dbfss) if len(dbfss) > 1 else 0.0
    peak_m           = statistics.mean(peaks)
    print(
        f"{tag:10s}  {rms_m:8.4f} +/- {rms_s:7.4f}  "
        f"{dbfs_m:+8.2f} +/- {dbfs_s:7.2f}      {peak_m:8.3f}"
    )

print(f"\nWAVs in {OUT_DIR} for ear A/B.")
print("If v8 dBFS mean is noticeably below base/v7 (>1 dB beyond combined std), the")
print("loudness shift is real; per-track loudness normalization in pre-encoding is the")
print("next thing to try (currently we apply a single global --audio_gain).")
