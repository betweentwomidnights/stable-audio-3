"""Measure latent + audio magnitude with/without LoRA at fixed seed.

Generates the same prompt three ways (no LoRA, v3, v4) on the ARC medium model,
prints latent/audio stats, and saves .wav files for ear A/B.
"""
import os
import torch
import torchaudio
from stable_audio_3.pipeline import StableAudioPipeline

PROMPT = "downtempo electronic, ambient pads, 95 bpm, A minor"
SEED = 42
DURATION = 10
STEPS = 8
CFG = 1.0
SR = 44100

LORAS = {
    "v4_hot_latents":   "/workspace/sa3/lora_out_saos_v4_r16_trunc/lora_step3000.safetensors",
    "v6_rescaled_g026": "/workspace/sa3/lora_out_saos_v6_r16_g026/lora_step3000.safetensors",
}
OUT_DIR = "/workspace/sa3/loudness_test"
os.makedirs(OUT_DIR, exist_ok=True)

pipe = StableAudioPipeline.from_pretrained("medium", model_half=True)
pipe.load_lora(list(LORAS.values()))


def stats(name, latents, audio):
    a = audio.float().detach().cpu()
    l = latents.float().detach().cpu()
    rms = a.pow(2).mean().sqrt().item()
    dbfs = 20 * torch.log10(torch.tensor(rms + 1e-10)).item()
    peak = a.abs().max().item()
    over = (a.abs() > 1.0).float().mean().item() * 100
    print(
        f"{name:22s}  latent[max={l.abs().max().item():6.3f} std={l.std().item():.3f}]  "
        f"audio[peak={peak:.3f} rms={rms:.4f} ({dbfs:+.2f}dBFS) clipped={over:5.2f}%]"
    )


def run(tag, strengths):
    for i, s in enumerate(strengths):
        pipe.set_lora_strength(s, lora_index=i)

    torch.manual_seed(SEED)
    latents = pipe.generate(
        prompt=PROMPT, duration=DURATION, steps=STEPS, cfg_scale=CFG,
        seed=SEED, return_latents=True,
    )
    torch.manual_seed(SEED)
    audio = pipe.generate(
        prompt=PROMPT, duration=DURATION, steps=STEPS, cfg_scale=CFG, seed=SEED,
    )

    if isinstance(audio, list):
        audio = audio[0]
    if audio.dim() == 3:
        audio_for_save = audio[0]
    else:
        audio_for_save = audio

    stats(tag, latents, audio)
    wav = audio_for_save.clamp(-1, 1).float().cpu()
    torchaudio.save(os.path.join(OUT_DIR, f"loudness_{tag}.wav"), wav, SR)


print(f"PROMPT: {PROMPT}")
print(f"SEED={SEED}  STEPS={STEPS}  CFG={CFG}  DURATION={DURATION}s  MODEL=medium(ARC)\n")

run("no_lora",         [0.0, 0.0])
run("v4_hot_s1",       [1.0, 0.0])
run("v6_rescaled_s1",  [0.0, 1.0])
print("\nWAVs in loudness_test/ for ear A/B.")
