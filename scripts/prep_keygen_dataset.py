"""One-shot: build ~/sa3_training_keygen/ from ~/audiocraft-dgx/keygen_music_v2/.

For each <stem>.wav + <stem>.json pair:
  - write <stem>.txt caption = "<genre>. <moods>. <bpm> bpm. Key of <key>."
  - symlink the .wav into the staging dir (except the 10-min "on repeat" track,
    which is trimmed to TRIM_SECONDS and written as a real .wav).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

SRC = Path("/home/kev/audiocraft-dgx/keygen_music_v2")
DST = Path("/home/kev/sa3_training_keygen")
TRIM_STEM = "10 Minutes of Best Keygen Sound"
TRIM_SECONDS = 90


def caption_from_json(meta: dict) -> str:
    parts = []
    genre = (meta.get("genre") or "").strip()
    if genre:
        parts.append(genre)
    moods = meta.get("moods") or []
    if isinstance(moods, list) and moods:
        parts.append(", ".join(str(m).strip() for m in moods if str(m).strip()))
    bpm = meta.get("bpm")
    if bpm:
        parts.append(f"{int(round(float(bpm)))} bpm")
    key = (meta.get("key") or "").strip()
    if key:
        parts.append(f"Key of {key}")
    return ". ".join(parts) + "."


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    jsons = sorted(SRC.glob("*.json"))
    print(f"Found {len(jsons)} json files in {SRC}")

    n_link = 0
    n_trim = 0
    n_caption = 0
    for jp in jsons:
        stem = jp.stem
        wav_src = SRC / f"{stem}.wav"
        if not wav_src.exists():
            print(f"  SKIP (no wav): {stem}")
            continue

        # caption
        with open(jp) as f:
            meta = json.load(f)
        caption = caption_from_json(meta)
        (DST / f"{stem}.txt").write_text(caption + "\n")
        n_caption += 1

        # wav: symlink or trim
        wav_dst = DST / f"{stem}.wav"
        if wav_dst.exists() or wav_dst.is_symlink():
            wav_dst.unlink()
        if stem == TRIM_STEM:
            print(f"  TRIM: {stem} -> first {TRIM_SECONDS}s")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(wav_src),
                    "-t", str(TRIM_SECONDS),
                    "-c:a", "pcm_s16le",
                    str(wav_dst),
                ],
                check=True,
            )
            n_trim += 1
        else:
            os.symlink(wav_src, wav_dst)
            n_link += 1

    print(f"\nDone: {n_caption} captions, {n_link} symlinks, {n_trim} trimmed")
    n_wav = len(list(DST.glob("*.wav")))
    n_txt = len(list(DST.glob("*.txt")))
    print(f"Staging dir totals: {n_wav} wav, {n_txt} txt")


if __name__ == "__main__":
    main()
