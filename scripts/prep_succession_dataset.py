"""One-shot: build ~/sa3_training_succession/ from ~/ace/training/succession/.

For each <stem>.wav + <stem>.txt (canonical ace sidecar) pair:
  - write SA3-format <stem>.txt = "<caption>. <bpm> bpm. Key of <key>."
  - symlink the .wav into the staging dir.

Idempotent — safe to re-run after adjusting fields.
"""

import os
import re
from pathlib import Path

SRC = Path("/home/kev/ace/training/succession")
DST = Path("/home/kev/sa3_training_succession")


def parse_ace_sidecar(text: str) -> dict:
    """Parse canonical-format ace sidecar. `lyrics:` is a multi-line trailing block."""
    fields = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(caption|genre|bpm|key|keyscale|signature|timesignature|is_instrumental|custom_tag):\s*(.*)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
        elif line.strip().startswith("lyrics:"):
            # consume to EOF
            fields["lyrics"] = "\n".join(lines[i + 1:]).strip()
            break
        i += 1
    return fields


def build_sa3_caption(meta: dict) -> str:
    """SA3 caption = short genre tag + bpm + key.

    Rationale: SA3's T5 conditioner caps at 256 tokens. The verbose ace `caption:`
    prose averages 400-500 words (~500-700 T5 tokens) which gets truncated and
    silently drops the bpm/key tail (see feedback_overfit_lora_blending_value +
    project_sa3_succession_lora). Using the short `genre:` tag instead keeps the
    full caption under the cap so bpm/key actually reach the model.
    """
    parts = []
    genre = (meta.get("genre") or "").strip()
    if genre:
        parts.append(genre.rstrip(". "))
    bpm = (meta.get("bpm") or "").strip()
    if bpm:
        parts.append(f"{bpm} bpm")
    key = (meta.get("key") or meta.get("keyscale") or "").strip()
    if key:
        parts.append(f"Key of {key}")
    return ". ".join(parts) + "."


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    sidecars = sorted(SRC.glob("*.txt"))
    print(f"Found {len(sidecars)} sidecars in {SRC}")

    n_link = 0
    n_caption = 0
    n_skip = 0
    for sc in sidecars:
        stem = sc.stem
        wav_src = SRC / f"{stem}.wav"
        if not wav_src.exists():
            print(f"  SKIP (no wav): {stem}")
            n_skip += 1
            continue

        meta = parse_ace_sidecar(sc.read_text())
        caption = build_sa3_caption(meta)
        if not caption.strip("."):
            print(f"  SKIP (empty caption): {stem}")
            n_skip += 1
            continue

        (DST / f"{stem}.txt").write_text(caption + "\n")
        n_caption += 1

        wav_dst = DST / f"{stem}.wav"
        if wav_dst.exists() or wav_dst.is_symlink():
            wav_dst.unlink()
        os.symlink(wav_src, wav_dst)
        n_link += 1

    print(f"\nDone: {n_caption} captions, {n_link} symlinks, {n_skip} skipped")
    n_wav = len(list(DST.glob("*.wav")))
    n_txt = len(list(DST.glob("*.txt")))
    print(f"Staging dir totals: {n_wav} wav, {n_txt} txt")


if __name__ == "__main__":
    main()
