"""Reformat a dataset of (clip.wav, clip.txt) pairs for SA3 training.

For each pair:
  - symlink the .wav (no disk cost)
  - rewrite the .txt as `genre, N bpm, key` (comma-separated, single line),
    stripping the long `caption:` field so we fit SA3's 256-token prompt limit.

Idempotent — re-running overwrites the destination txt and symlink cleanly.

Usage:
  python scripts/reformat_to_sa3.py --src ~/saos_training --dst ~/sa3_training
  python scripts/reformat_to_sa3.py --src ~/ace/training/patch_staging --dst ~/sa3_training_patch
"""

import argparse
from pathlib import Path
import re
import sys


def parse_fields(text: str) -> dict[str, str]:
    """Pull `key: value` lines until the first blank line or non-`key:` line.
    Multi-line values (e.g. lyrics:) are not collected."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if not m:
            continue
        out[m.group(1)] = m.group(2).strip()
    return out


def reformat(fields: dict[str, str]) -> str:
    parts = []
    if fields.get("genre"):
        parts.append(fields["genre"])  # already comma-separated
    if fields.get("bpm"):
        parts.append(f"{fields['bpm']} bpm")
    if fields.get("key"):
        parts.append(fields["key"])
    return ", ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Source dir with .wav/.txt pairs.")
    p.add_argument("--dst", type=Path, required=True, help="Destination dir for symlinks and reformatted txt.")
    args = p.parse_args()

    src: Path = args.src.expanduser()
    dst: Path = args.dst.expanduser()
    if not src.exists():
        print(f"missing source: {src}", file=sys.stderr)
        return 1
    dst.mkdir(exist_ok=True, parents=True)

    pairs = 0
    skipped = 0
    for wav in sorted(src.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"skip (no txt): {wav.name}")
            skipped += 1
            continue

        fields = parse_fields(txt.read_text())
        prompt = reformat(fields)
        if not prompt:
            print(f"skip (empty prompt): {wav.name}")
            skipped += 1
            continue

        # symlink wav
        link = dst / wav.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(wav.resolve())

        # write reformatted txt
        (dst / txt.name).write_text(prompt + "\n")
        pairs += 1

    print(f"\nwrote {pairs} pairs to {dst} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
