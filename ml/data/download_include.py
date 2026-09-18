"""INCLUDE dataset coverage check (PRD §7 M2, §10 risk 1).

INCLUDE is distributed on Zenodo behind a manual accept — there is no stable direct
download URL, so this tool does the part that actually matters: point it at an extracted
copy and it reports how many Appendix A glosses are really covered.

    python -m ml.data.download_include --include-dir D:/datasets/INCLUDE
    python -m ml.data.download_include --include-dir D:/datasets/INCLUDE --append

Risk-table decision rule: if fewer than 15 of the 24 words match, INCLUDE is not worth
wiring in — self-recording (`ml.data.record`) is the primary source either way.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from backend.config import ROOT
from backend.vocab.schema import load_pack
from ml.data.manifest import Clip, load, save

ZENODO = "https://zenodo.org/records/4010759"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
COVERAGE_FLOOR = 15  # PRD §10


def normalise(name: str) -> str:
    """INCLUDE folder names look like '1. Adjectives/2. loud' — reduce to a gloss key."""
    stem = re.sub(r"^\d+[\.\-_ ]*", "", name).strip()
    return re.sub(r"[^A-Z]+", "", stem.upper().replace("THANK YOU", "THANKYOU"))


def scan(include_dir: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for video in include_dir.rglob("*"):
        if video.suffix.lower() in VIDEO_EXT:
            found.setdefault(normalise(video.parent.name), []).append(video)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-dir", type=Path, help=f"extracted INCLUDE root (get it from {ZENODO})")
    ap.add_argument("--append", action="store_true", help="add matched clips to manifest.csv as source=include")
    ap.add_argument("--pack", type=Path, default=ROOT / "backend" / "vocab" / "isl_v1.json")
    args = ap.parse_args()

    if not args.include_dir:
        print(f"INCLUDE requires a manual download (accept the terms):\n  {ZENODO}\n"
              "Then re-run with --include-dir <extracted path>.")
        return 0
    if not args.include_dir.is_dir():
        print(f"not a directory: {args.include_dir}")
        return 1

    entries = load_pack(args.pack).entries
    words = [e.gloss for e in entries if e.pos != "letter"]
    found = scan(args.include_dir)

    matched = {g: found[normalise(g)] for g in words if normalise(g) in found}
    print(f"INCLUDE: {sum(len(v) for v in found.values())} videos, {len(found)} classes")
    print(f"covered {len(matched)}/{len(words)} Appendix A words:")
    for g in words:
        n = len(matched.get(g, []))
        print(f"  {'y' if n else '.'} {g:<12} {n}")
    print(f"\nletters: INCLUDE has no manual-alphabet classes — all 26 must be self-recorded.")

    if len(matched) < COVERAGE_FLOOR:
        print(f"\nWARNING: only {len(matched)} words matched (< {COVERAGE_FLOOR}). Per PRD §10, treat INCLUDE as "
              "augmentation only, or cut the vocabulary to what you can record.")

    if args.append:
        clips = load()
        known = {c.clip for c in clips}
        added = [
            Clip(clip=str(p), gloss=g, signer=f"include_{p.stem.split('_')[0]}", source="include")
            for g, paths in matched.items() for p in paths if str(p) not in known
        ]
        save(clips + added)
        print(f"\nappended {len(added)} clips (source=include). "
              "Their signer ids are folder-derived — verify them before splitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
