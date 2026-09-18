"""INCLUDE dataset download and coverage check (PRD §7 M2, §10 risk 1).

    python -m ml.data.download_include --fetch E:/datasets/INCLUDE --workers 4
    python -m ml.data.download_include --include-dir E:/datasets/INCLUDE/extracted

INCLUDE is on Zenodo under CC-BY-4.0 and is openly downloadable through the record API —
no accept step, despite what its landing page implies. 46 archives, 56.75 GB.

`--fetch` downloads and verifies them; `--include-dir` reports how many of our glosses an
extracted copy actually covers.

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
ZENODO_API = "https://zenodo.org/api/records/4010759"
RECORD_GB = 56.75


def fetch_record(dest: Path, workers: int = 4) -> int:
    """Download every archive, verifying each against its published size.

    Downloads to a .part file and renames only on a size match, so an interrupted
    transfer can never be mistaken for a finished one. Re-running skips what is already
    complete and repairs anything that is not, which makes this safe to restart.

    Do NOT resume with `curl -C -` against these: resuming onto an already-complete file
    appends a second copy, producing an archive larger than the original that still opens
    far enough to look valid, and fails at extraction much later.
    """
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API) as r:
        files = {f["key"]: f for f in json.load(r).get("files", [])}

    todo = []
    for name, meta in files.items():
        if not name.endswith(".zip"):
            continue
        target = dest / name
        if target.exists() and target.stat().st_size == meta["size"]:
            continue
        todo.append((name, meta))

    total = sum(m["size"] for _, m in todo)
    print(f"{len(files)} archives in the record ({RECORD_GB} GB); "
          f"{len(todo)} to fetch ({total/1e9:.1f} GB)")
    if not todo:
        print("everything already present and the right size")
        return 0

    def one(item) -> str:
        name, meta = item
        target, part = dest / name, dest / (name + ".part")
        url = f"{ZENODO_API}/files/{name}/content"
        try:
            with urllib.request.urlopen(url) as r, part.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as e:
            part.unlink(missing_ok=True)
            return f"FAILED {name}: {e}"
        if part.stat().st_size != meta["size"]:
            got = part.stat().st_size
            part.unlink(missing_ok=True)
            return f"SIZE MISMATCH {name}: got {got}, want {meta['size']}"
        part.replace(target)
        return f"ok {name} ({meta['size']/1e6:.0f} MB)"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for msg in pool.map(one, todo):
            print(" ", msg, flush=True)
    return 0


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
    ap.add_argument("--fetch", type=Path, metavar="DIR",
                    help="download every archive to DIR, verifying sizes (~57 GB)")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (Zenodo throttles per connection)")
    ap.add_argument("--append", action="store_true", help="add matched clips to manifest.csv as source=include")
    ap.add_argument("--pack", type=Path, default=ROOT / "backend" / "vocab" / "isl_v1.json")
    args = ap.parse_args()

    if args.fetch:
        return fetch_record(args.fetch, args.workers)

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
