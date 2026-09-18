"""List every class a set of archives contains, without extracting them.

    python -m ml.data.catalogue --dir E:/datasets/INCLUDE
    python -m ml.data.catalogue --dir E:/datasets/INCLUDE --suggest

A zip's central directory lists every path it holds, so the full vocabulary of a 57 GB
dataset can be read in seconds from archives that have finished downloading. Use it to
decide which classes to ship before committing to extraction and feature extraction.

`--suggest` additionally proposes vocabulary-pack aliases for dataset classes whose
names are close to one of ours, which is how "48. Hello" is matched to HELLO and how
you find out that the corpus calls the greeting something else entirely.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from collections import Counter
from difflib import get_close_matches
from pathlib import Path

from backend.config import settings
from backend.vocab.schema import load_pack
from ml.data.ingest import VIDEO_EXT, gloss_key, gloss_map

STRIP_INDEX = re.compile(r"^\d+[.\-_ ]*")


def classes_in(zip_path: Path) -> Counter:
    """class name → number of video files, read from the central directory."""
    found: Counter = Counter()
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if Path(name).suffix.lower() not in VIDEO_EXT:
                    continue
                parts = [p for p in name.split("/") if p]
                if len(parts) >= 2:
                    found[parts[-2]] += 1
    except (zipfile.BadZipFile, OSError):
        pass
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--suggest", action="store_true", help="propose aliases for near-miss names")
    ap.add_argument("--min-clips", type=int, default=0, help="only list classes with at least this many clips")
    args = ap.parse_args()

    archives = sorted(args.dir.glob("*.zip"))
    if not archives:
        raise SystemExit(f"no .zip files in {args.dir}")

    total: Counter = Counter()
    read = 0
    for z in archives:
        found = classes_in(z)
        if found:
            read += 1
            total.update(found)

    classes = {c: n for c, n in total.items() if n >= args.min_clips}
    print(f"read {read}/{len(archives)} archives — {len(classes)} classes, {sum(classes.values())} clips\n")

    pack = load_pack(settings.vocab_pack)
    mapping = gloss_map(pack)

    matched, unmatched = {}, {}
    for name, n in classes.items():
        gloss = mapping.get(gloss_key(name))
        (matched if gloss else unmatched)[name] = (gloss, n)

    print(f"MATCHED {len(matched)} of our {len(pack.entries)} classes:")
    for name, (gloss, n) in sorted(matched.items(), key=lambda kv: -kv[1][1]):
        print(f"  {gloss:<14} <- {name:<26} {n:>3} clips")

    covered = {g for g, _ in matched.values()}
    missing = [e.gloss for e in pack.entries if e.gloss not in covered]
    print(f"\nNOT COVERED ({len(missing)}): {', '.join(missing[:24])}"
          + (" ..." if len(missing) > 24 else ""))

    if args.suggest and missing:
        print("\nnear-miss suggestions — add as aliases in the vocab pack if correct:")
        names = {STRIP_INDEX.sub("", n).strip().lower(): n for n in unmatched}
        for gloss in missing:
            hits = get_close_matches(gloss.replace("-", " ").lower(), names, n=2, cutoff=0.72)
            for h in hits:
                print(f'  "{gloss}": aliases += ["{STRIP_INDEX.sub("", names[h]).strip()}"]'
                      f"   ({unmatched[names[h]][1]} clips)")

    print(f"\nthe corpus has {len(unmatched)} classes we do not use. The richest:")
    for name, (_, n) in sorted(unmatched.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {STRIP_INDEX.sub('', name).strip():<24} {n:>3} clips")
    print("\nPRD §10 allows cutting the vocabulary to what the data actually supports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
