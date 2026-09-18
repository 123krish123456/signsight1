"""One idempotent command that takes downloaded dataset archives all the way to a
manifest. Safe to re-run at any point — it skips whatever is already done.

    python -m ml.data.prepare --archives E:/datasets/INCLUDE

Steps, each skipped if already complete:
  1. extract any .zip that has not been extracted yet
  2. locate the dataset's own train/test split CSVs, if it ships them
  3. probe a sample for detectable hands (a dataset whose hands cannot be seen is
     worthless no matter how large it is)
  4. ingest every matching clip into the manifest
  5. report coverage and what still blocks training

Intended to be run repeatedly while a long download is still in progress: each run
picks up the archives that have finished since the last one.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from backend.config import settings
from backend.vocab.schema import load_pack
from ml.data import ingest as ingest_mod
from ml.data.manifest import Clip, load, save, summary, verify, disclosures

MARKER = ".extracted"


def is_complete(zip_path: Path) -> bool:
    """A zip still being downloaded is not a valid archive yet."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            return z.testzip() is None or True  # openable == complete enough
    except (zipfile.BadZipFile, OSError):
        return False


def extract_new(archives: Path, dest: Path) -> tuple[int, int]:
    """Extract every complete, not-yet-extracted zip. Returns (extracted, skipped)."""
    dest.mkdir(parents=True, exist_ok=True)
    done_dir = dest / MARKER
    done_dir.mkdir(exist_ok=True)
    extracted = skipped = 0

    for zip_path in sorted(archives.glob("*.zip")):
        stamp = done_dir / (zip_path.stem + ".done")
        if stamp.exists():
            skipped += 1
            continue
        if not is_complete(zip_path):
            print(f"  still downloading, skipping: {zip_path.name}")
            continue
        print(f"  extracting {zip_path.name} ...", flush=True)
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest)
        except (zipfile.BadZipFile, OSError) as e:
            print(f"    failed: {e}")
            continue
        stamp.write_text(zip_path.name, encoding="utf-8")
        extracted += 1
    return extracted, skipped


def find_split_csvs(root: Path) -> list[Path]:
    """The dataset's own splits, if any. Theirs beat ours: they usually know which
    videos share a signer, and we are guessing from filenames."""
    return [p for p in root.rglob("*.csv")
            if any(k in p.as_posix().lower() for k in ("train", "test", "split", "val"))][:10]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archives", type=Path, required=True, help="directory holding the .zip files")
    ap.add_argument("--extract-to", type=Path, default=None, help="default: <archives>/extracted")
    ap.add_argument("--source", default="include")
    ap.add_argument("--signer-pattern", default=None)
    ap.add_argument("--probe", type=int, default=6, help="0 to skip the usability probe")
    ap.add_argument("--no-write", action="store_true", help="report only, do not touch the manifest")
    args = ap.parse_args()

    dest = args.extract_to or (args.archives / "extracted")
    pack = load_pack(settings.vocab_pack)

    print("1. extracting archives")
    got, skipped = extract_new(args.archives, dest)
    print(f"   {got} newly extracted, {skipped} already done")

    print("\n2. dataset-provided splits")
    csvs = find_split_csvs(dest)
    if csvs:
        for c in csvs:
            print(f"   {c.relative_to(dest)}")
        print("   ^ prefer these over guessing signers from filenames")
    else:
        print("   none found yet")

    print("\n3. scanning for clips")
    clips, unmatched = ingest_mod.from_tree(
        dest, ingest_mod.VIDEO_EXT, pack, args.signer_pattern,
        ingest_mod.UNLABELLED, args.source,
    )
    if not clips:
        print("   no clips matching the vocabulary yet — more archives may still be extracting")
        if unmatched:
            top = sorted(unmatched.items(), key=lambda kv: kv[1], reverse=True)[:8]
            print(f"   {len(unmatched)} unmatched dataset classes so far, e.g. {[n for n, _ in top]}")
        return 0

    ingest_mod.report(clips, unmatched, pack, "videos")

    if args.probe:
        print("\n4. usability probe")
        ingest_mod.probe(clips, args.probe)

    if args.no_write:
        print("\n--no-write: manifest untouched")
        return 0

    print("\n5. manifest")
    existing = load()
    known = {c.clip for c in existing}
    added = [c for c in clips if c.clip not in known]
    save(existing + added)
    print(f"   added {len(added)} rows ({len(clips) - len(added)} already present)")

    everything = load()
    print("\n" + summary(everything))
    problems = verify(everything, {e.gloss for e in pack.entries})
    if problems:
        print("\nstill blocking training:")
        for p in problems[:8]:
            print(f"   - {p}")
    for note in disclosures(everything):
        print(f"\n   disclose: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
