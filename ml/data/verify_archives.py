"""Check downloaded archives against the repository's published sizes and checksums.

    python -m ml.data.verify_archives --dir E:/datasets/INCLUDE
    python -m ml.data.verify_archives --dir E:/datasets/INCLUDE --delete-bad

Resumed downloads go wrong quietly. `curl -C -` against a file that is already complete,
or one left half-written by a killed process, can append instead of continuing — leaving
an archive LARGER than the original that still opens far enough to look valid. Extraction
then fails with "bad magic number" much later, or worse, yields partial data.

Size is checked first because it is instant and catches the common case. `--checksum`
additionally verifies the MD5, which is slow but conclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/{record}"


def fetch_expected(record: str, dest: Path) -> dict:
    """Published size and checksum per file, cached next to the archives."""
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    with urllib.request.urlopen(ZENODO_API.format(record=record)) as r:
        data = json.load(r)
    out = {f["key"]: {"size": f["size"], "checksum": f.get("checksum", "")}
           for f in data.get("files", [])}
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--record", default="4010759")
    ap.add_argument("--checksum", action="store_true", help="also verify MD5 (slow)")
    ap.add_argument("--delete-bad", action="store_true",
                    help="delete archives that are oversized or fail checksum, so they re-download cleanly")
    args = ap.parse_args()

    expected = fetch_expected(args.record, args.dir / "expected.json")
    complete, partial, bad = [], [], []

    for name, meta in sorted(expected.items()):
        if not name.endswith(".zip"):
            continue
        path = args.dir / name
        if not path.exists():
            continue
        have, want = path.stat().st_size, meta["size"]
        if have == want:
            if args.checksum and meta["checksum"].startswith("md5:"):
                if md5_of(path) != meta["checksum"][4:]:
                    bad.append((name, "checksum mismatch"))
                    continue
            complete.append(name)
        elif have > want:
            # The dangerous case: a resumed download appended to a finished file.
            bad.append((name, f"oversized {have/want:.0%} — resume appended to a complete file"))
        else:
            partial.append((name, have / want))

    print(f"complete : {len(complete)}")
    print(f"partial  : {len(partial)} (still downloading)")
    print(f"CORRUPT  : {len(bad)}")
    for name, why in bad:
        print(f"    {name}  — {why}")

    if bad and args.delete_bad:
        for name, _ in bad:
            (args.dir / name).unlink()
            print(f"deleted {name}")
        print("\nre-download these; they will start clean rather than resume.")
    elif bad:
        print("\nrun again with --delete-bad to remove them for a clean re-download.")

    missing = [n for n in expected if n.endswith(".zip") and not (args.dir / n).exists()]
    if missing:
        print(f"\nnot started: {len(missing)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
