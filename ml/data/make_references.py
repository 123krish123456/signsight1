"""Build small, shareable reference clips for the recorder.

    python -m ml.data.make_references

The recorder shows the signer what to copy. That only works if the reference clips are
in the repository: the originals live in a 57 GB corpus on one machine, so a teammate who
clones the project would see an empty panel and be left inventing gestures.

Each reference is downscaled and re-encoded to a few hundred kilobytes, written to
`assets/reference/`, and the vocabulary pack is repointed at the repo-relative copy.
Resolution only has to be good enough for a human to copy the sign, not good enough for
landmark extraction, so this can be small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import ROOT, settings

OUT_DIR = ROOT / "assets" / "reference"
HEIGHT = 360  # plenty to copy a handshape from; keeps each clip well under a megabyte


def shrink(src: Path, dst: Path, height: int = HEIGHT, max_seconds: float = 4.0) -> int:
    """Re-encode one clip. Returns the written size in bytes, or 0 on failure."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not h:
        cap.release()
        return 0
    size = (int(w * height / h) // 2 * 2, height)  # even dimensions keep encoders happy

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    written = 0
    limit = int(fps * max_seconds)
    try:
        while written < limit:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, size))
            written += 1
    finally:
        cap.release()
        writer.release()
    return dst.stat().st_size if dst.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=settings.vocab_pack)
    ap.add_argument("--height", type=int, default=HEIGHT)
    args = ap.parse_args()

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    total = made = 0

    for entry in pack["entries"]:
        ref = entry.get("reference_video")
        if not ref:
            continue
        src = Path(ref)
        if not src.is_absolute():
            src = ROOT / ref
        if not src.exists():
            print(f"  missing source for {entry['gloss']}: {ref}")
            continue

        dst = OUT_DIR / f"{entry['gloss']}.mp4"
        size = shrink(src, dst, args.height)
        if not size:
            print(f"  FAILED {entry['gloss']}")
            continue
        entry["reference_video"] = dst.relative_to(ROOT).as_posix()
        total += size
        made += 1
        print(f"  {entry['gloss']:<14} {size/1000:>6.0f} KB")

    args.pack.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    print(f"\n{made} references, {total/1e6:.1f} MB total, written to {OUT_DIR.relative_to(ROOT)}")
    print(f"{args.pack.name} now points at repo-relative paths, so a fresh clone works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
