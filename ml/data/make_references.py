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
    """Re-encode one clip to browser-playable H.264. Returns bytes written, 0 on failure.

    OpenCV is not used for this. Its mp4v fourcc writes MPEG-4 Part 2, which no browser
    plays — the clips looked fine in the desktop tool and showed a black panel in Chrome.
    Its H.264 path needs an openh264 DLL that is usually absent, and its WebM tags fall
    back silently. ffmpeg, bundled by imageio-ffmpeg, just works.

    yuv420p and the faststart flag are both required for broad browser support.
    """
    import subprocess

    import imageio_ffmpeg

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(src),
        "-t", str(max_seconds),
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "30",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an",                     # no audio: nothing here needs it
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    ffmpeg: {proc.stderr.strip()[:200]}")
        return 0
    return dst.stat().st_size if dst.exists() else 0


def pick_sources() -> dict[str, str]:
    """One source clip per gloss, taken from the corpus rather than from the pack.

    Deriving sources from the manifest keeps this re-runnable. An earlier version read
    the pack's own reference_video and then overwrote it with the output path, so the
    second run had lost the originals and produced nothing.
    """
    import collections

    from ml.data.manifest import load

    by_gloss = collections.defaultdict(list)
    for c in load():
        if c.source == "include":          # never point a reference at our own recording
            by_gloss[c.gloss].append(c.clip)
    return {g: sorted(v)[len(v) // 2] for g, v in by_gloss.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=settings.vocab_pack)
    ap.add_argument("--height", type=int, default=HEIGHT)
    args = ap.parse_args()

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    sources = pick_sources()
    total = made = 0

    for entry in pack["entries"]:
        gloss = entry["gloss"]
        ref = sources.get(gloss)
        if not ref:
            print(f"  no corpus clip for {gloss}; leaving its reference alone")
            continue
        src = Path(ref)
        if not src.is_absolute():
            src = ROOT / ref
        if not src.exists():
            print(f"  missing source file for {gloss}: {ref}")
            continue

        dst = OUT_DIR / f"{gloss}.mp4"
        size = shrink(src, dst, args.height)
        if not size:
            print(f"  FAILED {gloss}")
            continue
        entry["reference_video"] = dst.relative_to(ROOT).as_posix()
        total += size
        made += 1
        print(f"  {gloss:<14} {size/1000:>6.0f} KB")

    args.pack.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    print(f"\n{made} references, {total/1e6:.1f} MB total, in {OUT_DIR.relative_to(ROOT)}")
    print("H.264 / yuv420p, so browsers can play them. Re-runnable: sources come from the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
