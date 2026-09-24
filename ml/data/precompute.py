"""Extract landmarks for every clip in the manifest, in parallel, once.

    python -m ml.data.precompute            # all splits
    python -m ml.data.precompute --workers 4 --limit 50

MediaPipe is single-threaded and spends a second or more per clip, so doing this inside
the training loop would waste hours on every run. Features are cached as .npy keyed by
path and mtime, so this is resumable: interrupt it, run it again, and it picks up the
clips it has not done.

Clips where the hands are never detected are reported but not dropped — that decision
belongs to whoever reads the number, not to a cache-filling script.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ml.data.manifest import Clip, load
from ml.dataset import CACHE, features_for_clip


def _cache_key(clip: str) -> Path:
    path = Path(clip)
    try:
        stamp = int(path.stat().st_mtime)
    except OSError:
        stamp = 0
    return CACHE / f"{path.stem}_{stamp}.npy"


def one(clip: str) -> tuple[str, int, float, str]:
    """Returns (clip, n_frames, hand_rate, error)."""
    from ml.features.extract import HAND_DIM, POSE_DIM

    try:
        seq = features_for_clip(clip)
    except Exception as e:  # a corrupt file must not kill the whole run
        return clip, 0, 0.0, f"{type(e).__name__}: {e}"
    if len(seq) == 0:
        return clip, 0, 0.0, "no frames"
    hands = float(np.any(seq[:, POSE_DIM : POSE_DIM + 2 * HAND_DIM] != 0, axis=1).mean())
    return clip, len(seq), hands, ""


def report(clips: list[Clip]) -> int:
    """Hand and face tracking rate per signer.

    The one number that says whether a batch of footage is worth anything. Arpit's first
    265 clips averaged 34% against the public corpus's 90%, because his hands left the
    bottom of the frame — invisible in every other check, and fatal to the fold.
    """
    from collections import defaultdict

    from ml.features.extract import FEATURE_DIM, HAND_DIM, POSE_DIM

    by: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for c in clips:
        seq = features_for_clip(c.clip)
        if len(seq) == 0:
            by[c.signer].append((0.0, 0.0, 0))
            continue
        hands = np.any(seq[:, POSE_DIM : POSE_DIM + 2 * HAND_DIM] != 0, axis=1)
        face = np.any(seq[:, POSE_DIM + 2 * HAND_DIM : FEATURE_DIM] != 0, axis=1)
        by[c.signer].append((float(hands.mean()), float(face.mean()), len(seq)))

    print(f"{'signer':<12} {'clips':>6} {'hands':>7} {'face':>7} {'frames':>7}")
    for signer in sorted(by):
        rows = by[signer]
        hands = float(np.mean([r[0] for r in rows]))
        flag = "  <- suspect" if hands < 0.70 else ""
        print(f"{signer:<12} {len(rows):>6} {hands:>6.0%} "
              f"{np.mean([r[1] for r in rows]):>6.0%} "
              f"{np.mean([r[2] for r in rows]):>7.0f}{flag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--report", action="store_true",
                    help="summarise the cache per signer instead of extracting. A signer "
                         "well below the others is a recording problem, not a hard signer.")
    args = ap.parse_args()

    clips: list[Clip] = load()
    if args.split:
        clips = [c for c in clips if c.split == args.split]
    if not clips:
        raise SystemExit("manifest is empty — run `python -m ml.data.prepare` first")

    if args.report:
        return report(clips)

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [c.clip for c in clips if not _cache_key(c.clip).exists()]
    cached = len(clips) - len(todo)
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(clips)} clips: {cached} already cached, {len(todo)} to extract")
    if not todo:
        print("nothing to do")
        return 0

    t0 = time.monotonic()
    done = failed = 0
    empty_hands: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, c): c for c in todo}
        for fut in as_completed(futures):
            clip, n, hands, err = fut.result()
            done += 1
            if err:
                failed += 1
                if failed <= 5:
                    print(f"  FAILED {Path(clip).name}: {err}")
            elif hands == 0.0:
                empty_hands.append(clip)
            if done % 25 == 0 or done == len(todo):
                rate = done / max(time.monotonic() - t0, 1e-6)
                left = (len(todo) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(todo)}  {rate:.1f} clips/s  ~{left/60:.0f} min left", flush=True)

    secs = time.monotonic() - t0
    print(f"\nextracted {done - failed} clips in {secs/60:.1f} min ({failed} failed)")
    if empty_hands:
        pct = len(empty_hands) / max(done, 1)
        print(f"{len(empty_hands)} clips ({pct:.0%}) had no hands detected in any frame.")
        print("Those carry almost no signal — check the footage before trusting a model trained on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
