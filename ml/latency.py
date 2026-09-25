"""End-to-end latency: how long after a sign finishes does its gloss appear? (PRD §7 M7)

    python -m ml.latency                  # 20 clips against a running backend
    python -m ml.latency --clips 60 --url ws://127.0.0.1:8000/ws/stream

M4's definition of done asks for this to be "measured and logged, not estimated", and the
budget is 600 ms. It has to be measured over a real socket at the real frame rate, because
every part of the number lives somewhere different: JSON encoding on the wire, the
segmenter's exit run waiting to confirm the sign has ended, the ONNX call, and whatever
the event loop was doing at the time.

Clips are replayed from the cached features at `target_fps`, which is what the browser
sends. Replaying as fast as possible would measure the server's throughput rather than
what a signer waits, and would report a number several times too good.

The clock starts at the last frame of the sign itself, not at the last frame sent. Those
differ by the whole reason the budget exists: the segmenter cannot know a sign has ended
until it has seen `exit_frames` of stillness through a `energy_smoothing_frames` window,
and that confirmation delay is time the signer spends waiting. Measuring from the last
frame sent hides it and reports single-digit milliseconds for a system that in fact takes
most of a second.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from backend.config import ROOT, settings

REPORTS = ROOT / "ml" / "reports"
BUDGET_MS = 600  # PRD M4 definition of done


def replay_clip(ws, frames: np.ndarray, fps: int, sign_ends_at: int) -> list[float]:
    """Send one clip at `fps`; return how long after the sign ended each gloss arrived.

    `sign_ends_at` is the index of the last frame of the sign proper. Everything after it
    is the still tail that lets the segmenter's exit condition fire, and the wait for that
    is precisely what the 600 ms budget is about.
    """
    interval = 1.0 / fps
    latencies: list[float] = []
    t_sign_end = None
    next_at = time.perf_counter()

    for i, vec in enumerate(frames):
        next_at += interval
        if (wait := next_at - time.perf_counter()) > 0:
            time.sleep(wait)
        ws.send(json.dumps({"type": "frame", "seq": i, "t_client_ms": time.time() * 1000,
                            "landmarks": [float(x) for x in vec],
                            "hands_present": [True, True]}))
        if i == sign_ends_at:
            t_sign_end = time.perf_counter()

        # Drain without blocking: most frames produce nothing.
        ws.settimeout(0.001)
        while True:
            try:
                msg = json.loads(ws.recv())
            except Exception:
                break
            if msg.get("type") == "gloss" and t_sign_end is not None:
                latencies.append((time.perf_counter() - t_sign_end) * 1000)
    return latencies


def main() -> int:
    import websocket  # noqa: PLC0415 — only needed for this measurement

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--url", default=f"ws://{settings.host}:{settings.port}/ws/stream")
    ap.add_argument("--fps", type=int, default=settings.target_fps)
    ap.add_argument("--out", type=Path, default=REPORTS / "latency.json")
    args = ap.parse_args()

    from ml.data.manifest import load
    from ml.dataset import features_for_clip

    # Round-robin across signs rather than taking the first N rows, which are all the
    # same gloss: segment length varies by sign and so does the wait for the exit run.
    by_gloss: dict[str, list] = {}
    for c in load():
        by_gloss.setdefault(c.gloss, []).append(c)
    clips = [c for row in zip(*by_gloss.values()) for c in row] if by_gloss else []
    if not clips:
        raise SystemExit("manifest is empty — nothing to replay")

    print(f"replaying up to {args.clips} clips at {args.fps} FPS against {args.url}")
    print("(the backend must already be running: uvicorn backend.main:app)\n")

    all_latencies: list[float] = []
    used = 0
    for clip in clips:
        if used >= args.clips:
            break
        seq = features_for_clip(clip.clip)
        if len(seq) < 10:
            continue
        try:
            ws = websocket.create_connection(args.url, timeout=10)
        except Exception as e:
            raise SystemExit(f"could not connect to {args.url}: {e}") from e
        try:
            ws.settimeout(5.0)
            ws.recv()  # the opening state event
            # A run of still frames after the clip lets the segmenter's exit condition
            # fire, which is where a real signer's pause-after-a-sign latency comes from.
            tail = np.repeat(seq[-1:], settings.exit_frames + settings.energy_smoothing_frames + 8, axis=0)
            got = replay_clip(ws, np.concatenate([seq, tail]), args.fps,
                              sign_ends_at=len(seq) - 1)
        finally:
            ws.close()
        if got:
            used += 1
            all_latencies.extend(got)
            print(f"  {clip.gloss:<14} {len(got)} segment(s)  "
                  f"{', '.join(f'{v:.0f} ms' for v in got)}", flush=True)

    if not all_latencies:
        raise SystemExit("no segments were produced — is the backend running this pack?")

    ordered = sorted(all_latencies)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    worst = ordered[-1]

    print(f"\n  {len(ordered)} segments from {used} clips")
    print(f"  p50 {p50:.0f} ms   p95 {p95:.0f} ms   max {worst:.0f} ms")
    verdict = "within" if p95 <= BUDGET_MS else "OVER"
    print(f"  {verdict} the {BUDGET_MS} ms budget (PRD M4)")

    REPORTS.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "segments": len(ordered), "clips": used, "fps": args.fps,
        "p50_ms": round(p50, 1), "p95_ms": round(p95, 1), "max_ms": round(worst, 1),
        "budget_ms": BUDGET_MS, "within_budget": p95 <= BUDGET_MS,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0 if p95 <= BUDGET_MS else 1


if __name__ == "__main__":
    raise SystemExit(main())
