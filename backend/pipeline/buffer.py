"""Inbound landmark ring buffer + throughput stats (PRD §3.2, §5.3).

`deque(maxlen=...)` *is* the "drop the oldest frame" policy — nothing to hand-roll.
The WebSocket read loop must never block on this.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    seq: int
    t_client_ms: float
    t_server_ms: float
    vector: np.ndarray
    hands_present: tuple[bool, bool] = (False, False)


@dataclass
class FrameBuffer:
    capacity: int
    frames: deque[Frame] = field(init=False)
    received: int = 0
    evicted: int = 0  # rolled out of the window — normal, not frame loss
    unconsumed_loss: int = 0  # evicted before the pipeline read them — real loss (§5.3)
    out_of_order: int = 0
    _last_seq: int = -1
    _consumed_seq: int = -1
    _window_start: float = field(default_factory=time.monotonic)
    _window_count: int = 0
    fps: float = 0.0

    def __post_init__(self) -> None:
        self.frames = deque(maxlen=self.capacity)

    def push(self, frame: Frame) -> bool:
        """Append, evicting the oldest if full.

        Returns True only when the evicted frame had NOT been consumed yet — that is
        real backpressure loss (§5.3) and is worth a warning. A full window that the
        pipeline is keeping up with rolls over silently, as it should.
        """
        self.received += 1
        self._window_count += 1
        if frame.seq <= self._last_seq:
            self.out_of_order += 1
        self._last_seq = max(self._last_seq, frame.seq)

        lost = False
        if len(self.frames) == self.capacity:
            self.evicted += 1
            lost = self.frames[0].seq > self._consumed_seq
            if lost:
                self.unconsumed_loss += 1
        self.frames.append(frame)

        elapsed = time.monotonic() - self._window_start
        if elapsed >= 1.0:
            self.fps = self._window_count / elapsed
            self._window_start, self._window_count = time.monotonic(), 0
        return lost

    def mark_consumed(self, seq: int) -> None:
        """The pipeline has processed everything up to `seq` (called from M4 onward)."""
        self._consumed_seq = max(self._consumed_seq, seq)

    def last(self, n: int) -> list[Frame]:
        return list(self.frames)[-n:]

    def matrix(self, n: int) -> np.ndarray:
        """Last n frames as an (n, D) array — oldest first."""
        return np.array([f.vector for f in self.last(n)])

    def clear(self) -> None:
        self.frames.clear()
        self._last_seq = -1

    def stats(self) -> dict:
        return {
            "received": self.received,
            "evicted": self.evicted,
            "lost": self.unconsumed_loss,
            "out_of_order": self.out_of_order,
            "buffered": len(self.frames),
            "fps": round(self.fps, 2),
        }


if __name__ == "__main__":
    def frame(i: int) -> Frame:
        return Frame(seq=i, t_client_ms=0, t_server_ms=0, vector=np.full(261, i))

    b = FrameBuffer(capacity=3)
    lost = [b.push(frame(i)) for i in range(5)]
    assert len(b.frames) == 3 and b.evicted == 2, b.stats()
    assert [f.seq for f in b.frames] == [2, 3, 4], "must drop the OLDEST, not the newest"
    assert lost == [False, False, False, True, True], "nothing consumed → every eviction is real loss"
    assert b.unconsumed_loss == 2
    assert b.matrix(2).shape == (2, 261)

    # a consumer keeping up: rollover is silent
    c = FrameBuffer(capacity=3)
    for i in range(6):
        assert c.push(frame(i)) is False, f"frame {i} flagged as lost while consumer kept up"
        c.mark_consumed(i)
    assert c.evicted == 3 and c.unconsumed_loss == 0, c.stats()

    b.push(frame(1))
    assert b.out_of_order == 1
    print("buffer.py self-check ok")
