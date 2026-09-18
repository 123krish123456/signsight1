"""Motion-energy state machine — finds where one sign ends and the next begins (PRD §4.3).

No learned component: pure thresholds on wrist velocity, so this is buildable and
testable before any training data exists. Every constant comes from `config.py`, because
the thresholds need tuning in the room on demo day without a code change (PRD §8).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from backend.config import settings
from ml.features.extract import L_WRIST, R_WRIST, resample


class State(str, Enum):
    IDLE = "IDLE"
    SIGNING = "SIGNING"


@dataclass
class Segment:
    """One candidate sign, ready for the classifier."""

    frames: np.ndarray  # (45, D) after resampling
    raw_length: int  # frames actually captured, before resampling
    truncated: bool  # hit MAX_SEGMENT_FRAMES rather than settling
    start_seq: int
    end_seq: int

    @property
    def duration_ms(self) -> float:
        return self.raw_length / settings.target_fps * 1000


def _wrists(vec: np.ndarray) -> np.ndarray:
    """The two wrist points as a (2,3) array. Pose lives in the first 75 dims."""
    return np.array([vec[L_WRIST * 3 : L_WRIST * 3 + 3], vec[R_WRIST * 3 : R_WRIST * 3 + 3]])


def _is_valid(vec: np.ndarray) -> bool:
    """A frame with no usable pose normalises to all zeros (PRD §4.2 step 3)."""
    return bool(np.any(vec[:75]))


@dataclass
class Segmenter:
    """Feed frames in order; get a Segment back when one completes.

    Hysteresis is the whole point: entering needs energy above ENTER_THRESH, leaving
    needs it below the *lower* EXIT_THRESH. A single threshold oscillates at the
    boundary and emits dozens of garbage segments a second.
    """

    state: State = State.IDLE
    energy: float = 0.0
    _smoothing: deque[float] = field(init=False)
    _prev_wrists: np.ndarray | None = None
    _above: int = 0  # consecutive frames over ENTER_THRESH
    _below: int = 0  # consecutive frames under EXIT_THRESH
    _buffer: list[np.ndarray] = field(default_factory=list)
    _preroll: deque[np.ndarray] = field(init=False)
    _start_seq: int = 0
    _last_seq: int = 0

    def __post_init__(self) -> None:
        self._smoothing = deque(maxlen=settings.energy_smoothing_frames)
        # The frames that triggered entry are part of the sign, so keep them.
        self._preroll = deque(maxlen=settings.enter_frames)

    # ---------- energy ----------

    def _motion_energy(self, vec: np.ndarray) -> float:
        """Mean L2 wrist velocity, smoothed over a 5-frame window.

        An invalid frame contributes no motion rather than a spike: the jump between a
        real position and the all-zero vector is an artefact of detection dropping out,
        not the signer moving, and it would otherwise trigger a false segment.
        """
        if not _is_valid(vec):
            self._prev_wrists = None
            raw = 0.0
        else:
            current = _wrists(vec)
            raw = 0.0 if self._prev_wrists is None else float(
                np.linalg.norm(current - self._prev_wrists, axis=1).mean()
            )
            self._prev_wrists = current
        self._smoothing.append(raw)
        return float(np.mean(self._smoothing))

    # ---------- state machine ----------

    def push(self, vec: np.ndarray, seq: int = 0) -> Segment | None:
        """Feed one normalised frame. Returns a Segment on the frame that completes one."""
        self._last_seq = seq
        self.energy = self._motion_energy(vec)

        if self.state is State.IDLE:
            self._preroll.append(vec)
            self._above = self._above + 1 if self.energy > settings.enter_thresh else 0
            if self._above >= settings.enter_frames:
                self.state = State.SIGNING
                self._buffer = list(self._preroll)  # the lead-in belongs to the sign
                self._preroll.clear()
                self._above = self._below = 0
                self._start_seq = seq - len(self._buffer) + 1
            return None

        # SIGNING
        self._buffer.append(vec)
        if len(self._buffer) >= settings.max_segment_frames:
            return self._emit(truncated=True)

        self._below = self._below + 1 if self.energy < settings.exit_thresh else 0
        if self._below >= settings.exit_frames:
            return self._emit(truncated=False)
        return None

    def _emit(self, truncated: bool) -> Segment | None:
        """Close the current segment. Too-short ones are noise and are dropped.

        The length test measures the *moving* part, excluding the trailing settle run.
        Those settle frames are kept for classification — a sign ending in a held
        handshape carries meaning in the hold — but counting them toward the length
        would make MIN_SEGMENT_FRAMES unreachable: the 5-frame smoothing window plus a
        4-frame exit run pads every segment past 8 frames, so a nose-scratch would
        survive the noise guard.
        """
        settle = self._below
        frames, self._buffer = self._buffer, []
        self.state = State.IDLE
        self._above = self._below = 0
        self._preroll.clear()

        if len(frames) - settle < settings.min_segment_frames:
            return None
        return Segment(
            frames=resample(np.array(frames), settings.segment_resample_frames),
            raw_length=len(frames),
            truncated=truncated,
            start_seq=self._start_seq,
            end_seq=self._last_seq,
        )

    def reset(self) -> None:
        self.state = State.IDLE
        self.energy = 0.0
        self._smoothing.clear()
        self._preroll.clear()
        self._buffer = []
        self._prev_wrists = None
        self._above = self._below = 0
