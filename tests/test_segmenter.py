"""Synthetic motion traces → expected segment boundaries (PRD §9).

The flicker case is tested explicitly: the PRD calls single-threshold oscillation
"the most common bug in student implementations of this pipeline".
"""

import numpy as np
import pytest

from backend.config import settings
from backend.pipeline.segmenter import Segmenter, State
from ml.features.extract import FEATURE_DIM, L_WRIST, R_WRIST


def frame(wrist_offset: float, valid: bool = True) -> np.ndarray:
    """A frame whose wrists sit at a given position. Successive offsets create velocity."""
    v = np.zeros(FEATURE_DIM)
    if not valid:
        return v  # all-zero == no pose detected
    v[:75] = 0.01  # non-zero pose block marks the frame valid
    for idx in (L_WRIST, R_WRIST):
        v[idx * 3] = wrist_offset
    return v


def trace(seg, velocities, valid=True):
    """Drive the segmenter along a velocity profile; collect emitted segments."""
    out, pos = [], 0.0
    for i, vel in enumerate(velocities):
        pos += vel
        s = seg.push(frame(pos, valid=valid), seq=i)
        if s:
            out.append(s)
    return out


HIGH = 0.30  # well above enter_thresh (0.08) once smoothed
LOW = 0.0


def test_still_signer_never_starts_a_segment():
    seg = Segmenter()
    assert trace(seg, [LOW] * 60) == []
    assert seg.state is State.IDLE


def test_one_clear_sign_emits_one_segment():
    seg = Segmenter()
    segments = trace(seg, [LOW] * 5 + [HIGH] * 20 + [LOW] * 12)
    assert len(segments) == 1
    s = segments[0]
    assert not s.truncated
    assert s.frames.shape == (settings.segment_resample_frames, FEATURE_DIM)
    assert s.raw_length >= settings.min_segment_frames
    assert seg.state is State.IDLE


def test_two_signs_separated_by_a_pause_emit_two_segments():
    seg = Segmenter()
    pattern = [HIGH] * 20 + [LOW] * 12
    assert len(trace(seg, [LOW] * 5 + pattern + pattern)) == 2


def test_hysteresis_prevents_flicker_at_the_boundary():
    """Energy parked between the two thresholds must not oscillate IDLE/SIGNING.

    With a single threshold this trace emits a burst of garbage segments.
    """
    between = (settings.enter_thresh + settings.exit_thresh) / 2  # 0.06
    seg = Segmenter()
    # Start the sign, then hover in the dead band. Total stays under
    # MAX_SEGMENT_FRAMES so that truncation cannot be mistaken for flicker.
    segments = trace(seg, [HIGH] * 8 + [between] * 25)
    assert segments == [], "hovering between thresholds must not close a segment"
    assert seg.state is State.SIGNING, "and must not drop back to IDLE"


def test_dead_band_does_not_start_a_segment_from_idle():
    between = (settings.enter_thresh + settings.exit_thresh) / 2
    seg = Segmenter()
    assert trace(seg, [between] * 40) == []
    assert seg.state is State.IDLE


def test_brief_twitch_is_discarded_as_noise():
    """Scratching, not signing: 4 frames of motion is under MIN_SEGMENT_FRAMES.

    Guards a real bug — the smoothing window and exit run pad such a twitch out to
    11 buffered frames, so a naive length check on the buffer lets it through.
    """
    seg = Segmenter()
    assert trace(seg, [LOW] * 4 + [HIGH] * 4 + [LOW] * 10) == []


def test_endless_motion_truncates_at_the_cap():
    seg = Segmenter()
    segments = trace(seg, [HIGH] * (settings.max_segment_frames * 2))
    assert segments, "a sign that never settles must still be emitted"
    assert segments[0].truncated
    assert segments[0].raw_length <= settings.max_segment_frames


def test_dropped_detection_does_not_fake_motion():
    """An all-zero frame means MediaPipe lost the pose. The jump to and from the
    origin is a detection artefact, not the signer moving, and must not start a sign."""
    seg = Segmenter()
    far = frame(5.0)
    for i in range(10):
        assert seg.push(far, seq=i) is None
    for i in range(10, 20):
        assert seg.push(frame(0, valid=False), seq=i) is None
    assert seg.state is State.IDLE, "detection dropout must not be read as motion"


def test_segment_is_always_resampled_to_the_configured_length():
    for hold in (10, 20, 40):
        seg = Segmenter()
        segments = trace(seg, [LOW] * 3 + [HIGH] * hold + [LOW] * 12)
        assert len(segments) == 1
        assert segments[0].frames.shape[0] == settings.segment_resample_frames


def test_reset_clears_in_progress_state():
    seg = Segmenter()
    trace(seg, [HIGH] * 12)
    assert seg.state is State.SIGNING
    seg.reset()
    assert seg.state is State.IDLE
    assert trace(seg, [LOW] * 20) == []


@pytest.mark.parametrize("enter,exit_", [(0.02, 0.01), (0.20, 0.10)])
def test_thresholds_are_tunable_without_code_changes(monkeypatch, enter, exit_):
    """PRD §8: demo-day tuning must not require touching code."""
    monkeypatch.setattr(settings, "enter_thresh", enter)
    monkeypatch.setattr(settings, "exit_thresh", exit_)
    seg = Segmenter()
    segments = trace(seg, [LOW] * 3 + [HIGH] * 20 + [LOW] * 12)
    assert len(segments) == 1
