"""Augmentation stack (PRD §4.4). Bugs here are silent: the model just trains worse."""

import numpy as np
import pytest

from backend.config import ROOT, settings
from backend.vocab.schema import load_pack
from ml.dataset import (
    MIRROR_INDEX,
    augment,
    frame_dropout,
    jitter,
    mirror,
    time_warp,
)
from ml.features.extract import FEATURE_DIM, HAND_DIM, POSE_DIM

PACK = load_pack(ROOT / "backend" / "vocab" / "isl_v1.json")
L_HAND = slice(POSE_DIM, POSE_DIM + HAND_DIM)
R_HAND = slice(POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM)


def seq(n=45, seed=0):
    return np.random.default_rng(seed).random((n, FEATURE_DIM)).astype(np.float32)


def test_mirror_is_its_own_inverse():
    s = seq()
    assert np.allclose(mirror(mirror(s)), s, atol=1e-6)


def test_mirror_is_a_permutation_of_every_dimension():
    assert sorted(MIRROR_INDEX.tolist()) == list(range(FEATURE_DIM))


def test_mirror_swaps_the_hand_blocks():
    s = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    s[0, L_HAND] = 1.0  # only the left hand is present
    m = mirror(s)
    assert np.allclose(np.abs(m[0, R_HAND]), 1.0), "left hand must land in the right block"
    assert np.allclose(m[0, L_HAND], 0.0), "and vacate the left block"


def test_mirror_negates_x_only():
    s = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    s[0, 0:3] = [0.5, 0.25, 0.75]  # nose: on the midline, maps to itself
    m = mirror(s)
    assert m[0, 0] == pytest.approx(-0.5), "x reflects"
    assert m[0, 1] == pytest.approx(0.25), "y unchanged"
    assert m[0, 2] == pytest.approx(0.75), "z unchanged"


def test_mirror_swaps_left_and_right_shoulders():
    s = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    s[0, 11 * 3 : 11 * 3 + 3] = [1.0, 2.0, 3.0]  # left shoulder
    m = mirror(s)
    assert np.allclose(m[0, 12 * 3 : 12 * 3 + 3], [-1.0, 2.0, 3.0]), "must become the right shoulder"
    assert np.allclose(m[0, 11 * 3 : 11 * 3 + 3], 0.0)


def test_handedness_sensitive_signs_are_never_mirrored():
    """A sign flagged mirror_safe=false must come back unmirrored every time."""
    unsafe = next(e.gloss for e in PACK.entries if not e.mirror_safe)
    s = seq()
    rng = np.random.default_rng(1)
    for _ in range(25):
        out = augment(s, unsafe, PACK, rng)
        # mirroring would move the left-hand block into the right; nothing else does
        assert not np.allclose(out[:, R_HAND], s[:, L_HAND], atol=0.2)


def test_time_warp_preserves_length():
    s = seq()
    for i in range(10):
        assert time_warp(s, np.random.default_rng(i)).shape == s.shape


def test_frame_dropout_zeroes_whole_frames_not_single_values():
    s = seq(seed=3) + 1.0  # keep every value away from zero
    out = frame_dropout(s, np.random.default_rng(0), p=0.5)
    for row in out:
        assert np.all(row == 0) or np.all(row != 0), "dropout must be per frame"


def test_jitter_leaves_undetected_frames_at_exactly_zero():
    """An all-zero frame means 'nothing detected'. Noise there teaches a false signal."""
    s = seq()
    s[3] = 0.0
    out = jitter(s, np.random.default_rng(0))
    assert np.all(out[3] == 0.0)
    assert not np.allclose(out[0], s[0]), "real frames should still be perturbed"


def test_augment_keeps_shape_and_dtype():
    s = seq()
    out = augment(s, "ME", PACK, np.random.default_rng(0))
    assert out.shape == (settings.segment_resample_frames, FEATURE_DIM)
    assert out.dtype == np.float32


def test_augment_actually_changes_the_data():
    s = seq()
    out = augment(s, "ME", PACK, np.random.default_rng(0))
    assert not np.allclose(out, s), "augmentation that changes nothing trains nothing"
