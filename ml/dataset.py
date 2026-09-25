"""Manifest → training tensors, with the augmentation stack (PRD §4.4).

Pure NumPy, no TensorFlow: this is the part most likely to contain a silent bug, so it
stays independently testable. Feature extraction is slow (MediaPipe over ~1,250 clips),
so per-clip features are cached to .npy and only recomputed when the clip changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.config import ROOT, settings
from backend.vocab.schema import UNKNOWN, VocabPack, load_pack
from ml.data.manifest import Clip, load as load_manifest
from ml.features.extract import FEATURE_DIM, resample, with_velocity

CACHE = ROOT / "ml" / "data" / "cache"

# ---------------------------------------------------------------- mirroring

# BlazePose upper body: left/right landmark pairs that swap under a mirror.
POSE_MIRROR_PAIRS = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
    (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
)

# Face-lite pairs, as positions within FACE_LITE_IDX (not FaceMesh ids).
# FACE_LITE_IDX = (61,291,39,181,0,17,269,405,270,314,13,14, 70,63,105,66,300,293,334,296)
#                   0  1   2  3  4 5  6   7   8   9  10 11  12 13 14  15 16  17  18  19
# Mouth corners 61↔291, 39↔269, 181↔405; brows 70↔300, 63↔293, 105↔334, 66↔296.
# 0/17/13/14 are on the midline and map to themselves.
# NOTE: face-lite positions 8 and 9 (FaceMesh 270 and 314) have no mirror partner in
# Appendix B's selection — their true mirrors (40 and 84) were not included. They keep
# their index and only have x negated. 2 of 261 dims, documented rather than hidden.
FACE_MIRROR_PAIRS = ((0, 1), (2, 6), (3, 7), (12, 16), (13, 17), (14, 18), (15, 19))


def _build_mirror_index() -> np.ndarray:
    """Permutation over 261 dims implementing a left/right mirror."""
    idx = np.arange(FEATURE_DIM)

    def swap(a: int, b: int, base: int) -> None:
        for k in range(3):
            i, j = base + a * 3 + k, base + b * 3 + k
            idx[i], idx[j] = idx[j], idx[i]

    for a, b in POSE_MIRROR_PAIRS:
        swap(a, b, 0)
    # whole left-hand block swaps with the right-hand block
    for h in range(21):
        swap(h, h + 21, 75)
    for a, b in FACE_MIRROR_PAIRS:
        swap(a, b, 201)
    return idx


MIRROR_INDEX = _build_mirror_index()
# x is every third value; a mirror reflects it about the (already centred) midline
X_SIGN = np.where(np.arange(FEATURE_DIM) % 3 == 0, -1.0, 1.0)


def mirror(seq: np.ndarray) -> np.ndarray:
    """Left↔right mirror of a (T,261) sequence."""
    return seq[:, MIRROR_INDEX] * X_SIGN


# ---------------------------------------------------------------- features

def features_for_clip(clip: str, use_cache: bool = True) -> np.ndarray:
    """(T,261) for one clip, cached on disk keyed by path and mtime.

    A `.npy` path is features already — segments confirmed live in the speaker app are
    stored that way. There is no video to extract, and treating them as ordinary clips
    means the manifest, the splits, cross-validation and training all handle them with
    no special case anywhere.
    """
    path = ROOT / clip if not Path(clip).is_absolute() else Path(clip)
    if path.suffix == ".npy":
        return np.load(path)
    key = f"{Path(clip).stem}_{int(path.stat().st_mtime)}.npy"
    cached = CACHE / key
    if use_cache and cached.exists():
        return np.load(cached)

    from ml.features.extract import extract_video  # heavy import, only when needed

    seq = extract_video(str(path))
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cached, seq)
    return seq


def load_split(
    split: str,
    pack: VocabPack | None = None,
    clips: list[Clip] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """All clips in one split → X (N,45,D), y (N,), and the signer of each sample."""
    pack = pack or load_pack(settings.vocab_pack)
    clips = [c for c in (clips or load_manifest()) if c.split == split]
    labels = {g: i for i, g in enumerate(pack.labels)}

    xs, ys, signers = [], [], []
    for c in clips:
        seq = features_for_clip(c.clip)
        if len(seq) == 0:
            continue
        xs.append(resample(seq, settings.segment_resample_frames))
        ys.append(labels.get(c.gloss, labels[UNKNOWN]))
        signers.append(c.signer)
    if not xs:
        return np.zeros((0, settings.segment_resample_frames, FEATURE_DIM)), np.zeros(0, int), []
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), signers


# ---------------------------------------------------------------- augmentation

def time_warp(seq: np.ndarray, rng: np.random.Generator, max_pct: float = 0.15) -> np.ndarray:
    """Random temporal crop/stretch of ±15%, resampled back to the fixed length."""
    n = len(seq)
    keep = int(round(n * (1.0 - rng.uniform(0, max_pct))))
    start = rng.integers(0, max(1, n - keep + 1))
    return resample(seq[start : start + keep], n)


def frame_dropout(seq: np.ndarray, rng: np.random.Generator, p: float = 0.10) -> np.ndarray:
    """Zero out ~10% of frames, mimicking detection dropouts on real video."""
    out = seq.copy()
    out[rng.random(len(seq)) < p] = 0.0
    return out


def jitter(seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.01) -> np.ndarray:
    """Gaussian landmark noise. Invalid (all-zero) frames stay exactly zero — the
    classifier learns 'no detection' from them, and noise would blur that signal."""
    out = seq + rng.normal(0.0, sigma, seq.shape).astype(seq.dtype)
    valid = np.any(seq != 0, axis=1)
    out[~valid] = 0.0
    return out


def rotate(seq: np.ndarray, radians: float) -> np.ndarray:
    """Rotate every landmark in the x-y plane about the shoulder midpoint.

    Normalisation makes the features invariant to how far away and how far to the side
    the signer is, but NOT to camera angle — the spec calls that out as a limitation.
    A signer leaning, or a laptop lid tilted back, rotates the whole skeleton and the
    model has never seen that. Coordinates are already centred on the shoulder midpoint,
    so a plain rotation about the origin is the right transform.

    z is left alone: it is a depth estimate in a different, noisier frame of reference,
    and rotating it against x/y would fabricate geometry rather than simulate a tilt.
    """
    c, s = np.cos(radians), np.sin(radians)
    out = seq.copy()
    x = out[:, 0::3].copy()
    y = out[:, 1::3].copy()
    out[:, 0::3] = c * x - s * y
    out[:, 1::3] = s * x + c * y
    # an all-zero frame means "nothing detected" and must stay exactly zero
    out[~np.any(seq != 0, axis=1)] = 0.0
    return out


def augment(
    seq: np.ndarray,
    gloss: str,
    pack: VocabPack,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    """The PRD §4.4 stack. Mirroring is skipped for signs whose meaning depends on
    handedness — the vocab pack's `mirror_safe` flag decides, not this code."""
    mirror_safe = {e.gloss: e.mirror_safe for e in pack.entries}
    seq = time_warp(seq, rng, max_pct=strength * 0.15)
    if mirror_safe.get(gloss, False) and rng.random() < 0.5:
        seq = mirror(seq)
    if strength > 1.0 or rng.random() < 0.5:
        seq = rotate(seq, np.deg2rad(rng.uniform(-12, 12) * strength))
    seq = jitter(seq, rng, sigma=0.01 * strength)
    seq = frame_dropout(seq, rng, p=0.10 * strength)
    return seq.astype(np.float32)


def augmented_batches(
    x: np.ndarray,
    y: np.ndarray,
    pack: VocabPack,
    batch_size: int,
    seed: int = 0,
    strength: float = 1.0,
):
    """Infinite shuffled generator of augmented batches, for model.fit()."""
    rng = np.random.default_rng(seed)
    labels = pack.labels
    n = len(x)
    while True:
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            pick = order[i : i + batch_size]
            batch = np.stack([augment(x[j], labels[y[j]], pack, rng, strength) for j in pick])
            if settings.use_velocity_features:
                batch = np.stack([with_velocity(b) for b in batch]).astype(np.float32)
            yield batch, y[pick]


def apply_velocity(x: np.ndarray) -> np.ndarray:
    """Match the training-time feature width for val/test (no augmentation)."""
    if not settings.use_velocity_features:
        return x
    return np.stack([with_velocity(s) for s in x]).astype(np.float32)
