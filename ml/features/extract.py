"""Video / landmark frames → 261-d feature vectors (PRD §4.2).

BINDING SPEC. `app/src/normalise.ts` must produce bit-identical output for the same
input; `tests/test_parity.py` enforces max abs diff < 1e-6. Change one, change both.
"""

from __future__ import annotations

import numpy as np

POSE_N = 25  # upper body, pose indices 0..24
HAND_N = 21
FACE_N = 20

POSE_DIM = POSE_N * 3  # 75
HAND_DIM = HAND_N * 3  # 63
FACE_DIM = FACE_N * 3  # 60
FEATURE_DIM = POSE_DIM + HAND_DIM * 2 + FACE_DIM  # 261

L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16  # used by the segmenter's motion energy
MIN_SHOULDER_WIDTH = 1e-6

# Appendix B — MediaPipe FaceMesh (468-point topology).
FACE_LITE_IDX = (
    # outer lip contour (12)
    61, 291, 39, 181, 0, 17, 269, 405, 270, 314, 13, 14,
    # eyebrows (8)
    70, 63, 105, 66, 300, 293, 334, 296,
)
assert len(FACE_LITE_IDX) == FACE_N

Landmarks = list[list[float]] | np.ndarray | None


def _as_array(lm: Landmarks, n: int) -> np.ndarray | None:
    """None / wrong-length / empty → None (absent). Otherwise an (n,3) float64 array."""
    if lm is None:
        return None
    arr = np.asarray(lm, dtype=np.float64)
    if arr.shape != (n, 3):
        return None
    return arr


def _select_face(face: Landmarks) -> np.ndarray | None:
    """Face mesh → the 20 face-lite points.

    Accepts the 468-point mesh, the 478-point mesh (MediaPipe Tasks appends 10 iris
    points; indices 0-467 are unchanged), or an already-selected 20-point block.
    """
    if face is None:
        return None
    arr = np.asarray(face, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    if arr.shape[0] >= 468:
        return arr[list(FACE_LITE_IDX)]
    if arr.shape[0] == FACE_N:
        return arr
    return None


def normalise_frame(
    pose: Landmarks,
    left_hand: Landmarks,
    right_hand: Landmarks,
    face: Landmarks,
) -> np.ndarray:
    """One frame of raw MediaPipe landmarks → 261 floats.

    `pose` is 33 (full BlazePose) or 25 (already trimmed) landmarks; `face` is the full
    468-point mesh or the 20 pre-selected face-lite points. Any absent block is zeros —
    never interpolated, because absence is informative (one-handed signs exist).
    Frame is invalid (all zeros) when the shoulders are missing or degenerate.
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float64)

    p = _as_array(pose, 33)
    if p is None:
        p = _as_array(pose, POSE_N)
    if p is None:
        return out  # no pose → no anchor → invalid frame

    mid = (p[L_SHOULDER] + p[R_SHOULDER]) / 2.0
    # Shoulder width over x,y only: MediaPipe's pose z is a noisy per-frame depth
    # estimate and dividing every feature by it injects that noise everywhere.
    # Deliberate, documented (PROJECT_RULES.md); normalise.ts does the same.
    delta = p[L_SHOULDER][:2] - p[R_SHOULDER][:2]
    scale = float(np.hypot(delta[0], delta[1]))
    if scale < MIN_SHOULDER_WIDTH:
        return out

    def norm(block: np.ndarray) -> np.ndarray:
        return ((block - mid) / scale).ravel()

    out[:POSE_DIM] = norm(p[:POSE_N])

    lh = _as_array(left_hand, HAND_N)
    if lh is not None:
        out[POSE_DIM : POSE_DIM + HAND_DIM] = norm(lh)

    rh = _as_array(right_hand, HAND_N)
    if rh is not None:
        out[POSE_DIM + HAND_DIM : POSE_DIM + 2 * HAND_DIM] = norm(rh)

    f = _select_face(face)
    if f is not None:
        out[POSE_DIM + 2 * HAND_DIM :] = norm(f)

    return out


def with_velocity(seq: np.ndarray) -> np.ndarray:
    """(T,261) → (T,522), appending frame-to-frame deltas. PRD §4.2 optional block."""
    deltas = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, deltas], axis=1)


def resample(seq: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly resample a (T,D) sequence to exactly (n_frames,D). PRD §4.3."""
    t = len(seq)
    if t == 0:
        return np.zeros((n_frames, FEATURE_DIM))
    if t == n_frames:
        return seq
    src = np.linspace(0.0, t - 1, num=n_frames)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None]
    return seq[lo] * (1 - w) + seq[hi] * w


# The same model file the browser loads, so training and inference see identical
# landmarks. Vendored by `npm run fetch-assets`; override with SIGNSIGHT_HOLISTIC_MODEL.
def _model_path() -> str:
    import os
    from pathlib import Path

    override = os.environ.get("SIGNSIGHT_HOLISTIC_MODEL")
    if override:
        return override
    root = Path(__file__).resolve().parent.parent.parent
    for candidate in (
        root / "app" / "public" / "models" / "holistic_landmarker.task",
        root / "extension" / "vendor" / "models" / "holistic_landmarker.task",
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "holistic_landmarker.task not found — run `npm run fetch-assets` in app/, "
        "or set SIGNSIGHT_HOLISTIC_MODEL"
    )


def _landmarker(running_mode):
    """MediaPipe Tasks holistic landmarker. The legacy `mp.solutions` API this used to
    call was removed in mediapipe 1.0; Tasks is also what the clients run, so both
    sides now share one model file."""
    import mediapipe as mp  # noqa: PLC0415
    from mediapipe.tasks.python import BaseOptions  # noqa: PLC0415
    from mediapipe.tasks.python import vision  # noqa: PLC0415

    return vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model_path()),
            running_mode=running_mode,
        )
    ), mp


def extract_video(path: str, every_nth: int = 2) -> np.ndarray:
    """Video file → (T,261). Needs the `ml` extra (mediapipe, opencv).

    `every_nth=2` decimates 30 FPS source clips to the 15 FPS the clients run at
    (PRD §4.1) so training features match inference features. Pass 1 for footage that
    is already at or below 15 FPS, or you throw away half of a short clip.
    """
    import cv2  # noqa: PLC0415 — heavy, import only when actually extracting
    from mediapipe.tasks.python import vision  # noqa: PLC0415

    landmarker, mp = _landmarker(vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_nth == 0:
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                res = landmarker.detect_for_video(image, int(i / fps * 1000))
                frames.append(_from_result(res))
            i += 1
    finally:
        cap.release()
        landmarker.close()
    return np.array(frames, dtype=np.float64) if frames else np.zeros((0, FEATURE_DIM))


def _from_result(res) -> np.ndarray:
    """HolisticLandmarkerResult → one normalised 261-d frame."""
    return normalise_frame(
        _mp_list(getattr(res, "pose_landmarks", None)),
        _mp_list(getattr(res, "left_hand_landmarks", None)),
        _mp_list(getattr(res, "right_hand_landmarks", None)),
        _mp_list(getattr(res, "face_landmarks", None)),
    )


def extract_image(path: str, n_frames: int = 45) -> np.ndarray:
    """Still image → (n_frames, 261), the same landmarks repeated.

    Fingerspelling letters are held handshapes rather than movements, so a still is a
    fair approximation of one and this is how the public alphabet datasets can be used
    at all. Be honest about the cost: every velocity feature is exactly zero, so the
    model cannot learn anything about how a letter is approached or released, and a
    letter that does move (J and Z in most alphabets) is represented wrongly.
    """
    import cv2  # noqa: PLC0415

    image = cv2.imread(str(path))
    if image is None:
        return np.zeros((0, FEATURE_DIM))

    from mediapipe.tasks.python import vision  # noqa: PLC0415

    landmarker, mp = _landmarker(vision.RunningMode.IMAGE)
    try:
        res = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        )
    finally:
        landmarker.close()

    frame = _from_result(res)
    if not frame.any():
        return np.zeros((0, FEATURE_DIM))  # nothing detected — caller drops it
    return np.tile(frame, (n_frames, 1))


def _mp_list(landmarks) -> list[list[float]] | None:
    """MediaPipe landmarks → plain [[x,y,z], ...]. Tolerates the Tasks API's flat list,
    a per-person nested list, and the legacy `.landmark` container."""
    if landmarks is None:
        return None
    if hasattr(landmarks, "landmark"):  # legacy solutions API
        landmarks = landmarks.landmark
    if len(landmarks) == 0:
        return None
    if isinstance(landmarks[0], (list, tuple)):  # nested per person
        landmarks = landmarks[0]
        if len(landmarks) == 0:
            return None
    return [[lm.x, lm.y, lm.z] for lm in landmarks]


if __name__ == "__main__":
    # Self-check: shape, invariance, absence handling.
    rng = np.random.default_rng(0)
    pose = rng.random((33, 3))
    pose[L_SHOULDER], pose[R_SHOULDER] = [0.6, 0.5, 0.0], [0.4, 0.5, 0.0]
    hand = rng.random((21, 3))
    face = rng.random((468, 3))

    v = normalise_frame(pose, hand, None, face)
    assert v.shape == (FEATURE_DIM,), v.shape
    assert np.allclose(v[POSE_DIM + HAND_DIM : POSE_DIM + 2 * HAND_DIM], 0), "absent hand must be zeros"
    assert not np.allclose(v[POSE_DIM : POSE_DIM + HAND_DIM], 0)

    # translate + scale the whole frame: normalised output must not move
    v2 = normalise_frame(pose * 3.0 + 0.7, hand * 3.0 + 0.7, None, face * 3.0 + 0.7)
    assert np.abs(v - v2).max() < 1e-9, np.abs(v - v2).max()

    degenerate = pose.copy()
    degenerate[L_SHOULDER] = degenerate[R_SHOULDER]
    assert np.allclose(normalise_frame(degenerate, hand, hand, face), 0), "degenerate shoulders → zero vector"

    assert resample(np.arange(20 * FEATURE_DIM).reshape(20, FEATURE_DIM), 45).shape == (45, FEATURE_DIM)
    assert with_velocity(np.zeros((5, FEATURE_DIM))).shape == (5, 522)
    print("extract.py self-check ok")
