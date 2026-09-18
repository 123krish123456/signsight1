"""JS ↔ Python normalisation parity (PRD §9, M1 DoD).

Silent drift between `normalise.ts` and `extract.py` destroys live accuracy while every
other test stays green. This is the test that catches it.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ml.features.extract import FEATURE_DIM, normalise_frame

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "parity_frames.json"
RUNNER = ROOT / "tests" / "parity_runner.ts"
TOLERANCE = 1e-6


def _round(a: np.ndarray) -> list[list[float]]:
    return np.round(a, 6).tolist()


def make_fixture() -> list[dict]:
    """Deterministic frames covering every branch: full frame, missing hands,
    missing face, degenerate shoulders, no pose at all."""
    rng = np.random.default_rng(20240817)

    def pose() -> np.ndarray:
        p = rng.random((33, 3))
        p[11], p[12] = [0.62, 0.51, 0.03], [0.38, 0.49, -0.02]
        return p

    frames = [
        # both hands + full 468-point face
        {"pose": _round(pose()), "left_hand": _round(rng.random((21, 3))),
         "right_hand": _round(rng.random((21, 3))), "face": _round(rng.random((468, 3)))},
        # one-handed sign, no face detected
        {"pose": _round(pose()), "left_hand": None,
         "right_hand": _round(rng.random((21, 3))), "face": None},
        # pre-trimmed 25-pose + pre-selected 20-point face
        {"pose": _round(pose()[:25]), "left_hand": _round(rng.random((21, 3))),
         "right_hand": None, "face": _round(rng.random((20, 3)))},
        # negative / large coordinates — z is signed in MediaPipe
        {"pose": _round(pose() * 4 - 2), "left_hand": _round(rng.random((21, 3)) * 4 - 2),
         "right_hand": _round(rng.random((21, 3)) * 4 - 2), "face": _round(rng.random((468, 3)) * 4 - 2)},
        # 478-point face mesh (MediaPipe Tasks, iris points appended)
        {"pose": _round(pose()), "left_hand": _round(rng.random((21, 3))),
         "right_hand": _round(rng.random((21, 3))), "face": _round(rng.random((478, 3)))},
        # degenerate shoulders → invalid frame, both sides must emit zeros
        {"pose": _round(np.zeros((33, 3))), "left_hand": _round(rng.random((21, 3))),
         "right_hand": None, "face": None},
        # no pose → invalid frame
        {"pose": None, "left_hand": _round(rng.random((21, 3))),
         "right_hand": None, "face": None},
    ]
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(frames))
    return frames


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    return make_fixture()


def test_python_reference(frames):
    vectors = [normalise_frame(f["pose"], f["left_hand"], f["right_hand"], f["face"]) for f in frames]
    assert all(v.shape == (FEATURE_DIM,) for v in vectors)
    assert np.allclose(vectors[5], 0), "degenerate shoulders must zero the frame"
    assert np.allclose(vectors[6], 0), "missing pose must zero the frame"
    assert not np.allclose(vectors[0], 0)
    assert not np.allclose(vectors[4][201:], 0), "478-point mesh must still fill face-lite"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python(frames):
    proc = subprocess.run(
        ["node", str(RUNNER), str(FIXTURE)], capture_output=True, text=True, cwd=ROOT
    )
    if proc.returncode != 0:
        pytest.fail(f"parity_runner.ts failed (needs node >= 22.18):\n{proc.stderr}")

    js = np.array(json.loads(proc.stdout), dtype=np.float64)
    py = np.array(
        [normalise_frame(f["pose"], f["left_hand"], f["right_hand"], f["face"]) for f in frames]
    )

    assert js.shape == py.shape == (len(frames), FEATURE_DIM)
    drift = np.abs(js - py).max()
    assert drift < TOLERANCE, f"JS/Python normalisation drift {drift:.3e} at {np.unravel_index(np.abs(js - py).argmax(), js.shape)}"
