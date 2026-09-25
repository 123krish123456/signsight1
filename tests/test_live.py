"""The live path end to end: frames in over the WebSocket, glosses and sentences out.

This is the test that would have caught M4 being half-finished. The segmenter had its own
tests and the classifier had its own accuracy numbers, and between them sat a branch that
returned the string "UNKNOWN" unconditionally, which nothing exercised.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from ml.features.extract import FEATURE_DIM, L_WRIST, R_WRIST


def frame(offset: float) -> list[float]:
    """A valid frame with both wrists at a given x. Successive offsets make motion."""
    v = np.zeros(FEATURE_DIM)
    v[:75] = 0.01  # a non-zero pose block is what marks the frame valid
    for idx in (L_WRIST, R_WRIST):
        v[idx * 3] = offset
    return v.tolist()


def replay(ws, vectors: list[list[float]], start_seq: int = 0) -> list[dict]:
    """Send every frame, then drain everything the server sent back.

    Two things make this less obvious than it looks. The server only speaks on a state
    change or a finished segment, so reading after each frame deadlocks on the majority
    that produce nothing. And the socket closing does not raise on this client, so a
    drain needs its own terminator.

    Hence the deliberately malformed final message: the server answers it with a
    BAD_MESSAGE error, and because it handles messages strictly in order, that error can
    only arrive after everything the frames produced. No production code exists for the
    benefit of this test.
    """
    for i, vec in enumerate(vectors):
        ws.send_json({"type": "frame", "seq": start_seq + i, "t_client_ms": 0.0,
                      "landmarks": vec, "hands_present": [True, True]})
    ws.send_json({"type": "end-of-replay"})

    events = []
    while (msg := ws.receive_json()).get("code") != "BAD_MESSAGE":
        events.append(msg)
    return events


def motion(n: int, step: float, start: float = 0.0) -> list[list[float]]:
    return [frame(start + i * step) for i in range(n)]


def still(n: int, at: float = 0.0) -> list[list[float]]:
    return [frame(at) for _ in range(n)]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_rejects_wrong_dimension(client):
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json() == {"type": "state", "value": "IDLE"}
        ws.send_json({"type": "frame", "seq": 0, "landmarks": [0.0] * 10})
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "BAD_DIMENSION"
        assert str(settings.expected_dim) in err["message"]


def test_a_moving_signer_produces_a_gloss(client):
    """Motion then stillness must yield exactly one gloss event for the segment.

    The value is not asserted: on synthetic wrist motion the model has every right to
    say UNKNOWN. What matters is that a segment reaches the classifier at all and that
    the event carries the fields the clients read.
    """
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["value"] == "IDLE"
        events = replay(ws, still(5) + motion(20, 0.3) + still(20, at=20 * 0.3))

    states = [e["value"] for e in events if e["type"] == "state"]
    glosses = [e for e in events if e["type"] == "gloss"]

    assert "SIGNING" in states, f"never entered SIGNING: {states}"
    assert len(glosses) == 1, f"expected one segment, got {len(glosses)}"
    g = glosses[0]
    assert set(g) >= {"value", "confidence", "segment_ms"}
    assert 0.0 <= g["confidence"] <= 1.0
    assert g["segment_ms"] > 0


def test_reset_clears_state(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        replay(ws, still(5) + motion(20, 0.3))
        ws.send_json({"type": "control", "action": "reset_buffer"})
        assert ws.receive_json() == {"type": "state", "value": "IDLE"}


@pytest.mark.skipif(not settings.model_path.exists(), reason="no exported model")
def test_the_classifier_is_actually_connected(client):
    """A real clip replayed through the socket must come back as its own gloss.

    Skipped without the corpus. When it runs it is the only check that the model, the
    feature spec and the wire format all agree — each is fine in isolation and the
    system is useless if any pair disagrees.
    """
    from ml.data.manifest import load
    from ml.dataset import features_for_clip

    clips = [c for c in load() if c.signer.startswith("signer")]
    if not clips:
        pytest.skip("no ingested clips")

    hits = 0
    tried = 0
    for clip in clips[:12]:
        seq = features_for_clip(clip.clip)
        if len(seq) < 10:
            continue
        tried += 1
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            vectors = [v.tolist() for v in seq]
            events = replay(ws, still(5) + vectors + still(20, at=float(seq[-1][L_WRIST * 3])))
        said = [e["value"] for e in events if e["type"] == "gloss"]
        hits += clip.gloss in said

    if tried == 0:
        pytest.skip("no usable cached clips")
    # Far below the model's accuracy on purpose. This asserts "wired up", not "accurate";
    # live segmentation also re-cuts the clip, so some takes legitimately miss.
    assert hits > 0, f"no clip of {tried} was recognised — the classifier is not connected"
