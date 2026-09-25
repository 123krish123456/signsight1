"""SQLite session logging and its REST endpoint (PRD §3.3, §5.2).

Two layers, tested separately: `backend/storage/session_log.py` against a throwaway
database, then `GET /sessions/{id}` end to end over a real `/ws/stream` connection —
because a log that is correct in isolation is not the same claim as "the live handler
actually calls it".
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from backend.storage import session_log
from ml.features.extract import FEATURE_DIM, L_WRIST, R_WRIST


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test in this file gets its own database, never the real `signsight.db`."""
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")


# --- backend/storage/session_log.py, directly ---------------------------------------

def test_unopened_session_is_not_found():
    assert session_log.get_session("never-heard-of-it") is None


def test_open_then_close_round_trips_timestamps():
    session_log.open_session("s1")
    before = session_log.get_session("s1")
    assert before["session_id"] == "s1"
    assert before["opened_at"] > 0
    assert before["closed_at"] is None, "an open session has not closed yet"

    session_log.close_session("s1")
    after = session_log.get_session("s1")
    assert after["closed_at"] is not None
    assert after["closed_at"] >= after["opened_at"]


def test_opening_twice_does_not_reset_the_start_time():
    """A reconnect under the same id (there isn't one yet, but nothing should assume
    there won't be) must not look like the session restarted."""
    session_log.open_session("s1")
    first = session_log.get_session("s1")["opened_at"]
    session_log.open_session("s1")
    assert session_log.get_session("s1")["opened_at"] == first


def test_gloss_and_sentence_events_come_back_in_order_with_the_right_shape():
    session_log.open_session("s1")
    session_log.log_gloss("s1", "HELLO", 0.91)
    session_log.log_gloss("s1", "UNKNOWN", 0.42)
    session_log.log_sentence("s1", "Hello.")

    events = session_log.get_session("s1")["events"]
    assert [e["type"] for e in events] == ["gloss", "gloss", "transcript"]

    gloss = events[0]
    assert gloss["value"] == "HELLO" and gloss["confidence"] == 0.91

    # UNKNOWN is logged too, with whatever confidence the classifier actually reported —
    # the same reason it is sent over the wire rather than dropped (PRD §4.5).
    unknown = events[1]
    assert unknown["value"] == "UNKNOWN" and unknown["confidence"] == 0.42

    sentence = events[2]
    assert sentence["text"] == "Hello." and "confidence" not in sentence


def test_events_are_scoped_to_their_own_session():
    session_log.open_session("a")
    session_log.open_session("b")
    session_log.log_gloss("a", "HELLO", 0.9)
    session_log.log_gloss("b", "MOTHER", 0.9)

    assert [e["value"] for e in session_log.get_session("a")["events"]] == ["HELLO"]
    assert [e["value"] for e in session_log.get_session("b")["events"]] == ["MOTHER"]


# --- GET /sessions/{id}, and the WebSocket handler that feeds it --------------------

def frame(offset: float) -> list[float]:
    """A valid, hand-present frame with both wrists at a given x.

    Matches `tests/test_live.py`'s convention: the segmenter now ignores wrist motion
    when no hand is detected, so the hand block has to be non-zero too, not just the
    `hands_present` flag on the wire message.
    """
    v = np.zeros(FEATURE_DIM)
    v[:75] = 0.01
    v[75:75 + 63] = 0.02
    for idx in (L_WRIST, R_WRIST):
        v[idx * 3] = offset
    return v.tolist()


def test_unknown_session_id_is_404():
    with TestClient(app) as client:
        resp = client.get("/sessions/does-not-exist")
    assert resp.status_code == 404


def test_a_live_session_can_be_read_back_afterwards():
    """Drive a real WebSocket connection, then confirm the REST endpoint sees exactly
    the gloss the live path produced — the thing that actually proves main.py is
    wired to session_log rather than just importing it."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["value"] == "IDLE"
            still = [frame(0.0)] * 5
            motion = [frame(i * 0.3) for i in range(20)]
            settle = [frame(20 * 0.3)] * 20
            for i, vec in enumerate(still + motion + settle):
                ws.send_json({"type": "frame", "seq": i, "t_client_ms": 0.0,
                              "landmarks": vec, "hands_present": [True, True]})
            ws.send_json({"type": "control", "action": "stop"})

        # Recover the id the server assigned; it is never sent to the client (PRD §5.1
        # only specifies it client → server), so the log itself is the only way to find
        # a session afterwards — which is the whole point of the endpoint.
        with session_log._connect() as conn:
            session_id = conn.execute(
                "SELECT id FROM sessions ORDER BY opened_at DESC LIMIT 1"
            ).fetchone()[0]

        resp = client.get(f"/sessions/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == session_id
    assert body["closed_at"] is not None, "the 'stop' control message must close the session"
    glosses = [e for e in body["events"] if e["type"] == "gloss"]
    assert len(glosses) == 1, f"expected the one segment the motion trace produces: {body['events']}"
    assert 0.0 <= glosses[0]["confidence"] <= 1.0
