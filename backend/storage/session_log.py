"""SQLite session logging (PRD §3.3, §5.2).

Every WebSocket connection is a session row; every gloss the recogniser reports and
every sentence the assembler emits is a timestamped event under it, logged at the same
point `backend/main.py` sends the matching event onto the wire. `GET /sessions/{id}`
reads this back as that session's transcript log.

Out of scope deliberately kept out: no auth, no multi-session queries, no cloud database
(PRD §2.2 — SQLite locally, "and only for session logs"). This is a demo-support feature,
not a product surface.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    opened_at  REAL NOT NULL,
    closed_at  REAL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    kind        TEXT NOT NULL CHECK (kind IN ('gloss', 'transcript')),
    value       TEXT NOT NULL,
    confidence  REAL,
    ts          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """A fresh connection against the *current* `settings.db_path`, committed and closed
    when the `with` block exits — plain `sqlite3.Connection` only commits or rolls back
    as a context manager and never closes, which leaks a file handle per call here.

    Reading the setting inside the function rather than caching it at import time is
    what lets a test point this at a tmp_path database with `monkeypatch.setattr(
    settings, "db_path", ...)` — the same overridability every other tunable gets
    (PROJECT_RULES rule 5), just applied to a path instead of a threshold.

    One connection per call rather than a pooled one: writes only happen on a segment
    boundary or a sentence, at most a few times a second, so a fresh `sqlite3.connect`
    costs nothing that matters here, and nothing needs to know when the app is shutting
    down to flush it.
    """
    path = Path(settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def open_session(session_id: str) -> None:
    """Record a session starting. Called once, when the WebSocket accepts."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, opened_at) VALUES (?, ?)",
            (session_id, time.time()),
        )


def close_session(session_id: str) -> None:
    """Stamp when a session ended. Called from the handler's `finally`, so this still
    runs on a client disconnect, not only on a clean `stop`."""
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET closed_at = ? WHERE id = ?", (time.time(), session_id),
        )


def log_gloss(session_id: str, gloss: str, confidence: float) -> None:
    """One recognised segment, logged wherever the matching `gloss` event goes out on
    the wire — including a `gloss` of UNKNOWN. The recogniser always reports its real
    top confidence even then (see `backend/pipeline/recogniser.py`), so the log can show
    how close a rejected sign came, the same reason the UI is shown it."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (session_id, kind, value, confidence, ts) "
            "VALUES (?, 'gloss', ?, ?, ?)",
            (session_id, gloss, confidence, time.time()),
        )


def log_sentence(session_id: str, text: str) -> None:
    """One assembled sentence, logged wherever the matching `transcript` event goes out
    — both the template match and the timeout fallback in `backend/main.py`."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (session_id, kind, value, confidence, ts) "
            "VALUES (?, 'transcript', ?, NULL, ?)",
            (session_id, text, time.time()),
        )


def get_session(session_id: str) -> dict | None:
    """The full transcript log for one session, or None if no such session was ever
    opened. `backend/main.py` turns the None case into the 404 (PRD §5.2).

    Events come back shaped like the WebSocket events that produced them — `type` and
    `value`/`confidence` for a gloss, `type` and `text` for a transcript — so a client
    already parsing the live stream reads the replay the same way.
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        session = conn.execute(
            "SELECT id, opened_at, closed_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            return None
        rows = conn.execute(
            "SELECT kind, value, confidence, ts FROM events "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    def event(row: sqlite3.Row) -> dict:
        if row["kind"] == "gloss":
            return {"type": "gloss", "value": row["value"],
                    "confidence": row["confidence"], "ts": row["ts"]}
        return {"type": "transcript", "text": row["value"], "ts": row["ts"]}

    return {
        "session_id": session["id"],
        "opened_at": session["opened_at"],
        "closed_at": session["closed_at"],
        "events": [event(r) for r in rows],
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        settings.db_path = Path(tmp) / "self_check.db"

        assert get_session("nope") is None, "an unopened session must not be found"

        open_session("abc")
        log_gloss("abc", "HELLO", 0.91)
        log_gloss("abc", "UNKNOWN", 0.42)
        log_sentence("abc", "Hello.")
        close_session("abc")

        result = get_session("abc")
        assert result["session_id"] == "abc"
        assert result["closed_at"] is not None, "close_session must stamp closed_at"
        assert [e["type"] for e in result["events"]] == ["gloss", "gloss", "transcript"]
        assert result["events"][0] == {
            "type": "gloss", "value": "HELLO", "confidence": 0.91,
            "ts": result["events"][0]["ts"],
        }
        assert result["events"][2] == {
            "type": "transcript", "text": "Hello.", "ts": result["events"][2]["ts"],
        }

        # Opening the same id twice must not reset an in-progress session's start time.
        opened_at_before = get_session("abc")["opened_at"]
        open_session("abc")
        assert get_session("abc")["opened_at"] == opened_at_before

    print("session_log.py self-check ok")
