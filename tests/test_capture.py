"""Clip upload bookkeeping (PRD §7 M2).

The recorder shows the returned count back to whoever is recording, so an inflated one
tells them they have more clips of a sign than exist.
"""

import io

from fastapi.testclient import TestClient

from backend.main import app


def _upload(client, signer, gloss):
    return client.post("/capture/clip", data={"signer": signer, "gloss": gloss},
                       files={"video": ("c.webm", io.BytesIO(b"x" * 64), "video/webm")})


def test_upload_reports_the_number_of_clips_that_exist(tmp_path, monkeypatch):
    from backend import capture

    # ROOT too, not just CLIPS_DIR: the handler stores paths relative to the project
    # root, so a clips directory outside it cannot be expressed.
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    monkeypatch.setattr(capture, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(capture, "load", lambda: [])
    saved: list = []
    monkeypatch.setattr(capture, "save", lambda c: saved.extend(c))

    with TestClient(app) as client:
        body = _upload(client, "checkbot", "HELLO").json()

    assert body["count"] == 1, f"one upload reported {body['count']} clips"
    assert body["saved"].endswith("checkbot_HELLO_000.webm"), body["saved"]
    assert body["bytes"] == 64
