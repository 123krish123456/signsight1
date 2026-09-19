"""Browser-based capture: serve reference clips, accept recorded ones (PRD §7 M2).

The desktop recorder needs Python, OpenCV and MediaPipe installed. Teammates recording a
few hundred clips should not have to build a toolchain first, so the same job is exposed
over HTTP: the browser shows the reference beside the webcam, records, and posts the clip
here to be written into the same clips directory and manifest the desktop tool uses.

Deliberately unauthenticated and bound to localhost — this writes files to disk, and it
is a tool for the people building the project, not an endpoint for the internet.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import ROOT, settings
from backend.vocab.schema import load_pack
from ml.data.manifest import Clip, load, save

router = APIRouter(tags=["capture"])

CLIPS_DIR = ROOT / "ml" / "data" / "clips"
REFERENCE_DIR = ROOT / "assets" / "reference"
MAX_BYTES = 40 * 1024 * 1024
SAFE_NAME = re.compile(r"^[a-z0-9_-]{1,32}$")
EXT_FOR = {"video/webm": ".webm", "video/mp4": ".mp4", "video/x-matroska": ".webm"}


@router.get("/capture/plan")
def plan(signer: str = "") -> dict:
    """The signs to record, how many of each exist already, and where to find each
    reference clip. One call gives the browser everything it needs to run a session."""
    pack = load_pack(settings.vocab_pack)
    mine = [c for c in load() if c.signer == signer] if signer else []
    counts: dict[str, int] = {}
    for c in mine:
        counts[c.gloss] = counts.get(c.gloss, 0) + 1

    return {
        "signer": signer,
        "recorded": len(mine),
        "signs": [
            {
                "gloss": e.gloss,
                "pos": e.pos,
                "count": counts.get(e.gloss, 0),
                "reference": f"/reference/{e.gloss}.mp4"
                if (REFERENCE_DIR / f"{e.gloss}.mp4").exists() else None,
            }
            for e in pack.entries
        ],
    }


@router.post("/capture/clip")
async def upload(
    signer: str = Form(...),
    gloss: str = Form(...),
    video: UploadFile = File(...),
) -> dict:
    """Store one recorded clip and add it to the manifest."""
    if not SAFE_NAME.match(signer):
        raise HTTPException(400, "signer must be 1-32 chars of a-z, 0-9, _ or - "
                                 "(it becomes a folder name and a split key)")

    pack = load_pack(settings.vocab_pack)
    if gloss not in {e.gloss for e in pack.entries}:
        raise HTTPException(400, f"{gloss!r} is not in {pack.name}")

    payload = await video.read()
    if not payload:
        raise HTTPException(400, "empty upload")
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, f"clip is {len(payload)/1e6:.0f} MB, limit is {MAX_BYTES/1e6:.0f} MB")

    ext = EXT_FOR.get((video.content_type or "").split(";")[0], ".webm")
    out_dir = CLIPS_DIR / signer / gloss
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load()
    existing = sum(1 for c in clips if c.signer == signer and c.gloss == gloss)
    path = out_dir / f"{signer}_{gloss}_{existing:03d}{ext}"
    path.write_bytes(payload)

    clips.append(Clip(clip=path.relative_to(ROOT).as_posix(), gloss=gloss,
                      signer=signer, source="self"))
    save(clips)
    return {"saved": path.relative_to(ROOT).as_posix(), "count": existing + 1,
            "bytes": len(payload)}


@router.delete("/capture/clip")
def undo(signer: str, gloss: str) -> dict:
    """Remove that signer's most recent clip of a sign — the browser's undo."""
    clips = load()
    for i in range(len(clips) - 1, -1, -1):
        if clips[i].signer == signer and clips[i].gloss == gloss:
            removed = clips.pop(i)
            (ROOT / removed.clip).unlink(missing_ok=True)
            save(clips)
            return {"removed": removed.clip}
    raise HTTPException(404, f"no clips of {gloss} by {signer}")
