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
from fastapi.responses import FileResponse

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


def _clip_dir(signer: str, gloss: str) -> Path:
    """Validated path to one signer's clips of one sign.

    Both parts come from the client, so both are checked: the signer against a strict
    pattern and the gloss against the vocabulary. Neither may contain a separator, and
    the result is confirmed to sit under the clips directory before anything touches it.
    """
    if not SAFE_NAME.match(signer):
        raise HTTPException(400, "bad signer")
    if gloss not in {e.gloss for e in load_pack(settings.vocab_pack).entries}:
        raise HTTPException(400, f"{gloss!r} is not in the vocabulary")
    path = (CLIPS_DIR / signer / gloss).resolve()
    if not path.is_relative_to(CLIPS_DIR.resolve()):
        raise HTTPException(400, "bad path")
    return path


@router.get("/capture/clips")
def list_clips(signer: str, gloss: str) -> dict:
    """Filenames of every clip this signer has of this sign, oldest first."""
    d = _clip_dir(signer, gloss)
    names = sorted(p.name for p in d.glob("*") if p.is_file()) if d.exists() else []
    return {"clips": names}


@router.get("/capture/file")
def serve_clip(signer: str, gloss: str, name: str) -> FileResponse:
    """Play back one clip so it can be reviewed before being kept or deleted."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "bad name")
    path = _clip_dir(signer, gloss) / name
    if not path.is_file():
        raise HTTPException(404, "no such clip")
    return FileResponse(path)


@router.delete("/capture/clip")
def delete_clip(signer: str, gloss: str, name: str | None = None) -> dict:
    """Delete one clip by name, or the most recent if no name is given."""
    clips = load()
    mine = [i for i, c in enumerate(clips) if c.signer == signer and c.gloss == gloss]
    if not mine:
        raise HTTPException(404, f"no clips of {gloss} by {signer}")

    if name is None:
        index = mine[-1]
    else:
        if "/" in name or "\\" in name:
            raise HTTPException(400, "bad name")
        match = [i for i in mine if Path(clips[i].clip).name == name]
        if not match:
            raise HTTPException(404, f"{name} not found")
        index = match[0]

    removed = clips.pop(index)
    (ROOT / removed.clip).unlink(missing_ok=True)
    save(clips)
    return {"removed": removed.clip}
