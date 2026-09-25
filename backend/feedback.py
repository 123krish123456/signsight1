"""Confirm or correct a live recognition, and keep the segment as training data.

The gap this closes: a signer's recorded clips and their live signing are not the same
thing. The clips in this project score 97.6% through the live pipeline offline, while the
same person in front of the camera produces confidences half that. Whatever the
difference is, the only data that captures it is the live segment itself.

So every judgement the signer makes — "yes that was HELLO", "no, that was THANK-YOU" —
saves the 45x261 segment that produced it, labelled. Those land in the manifest as
ordinary clips, because `features_for_clip` treats a `.npy` path as features already, and
from there cross-validation and training need no changes at all.

Kept honestly: a corrected segment is stored under the signer who made it, so
signer-disjoint splits still hold and nothing leaks between train and test.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import ROOT, settings
from backend.vocab.schema import load_pack
from ml.data.manifest import Clip, load, save

log = logging.getLogger("signsight")
router = APIRouter(tags=["feedback"])

LIVE_DIR = ROOT / "ml" / "data" / "live"
SAFE_SIGNER = re.compile(r"^[a-z0-9_-]{1,32}$")

# The segments still on offer to judge. Bounded because this is a live buffer, not a
# store: anything the signer has not judged within a few signs, they never will.
RECENT: OrderedDict[str, np.ndarray] = OrderedDict()
MAX_PENDING = 32


def remember(frames: np.ndarray) -> str:
    """Hold one segment so a later judgement can refer to it. Returns its id."""
    token = uuid.uuid4().hex[:12]
    RECENT[token] = frames.astype(np.float32)
    while len(RECENT) > MAX_PENDING:
        RECENT.popitem(last=False)
    return token


class Judgement(BaseModel):
    segment: str = Field(min_length=1, max_length=64)
    gloss: str = Field(min_length=1, max_length=64)
    signer: str = Field(min_length=1, max_length=32)


@router.post("/feedback")
def judge(body: Judgement) -> dict:
    """Store one judged segment as a labelled training example."""
    if not SAFE_SIGNER.match(body.signer):
        raise HTTPException(400, "signer must be 1-32 chars of a-z, 0-9, _ or -")

    pack = load_pack(settings.vocab_pack)
    if body.gloss not in {e.gloss for e in pack.entries}:
        raise HTTPException(400, f"{body.gloss!r} is not in {pack.name}")

    frames = RECENT.pop(body.segment, None)
    if frames is None:
        raise HTTPException(404, "that segment has expired — judge a sign soon after it")

    out_dir = LIVE_DIR / body.signer / body.gloss
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{body.segment}.npy"
    np.save(path, frames)

    rel = path.relative_to(ROOT).as_posix()
    clips = load()
    clips.append(Clip(clip=rel, gloss=body.gloss, signer=body.signer, source="live"))
    save(clips)

    mine = sum(1 for c in clips if c.source == "live" and c.signer == body.signer)
    log.info("feedback: %s confirmed as %s (%d live examples)", body.segment[:8],
             body.gloss, mine)
    return {"saved": rel, "gloss": body.gloss, "live_examples": mine}


@router.get("/feedback/count")
def count() -> dict:
    """How much live-confirmed data exists, per sign."""
    per: dict[str, int] = {}
    for c in load():
        if c.source == "live":
            per[c.gloss] = per.get(c.gloss, 0) + 1
    return {"total": sum(per.values()), "per_sign": per, "pending": len(RECENT)}
