"""SignSight backend — FastAPI + the landmark WebSocket (PRD §5).

Accepts landmark frames, buffers them, and runs the motion-energy segmenter over the
stream so sign boundaries are detected live. Classification and sentence assembly hang
off the same path once a model exists (M3/M4).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.pipeline.buffer import Frame, FrameBuffer
from backend.mock import MockRecogniser
from backend.pipeline.segmenter import Segmenter
from backend.vocab.schema import load_pack

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("signsight")

MODEL_NAME = "none — segmenter only, classifier arrives in M3"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pack = load_pack(settings.vocab_pack)
    log.info(
        "vocab %s loaded: %d classes (+UNKNOWN), expecting %d-d frames",
        app.state.pack.name, len(app.state.pack.entries), settings.expected_dim,
    )
    yield


app = FastAPI(title="SignSight", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "vocab": app.state.pack.name,
        "feature_dim": settings.expected_dim,
        "target_fps": settings.target_fps,
    }


@app.get("/vocab")
def vocab() -> dict:
    """Clients render the sign reference sheet from this."""
    return app.state.pack.model_dump()


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    session_id = str(uuid.uuid4())
    buf = FrameBuffer(capacity=settings.max_queue_frames)
    segmenter = Segmenter()
    mock = MockRecogniser() if settings.mock_recognition else None
    last_state = "IDLE"
    segments_seen = 0
    warned_drop = False
    t_open = time.monotonic()
    log.info("session %s open", session_id[:8])

    try:
        await ws.send_json({"type": "state", "value": "IDLE"})
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "frame":
                lm = msg.get("landmarks")
                if not isinstance(lm, list) or len(lm) != settings.expected_dim:
                    await ws.send_json({
                        "type": "error",
                        "code": "BAD_DIMENSION",
                        "message": f"expected {settings.expected_dim} floats, got {len(lm) if isinstance(lm, list) else type(lm).__name__}",
                    })
                    continue

                lost = buf.push(Frame(
                    seq=int(msg.get("seq", buf.received)),
                    t_client_ms=float(msg.get("t_client_ms", 0.0)),
                    t_server_ms=time.time() * 1000,
                    vector=np.asarray(lm, dtype=np.float32),
                    hands_present=tuple(msg.get("hands_present", (False, False))),
                ))
                if lost and not warned_drop:
                    warned_drop = True
                    log.warning(
                        "session %s: unconsumed frames evicted at %d-frame capacity — "
                        "the pipeline is behind the stream",
                        session_id[:8], buf.capacity,
                    )
                seq = int(msg.get("seq", buf.received))
                segment = segmenter.push(buf.frames[-1].vector, seq=seq)

                # The segmenter is the consumer now, so `lost` is a real backpressure
                # signal from here on. The classifier joins this path in M3/M4 and,
                # being the expensive step, will move to a thread pool (PRD §5.3).
                buf.mark_consumed(seq)

                if segmenter.state.value != last_state:
                    last_state = segmenter.state.value
                    await ws.send_json({"type": "state", "value": last_state})

                if segment is not None:
                    segments_seen += 1
                    log.info(
                        "session %s segment #%d: %d frames (%.0f ms)%s",
                        session_id[:8], segments_seen, segment.raw_length,
                        segment.duration_ms, " [truncated]" if segment.truncated else "",
                    )
                    # No classifier until M3. The PRD requires the user be able to tell
                    # "not understood" from "not signing", so an unclassified segment
                    # surfaces as UNKNOWN, which the UI renders as "…" (§4.5).
                    gloss, confidence = mock.classify(segment) if mock else ("UNKNOWN", 0.0)
                    await ws.send_json({
                        "type": "gloss", "value": gloss,
                        "confidence": confidence, "segment_ms": round(segment.duration_ms),
                    })
                    if mock and (sentence := mock.advance()) is not None:
                        await ws.send_json({
                            "type": "transcript", "text": sentence, "is_final": True,
                        })

                if buf.received % (settings.target_fps * 5) == 0:
                    log.info("session %s %s", session_id[:8], buf.stats())

            elif kind == "control":
                action = msg.get("action")
                if action == "reset_buffer":
                    buf.clear()
                    segmenter.reset()
                    if mock:
                        mock.reset()
                    last_state = "IDLE"
                elif action == "stop":
                    break
                await ws.send_json({"type": "state", "value": "IDLE"})

            else:
                await ws.send_json({
                    "type": "error", "code": "BAD_MESSAGE",
                    "message": f"unknown type {kind!r}",
                })

    except WebSocketDisconnect:
        pass
    finally:
        secs = time.monotonic() - t_open
        log.info(
            "session %s closed after %.1fs — %s, mean %.1f FPS",
            session_id[:8], secs, buf.stats(), buf.received / secs if secs else 0.0,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
