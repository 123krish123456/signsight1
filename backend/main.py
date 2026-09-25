"""SignSight backend — FastAPI + the landmark WebSocket (PRD §5).

Accepts landmark frames, buffers them, runs the motion-energy segmenter over the stream
to find sign boundaries, classifies each segment, and assembles the recognised glosses
into English with the vocab pack's templates. That is the whole live path (M4).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.capture import REFERENCE_DIR, router as capture_router
from backend.config import ROOT, settings
from backend.pipeline.assembler import Assembler
from backend.pipeline.buffer import Frame, FrameBuffer
from backend.mock import MockRecogniser
from backend.pipeline.recogniser import Recogniser
from backend.pipeline.segmenter import Segmenter
from backend.vocab.schema import UNKNOWN, load_pack

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("signsight")

MODEL_NAME = "none — set SIGNSIGHT_MODEL_PATH or export one"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pack = load_pack(settings.vocab_pack)
    log.info(
        "vocab %s loaded: %d classes (+UNKNOWN), expecting %d-d frames",
        app.state.pack.name, len(app.state.pack.entries), settings.expected_dim,
    )

    # Loaded once for the process: an ONNX session is a few MB and thread-safe to call,
    # so building one per WebSocket would add a second of latency to every connection
    # for nothing. A missing model is not fatal — the segmenter still works and every
    # segment reports UNKNOWN, which is exactly the pre-M4 behaviour.
    app.state.recogniser = None
    if settings.mock_recognition:
        log.warning("MOCK RECOGNITION IS ON — glosses are scripted, the camera is ignored")
    elif not settings.model_path.exists():
        log.warning(
            "no model at %s — segmentation only, every segment will report UNKNOWN. "
            "Run: python -m ml.export_onnx --model ml/models/signsight_isl24.keras",
            settings.model_path,
        )
    else:
        try:
            app.state.recogniser = Recogniser(pack=app.state.pack)
        except Exception:
            log.exception("could not load %s — continuing without a classifier",
                          settings.model_path)
    yield


app = FastAPI(title="SignSight", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(capture_router)
# Reference clips for the browser recorder. Versioned in the repo so a fresh clone
# can record without the 57 GB corpus.
if REFERENCE_DIR.exists():
    app.mount("/reference", StaticFiles(directory=REFERENCE_DIR), name="reference")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.model_path.name if app.state.recogniser else MODEL_NAME,
        "mock": settings.mock_recognition,
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
    recogniser = app.state.recogniser
    assembler = Assembler(pack=app.state.pack)
    last_state = "IDLE"
    segments_seen = 0
    warned_drop = False
    t_open = time.monotonic()
    log.info("session %s open", session_id[:8])

    try:
        await ws.send_json({"type": "state", "value": "IDLE"})
        while True:
            try:
                msg = await ws.receive_json()
            except json.JSONDecodeError:
                # A client sending malformed JSON used to take the whole session down
                # with an unhandled exception and no explanation, which is a miserable
                # way to find a bug in your own client. Every other bad input already
                # gets an error event; this one should too.
                await ws.send_json({
                    "type": "error", "code": "BAD_MESSAGE",
                    "message": "not valid JSON",
                })
                continue
            if not isinstance(msg, dict):
                await ws.send_json({
                    "type": "error", "code": "BAD_MESSAGE",
                    "message": f"expected an object, got {type(msg).__name__}",
                })
                continue
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
                    # With no classifier loaded every segment is UNKNOWN, which the UI
                    # renders as "…". The PRD requires the user be able to tell "not
                    # understood" from "not signing", so it is never silently dropped.
                    took = 0.0
                    if mock:
                        gloss, confidence = mock.classify(segment)
                    elif recogniser:
                        # ONNX is ~12 ms and blocks the event loop. At 15 FPS that is a
                        # fifth of one frame interval and only on a segment boundary, so
                        # a thread pool would cost more in complexity than it saves.
                        gloss, confidence, took = recogniser.classify(segment)
                    else:
                        gloss, confidence = UNKNOWN, 0.0

                    await ws.send_json({
                        "type": "gloss", "value": gloss,
                        "confidence": round(confidence, 3),
                        "segment_ms": round(segment.duration_ms),
                    })
                    log.info(
                        "session %s   -> %s (%.2f)%s",
                        session_id[:8], gloss, confidence,
                        f" {took:.0f} ms" if took else "",
                    )

                    # One path for both: the mock emits glosses and the real assembler
                    # turns them into English, so mock output cannot drift away from
                    # what a real session would say.
                    sentence = assembler.push(gloss)
                    if sentence is not None:
                        await ws.send_json({
                            "type": "transcript", "text": sentence, "is_final": True,
                        })

                # A gloss run the templates never match must not strand the signer
                # waiting for a sentence that is not coming (PRD §4.6).
                if assembler.due() and (text := assembler.flush()):
                    await ws.send_json({
                        "type": "transcript", "text": text, "is_final": True,
                    })

                if buf.received % (settings.target_fps * 5) == 0:
                    log.info("session %s %s", session_id[:8], buf.stats())

            elif kind == "control":
                action = msg.get("action")
                if action == "reset_buffer":
                    buf.clear()
                    segmenter.reset()
                    assembler.reset()
                    if recogniser:
                        recogniser.reset()
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
