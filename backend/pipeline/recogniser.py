"""Segment → gloss, with the gating the PRD requires before anything is said (§4.5).

The classifier itself is one ONNX call. Everything else here exists because a raw argmax
makes a terrible demo: it is confidently wrong on signs it has never seen, and it fires
the same gloss three times while you are still finishing the sign.

Two rules, both from §4.5:

- below `confidence_thresh`, emit UNKNOWN rather than a guess. The user has to be able to
  tell "not understood" from "not signing", so this is never silently dropped.
- suppress an immediate repeat within `repeat_cooldown_ms`, unless the pack marks that
  sign `repeatable`. Signs take longer than one segment to finish and the segmenter will
  happily cut one in half.

The PRD also describes a rolling window of the last three predictions. Nothing in the
spec then uses it, and the obvious reading — require agreement across three segments —
would delay every word by two more signs. Not implemented rather than invented; if a
stability rule is wanted later it belongs here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.config import settings
from backend.vocab.schema import UNKNOWN, VocabPack

log = logging.getLogger("signsight")


@dataclass
class Recogniser:
    """Wraps the ONNX model and the emit rules. One per session is unnecessary — the
    session owns the cooldown state, the model is shared and stateless."""

    pack: VocabPack
    model_path: Path = field(default_factory=lambda: settings.model_path)
    _session: object | None = field(default=None, repr=False)
    _input: str = ""
    _last_gloss: str = ""
    _last_ms: float = 0.0

    def __post_init__(self) -> None:
        import onnxruntime as ort  # noqa: PLC0415 — only when recognition is on

        self._session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        n_out = self._session.get_outputs()[0].shape[-1]
        if n_out != len(self.pack.labels):
            raise ValueError(
                f"{self.model_path.name} has {n_out} outputs but {self.pack.name} has "
                f"{len(self.pack.labels)} labels. The model and the pack must be the "
                f"same vintage — retrain, or point SIGNSIGHT_VOCAB_PACK at the right one."
            )
        log.info("recogniser ready: %s, %d classes", self.model_path.name, n_out)

    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        """(45, D) → softmax over `pack.labels`."""
        batch = frames[None].astype(np.float32)
        return self._session.run(None, {self._input: batch})[0][0]

    def top(self, frames: np.ndarray, n: int = 3) -> list[tuple[str, float]]:
        """The n most likely labels. Only useful for diagnosis, but very: "UNKNOWN 0.70"
        says nothing about whether the right sign came second or nowhere at all."""
        probs = self.probabilities(frames)
        return [(self.pack.labels[i], float(probs[i]))
                for i in np.argsort(probs)[::-1][:n]]

    def classify(self, segment) -> tuple[str, float, float]:
        """One segment → (gloss, confidence, inference_ms), after gating.

        `confidence` is always the model's own top probability, even when the gloss is
        forced to UNKNOWN, so the UI can show how close a rejected sign came.
        """
        t0 = time.perf_counter()
        probs = self.probabilities(segment.frames)
        took = (time.perf_counter() - t0) * 1000

        top = int(np.argmax(probs))
        confidence = float(probs[top])
        gloss = self.pack.labels[top]

        if confidence < settings.confidence_thresh:
            return UNKNOWN, confidence, took
        if gloss == UNKNOWN:
            return UNKNOWN, confidence, took

        now = time.monotonic() * 1000
        if gloss == self._last_gloss and now - self._last_ms < settings.repeat_cooldown_ms:
            entry = next((e for e in self.pack.entries if e.gloss == gloss), None)
            if entry is None or not entry.repeatable:
                return UNKNOWN, confidence, took

        self._last_gloss, self._last_ms = gloss, now
        return gloss, confidence, took

    def reset(self) -> None:
        self._last_gloss, self._last_ms = "", 0.0


if __name__ == "__main__":
    from dataclasses import dataclass as dc

    from backend.vocab.schema import load_pack

    pack = load_pack(settings.vocab_pack)
    rec = Recogniser(pack=pack)

    @dc
    class FakeSegment:
        frames: np.ndarray

    seg = FakeSegment(np.zeros((settings.segment_resample_frames, settings.feature_dim), np.float32))
    probs = rec.probabilities(seg.frames)
    assert probs.shape == (len(pack.labels),), probs.shape
    assert abs(probs.sum() - 1.0) < 1e-4, f"not a softmax: sums to {probs.sum()}"

    # The cooldown must suppress an immediate repeat of a confident gloss.
    rec._last_gloss, rec._last_ms = pack.labels[0], time.monotonic() * 1000
    entry = pack.entries[0]
    if not entry.repeatable:
        rec_probs = np.zeros(len(pack.labels), np.float32)
        rec_probs[0] = 1.0
        rec._session = type("S", (), {  # a stub session returning a certain prediction
            "run": lambda self, _o, _i: [rec_probs[None]],
            "get_inputs": lambda self: [],
        })()
        gloss, conf, _ = rec.classify(seg)
        assert gloss == UNKNOWN, f"repeat within cooldown must be suppressed, got {gloss}"
        assert conf == 1.0, "confidence is still reported for a suppressed gloss"

    print("recogniser.py self-check ok")
