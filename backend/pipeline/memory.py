"""Remember corrected signs, and recognise them again straight away.

Retraining takes ten minutes and a restart. Someone correcting the system in front of the
camera expects the correction to hold on the next sign, not tomorrow — so confirmed
segments also go into a plain nearest-neighbour memory that is consulted whenever the
classifier is unsure.

Measured on 158 confirmed segments from one signer: the nearest neighbour is the same
sign 79% of the time, and above a cosine similarity of 0.97 that rises to 85%. Both beat
what the model manages on the same person live, which is the whole reason this exists.

It is deliberately a fallback, not a replacement. A confident classifier is still trusted
first: the memory holds one person's handful of examples, while the model holds eleven
signers and a thousand clips, and preferring the smaller evidence would be daft.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.config import ROOT, settings

log = logging.getLogger("signsight")

# Above this cosine similarity a remembered segment is taken as a match. From the
# measurement above: 0.97 fires on 79% of segments and is right 85% of the time, against
# 0.90 which fires on everything and is right 79%. Overridable like every other tunable.
MATCH_THRESHOLD = 0.97


def _flat(frames: np.ndarray) -> np.ndarray:
    v = np.asarray(frames, np.float32).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-9)


@dataclass
class SignMemory:
    """Unit-norm segment vectors and their labels, searched by cosine similarity."""

    threshold: float = MATCH_THRESHOLD
    _vectors: list[np.ndarray] = field(default_factory=list)
    _labels: list[str] = field(default_factory=list)
    _signers: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._labels)

    def add(self, frames: np.ndarray, gloss: str, signer: str) -> None:
        self._vectors.append(_flat(frames))
        self._labels.append(gloss)
        self._signers.append(signer)

    def load(self) -> int:
        """Every live-confirmed segment in the manifest. Called once at startup."""
        from ml.data.manifest import load as load_manifest

        for c in load_manifest():
            if c.source != "live":
                continue
            path = ROOT / c.clip
            if not path.exists():
                continue
            try:
                self.add(np.load(path), c.gloss, c.signer)
            except Exception:  # a truncated file must not stop the server starting
                log.warning("could not read remembered segment %s", c.clip)
        return len(self)

    def lookup(self, frames: np.ndarray, signer: str | None = None) -> tuple[str, float] | None:
        """Closest remembered sign, or None if nothing is close enough.

        `signer` narrows the search to one person's corrections when given. How somebody
        signs is personal, and a match against a different person's memory is a much
        weaker claim than a match against your own.
        """
        if not self._vectors:
            return None
        pool = range(len(self._vectors))
        if signer is not None:
            mine = [i for i in pool if self._signers[i] == signer]
            pool = mine or pool

        q = _flat(frames)
        idx = list(pool)
        sims = np.array([float(q @ self._vectors[i]) for i in idx])
        best = int(np.argmax(sims))
        if sims[best] < self.threshold:
            return None
        return self._labels[idx[best]], float(sims[best])


MEMORY = SignMemory()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = SignMemory(threshold=0.9)
    assert m.lookup(rng.random((45, 261))) is None, "empty memory must match nothing"

    a = rng.random((45, 261)).astype(np.float32)
    m.add(a, "HELLO", "eashan")
    hit = m.lookup(a)
    assert hit is not None and hit[0] == "HELLO" and hit[1] > 0.99, hit

    # Something unrelated must not match.
    assert m.lookup(-a) is None, "an opposite vector must not be recalled as a match"

    # A different person's memory is only used when yours holds nothing.
    m.add(rng.random((45, 261)).astype(np.float32), "SICK", "krish")
    assert m.lookup(a, signer="eashan")[0] == "HELLO"
    assert len(m) == 2
    print("memory.py self-check ok")
