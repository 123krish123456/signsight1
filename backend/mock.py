"""Fake recogniser, so the front ends can be built before a model exists.

Enable with `SIGNSIGHT_MOCK_RECOGNITION=true`. Every detected segment then returns a
scripted gloss instead of UNKNOWN, and finished sentences arrive as transcript events —
so the transcript view, speech output and the extension overlay can all be built and
demoed today, against exactly the event shapes the real pipeline will send (PRD §5.1).

This is a development aid and never runs unless the flag is set. It is not a classifier
and does not look at the landmarks at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Each script is the gloss sequence a signer would produce, and the English the
# assembler will eventually build from it. Chosen to match the templates already in
# isl_v1.json, so the mocked output looks like the real output.
SCRIPTS: list[tuple[list[str], str]] = [
    (["ME", "NAME", "E", "A", "S", "H", "A", "N"], "My name is Eashan."),
    (["YOU", "NAME", "WHAT"], "What is your name?"),
    (["ME", "WATER", "WANT"], "I want water."),
    (["HELLO"], "Hello."),
    (["ME", "HELP", "NEED"], "I need help."),
    (["YOU", "HOW"], "How are you?"),
    (["ME", "UNDERSTAND", "NO"], "I do not understand."),
    (["THANK-YOU"], "Thank you."),
]


@dataclass
class MockRecogniser:
    """Walks the scripts one gloss per detected segment."""

    seed: int = 0
    _rng: random.Random = field(init=False)
    _script: int = 0
    _step: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def classify(self, _segment=None) -> tuple[str, float]:
        """Next gloss in the current script, with a plausible confidence.

        Roughly one in twelve comes back as UNKNOWN so the front ends are built against
        recognition that sometimes fails — which it will.
        """
        if self._rng.random() < 0.08:
            return "UNKNOWN", round(self._rng.uniform(0.30, 0.74), 2)
        glosses, _ = SCRIPTS[self._script]
        gloss = glosses[self._step]
        return gloss, round(self._rng.uniform(0.78, 0.99), 2)

    def advance(self) -> str | None:
        """Step the script. Returns the English sentence when one completes."""
        glosses, sentence = SCRIPTS[self._script]
        self._step += 1
        if self._step < len(glosses):
            return None
        self._script = (self._script + 1) % len(SCRIPTS)
        self._step = 0
        return sentence

    def reset(self) -> None:
        self._script = self._step = 0


if __name__ == "__main__":
    m = MockRecogniser(seed=1)
    seen, sentences = [], []
    for _ in range(40):
        gloss, conf = m.classify()
        assert 0.0 <= conf <= 1.0
        seen.append(gloss)
        if (s := m.advance()) is not None:
            sentences.append(s)
    assert sentences, "scripts must complete and yield sentences"
    assert any(g == "UNKNOWN" for g in seen), "should occasionally fail to recognise"
    assert all(s in {t for _, t in SCRIPTS} for s in sentences)
    print("mock.py self-check ok —", len(sentences), "sentences:", sentences[:3])
