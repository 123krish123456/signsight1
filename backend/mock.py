"""Fake recogniser, so the front ends can be exercised without signing at the camera.

Enable with `SIGNSIGHT_MOCK_RECOGNITION=true`. Every detected segment then returns a
scripted gloss instead of whatever the model thinks, so the transcript, the speech output
and the extension overlay can all be driven from a desk.

It emits **glosses only**. The sentence is built by the real assembler from the real
templates, exactly as it is for the real classifier — so the mock cannot drift away from
what the system actually says. An earlier version carried its own hardcoded English and
a vocabulary from a pack we stopped shipping, which meant it demonstrated sentences the
system could not produce.

This is a development aid. It never runs unless the flag is set, and it does not look at
the landmarks at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Gloss sequences a signer would actually produce, using the shipped vocabulary. Each
# one matches a template in the pack, so the assembler turns it into English:
# "I am happy.", "The mother is sick.", "Hello." and so on.
SCRIPTS: list[list[str]] = [
    ["HELLO"],
    ["ME", "HAPPY"],
    ["HOW-ARE-YOU"],
    ["ME", "ALRIGHT"],
    ["MOTHER", "SICK"],
    ["YOU", "PLEASED"],
    ["FATHER", "HEALTHY"],
    ["THANK-YOU"],
    ["SHE", "COLD"],
    ["HOUSE", "BIG"],
    ["GOOD-MORNING"],
    ["WE", "HAPPY"],
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

        gloss = SCRIPTS[self._script][self._step]
        self._step += 1
        if self._step >= len(SCRIPTS[self._script]):
            self._script = (self._script + 1) % len(SCRIPTS)
            self._step = 0
        return gloss, round(self._rng.uniform(0.78, 0.99), 2)

    def reset(self) -> None:
        self._script = self._step = 0


if __name__ == "__main__":
    from backend.config import settings
    from backend.pipeline.assembler import Assembler
    from backend.vocab.schema import load_pack

    pack = load_pack(settings.vocab_pack)
    known = set(pack.labels)

    # The failure this guards against: a mock that names signs the shipped vocabulary
    # does not contain, demonstrating a system that cannot exist.
    scripted = {g for script in SCRIPTS for g in script}
    assert scripted <= known, f"not in {pack.name}: {sorted(scripted - known)}"

    # And every script must actually assemble into English, or the mock shows glosses
    # where a real session would show a sentence.
    for script in SCRIPTS:
        a = Assembler(pack=pack)
        out = [a.push(g) for g in script]
        assert out[-1], f"{script} matches no template"

    m = MockRecogniser(seed=1)
    seen = [m.classify()[0] for _ in range(60)]
    assert "UNKNOWN" in seen, "should occasionally fail to recognise"
    assert set(seen) - {"UNKNOWN"} <= known
    assert all(0.0 <= c <= 1.0 for c in (m.classify()[1] for _ in range(20)))

    print(f"mock.py self-check ok — {len(SCRIPTS)} scripts, all assemble")
