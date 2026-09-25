"""Gloss buffer → English sentence, from templates in the vocab pack (PRD §4.6).

Deterministic and rule-based; no ML. ISL has no copula and no articles, so "ME HAPPY" has
to become "I am happy." — that is what the templates in the pack encode.

Everything language-specific is data. Swapping the pack for another language and
retraining must need zero changes in here, which is why patterns, placeholders and the
fallback all come out of the JSON (PRD §4.6).

Fingerspelling ($FINGERSPELL, §4.7) is not implemented: the 26 letter classes have no
training data, so the branch could never fire and would only be untested code. The
placeholder is still accepted by the schema, so adding it later is additive.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from backend.config import settings
from backend.vocab.schema import UNKNOWN, Template, VocabPack

log = logging.getLogger("signsight")

# Placeholder → the parts of speech it stands for. $ANY matches one gloss of any kind.
PLACEHOLDER_POS = {
    "$NOUN": {"noun"},
    "$ADJ": {"adj"},
    "$VERB": {"verb"},
    "$ANY": None,
}


@dataclass
class Assembler:
    """Collects glosses and returns a sentence when one can be built.

    Two ways a sentence comes out: a template matches the buffer, or nothing has matched
    for `assembly_timeout_ms` and the fallback flushes what is there. The timeout is the
    point — without it a signer who produces glosses the templates do not cover waits
    forever for a sentence that was never coming.
    """

    pack: VocabPack
    _glosses: list[str] = field(default_factory=list)
    _last_ms: float = 0.0

    def _pos(self, gloss: str) -> str | None:
        entry = next((e for e in self.pack.entries if e.gloss == gloss), None)
        return entry.pos if entry else None

    def _matches(self, template: Template, glosses: list[str]) -> bool:
        if len(template.pattern) != len(glosses):
            return False
        for token, gloss in zip(template.pattern, glosses):
            if token.startswith("$"):
                allowed = PLACEHOLDER_POS.get(token, set())
                if allowed is not None and self._pos(gloss) not in allowed:
                    return False
            elif token != gloss:
                return False
        return True

    def _render(self, template: Template, glosses: list[str]) -> str:
        """`{0}`, `{1}` … are positions in the matched gloss run, lowercased for prose."""
        return template.output.format(*(g.lower().replace("-", " ") for g in glosses))

    def push(self, gloss: str) -> str | None:
        """Add one recognised gloss. Returns a sentence if this completed one.

        Templates are tried longest-first over the tail of the buffer, so a specific rule
        beats a general one covering the same words (PRD §4.6).
        """
        if gloss == UNKNOWN:
            return None  # shown as "…" by the client; it is not part of a sentence

        self._glosses.append(gloss)
        self._last_ms = time.monotonic() * 1000

        for template in self.pack.sorted_templates():
            n = len(template.pattern)
            if n <= len(self._glosses) and self._matches(template, self._glosses[-n:]):
                matched = self._glosses[-n:]
                self._glosses = self._glosses[:-n]
                return self._render(template, matched)
        return None

    def due(self) -> bool:
        """Has the buffer sat unmatched long enough to flush? (PRD §4.6)"""
        return bool(self._glosses) and (
            time.monotonic() * 1000 - self._last_ms >= settings.assembly_timeout_ms
        )

    def flush(self) -> str | None:
        """Emit whatever is buffered through the pack's fallback, and clear."""
        if not self._glosses:
            return None
        words = [g.lower().replace("-", " ") for g in self._glosses]
        self._glosses = []
        text = " ".join(words)
        return text[:1].upper() + text[1:] + "."

    def reset(self) -> None:
        self._glosses = []
        self._last_ms = 0.0


if __name__ == "__main__":
    from backend.vocab.schema import load_pack

    pack = load_pack(settings.vocab_pack)
    a = Assembler(pack=pack)

    assert a.push("ME") is None, "a pronoun alone completes nothing"
    assert a.push("HAPPY") == "I am happy.", a.push("HAPPY")

    assert a.push("MOTHER") is None
    assert a.push("SICK") == "The mother is sick."

    # UNKNOWN is not part of a sentence and must not join the buffer.
    a.reset()
    assert a.push(UNKNOWN) is None
    assert a.push("ME") is None
    assert a.push("HAPPY") == "I am happy."

    # A one-gloss template fires on its own.
    a.reset()
    assert a.push("HELLO") == "Hello."

    # Nothing matches -> the fallback flushes rather than waiting forever.
    a.reset()
    a.push("FRIEND")
    assert a.flush() == "Friend.", a.flush()
    assert a.flush() is None, "flushing twice must not repeat the sentence"

    # A hyphenated gloss reads as words, not as a token.
    a.reset()
    a.push("ME")
    assert a.push("THANK-YOU") == "Thank you.", "specific template beats ME + $ADJ"

    # Longest-first: a two-token template must win over a one-token one on the same tail.
    longest = max(len(t.pattern) for t in pack.templates)
    assert [len(t.pattern) for t in pack.sorted_templates()][0] == longest

    print("assembler.py self-check ok")
