"""Vocabulary pack format (PRD §4.6).

Language-agnostic requirement: swapping `isl_v1.json` for `asl_v1.json` and retraining
must need ZERO Python changes. Everything language-specific lives in the pack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

UNKNOWN = "UNKNOWN"
PartOfSpeech = Literal["letter", "pronoun", "noun", "verb", "adj", "question", "interjection"]

# Placeholders usable in template patterns. Resolved by the assembler (M5).
PLACEHOLDERS = {"$FINGERSPELL", "$NOUN", "$ADJ", "$VERB", "$ANY"}


class VocabEntry(BaseModel):
    gloss: str
    pos: PartOfSpeech
    mirror_safe: bool = True  # false → left/right mirror augmentation changes the meaning
    repeatable: bool = False  # true → exempt from the repeat cooldown (PRD §4.5)
    reference_video: str | None = None


class Template(BaseModel):
    pattern: list[str] = Field(min_length=1)
    output: str

    @field_validator("pattern")
    @classmethod
    def known_tokens(cls, v: list[str]) -> list[str]:
        for tok in v:
            if tok.startswith("$") and tok not in PLACEHOLDERS:
                raise ValueError(f"unknown placeholder {tok}; allowed: {sorted(PLACEHOLDERS)}")
        return v


class VocabPack(BaseModel):
    name: str
    language: str
    version: str
    entries: list[VocabEntry] = Field(min_length=1)
    templates: list[Template] = []
    fallback: Literal["join_with_spaces_and_capitalise"] = "join_with_spaces_and_capitalise"

    @model_validator(mode="after")
    def check(self) -> VocabPack:
        glosses = [e.gloss for e in self.entries]
        dupes = {g for g in glosses if glosses.count(g) > 1}
        if dupes:
            raise ValueError(f"duplicate glosses: {sorted(dupes)}")
        if UNKNOWN in glosses:
            raise ValueError(f"{UNKNOWN} is implicit — do not list it as an entry")
        known = set(glosses)
        for t in self.templates:
            unknown = [tok for tok in t.pattern if not tok.startswith("$") and tok not in known]
            if unknown:
                raise ValueError(f"template {t.pattern} references unknown glosses {unknown}")
        return self

    @property
    def labels(self) -> list[str]:
        """Class order for the classifier: vocabulary then UNKNOWN (PRD §4.4)."""
        return [e.gloss for e in self.entries] + [UNKNOWN]

    @property
    def letters(self) -> set[str]:
        return {e.gloss for e in self.entries if e.pos == "letter"}

    def sorted_templates(self) -> list[Template]:
        """Longest-first — PRD §4.6 matches specific patterns before general ones."""
        return sorted(self.templates, key=lambda t: len(t.pattern), reverse=True)


def load_pack(path: str | Path) -> VocabPack:
    return VocabPack.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
