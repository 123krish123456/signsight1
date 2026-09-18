"""Vocab pack validation (PRD §9) — including the language-agnostic claim."""

import json

import pytest
from pydantic import ValidationError

from backend.config import ROOT
from backend.vocab.schema import UNKNOWN, VocabPack, load_pack

ISL = ROOT / "backend" / "vocab" / "isl_v1.json"
ASL = ROOT / "backend" / "vocab" / "asl_v1.json"


def test_isl_pack_matches_appendix_a():
    pack = load_pack(ISL)
    assert len(pack.entries) == 50
    assert len(pack.letters) == 26
    assert len(pack.labels) == 51 and pack.labels[-1] == UNKNOWN


def test_templates_sorted_longest_first():
    lengths = [len(t.pattern) for t in load_pack(ISL).sorted_templates()]
    assert lengths == sorted(lengths, reverse=True)


def test_asl_stub_loads_with_zero_python_changes():
    """Swapping the pack must not require touching code (PRD §4.6)."""
    pack = load_pack(ASL)
    assert pack.language == "American Sign Language"
    assert pack.labels[-1] == UNKNOWN


def _pack(**over) -> dict:
    base = json.loads(ISL.read_text(encoding="utf-8"))
    base.update(over)
    return base


def test_rejects_duplicate_glosses():
    entries = _pack()["entries"]
    with pytest.raises(ValidationError, match="duplicate"):
        VocabPack.model_validate(_pack(entries=entries + [entries[0]]))


def test_rejects_explicit_unknown_entry():
    entries = _pack()["entries"] + [{"gloss": UNKNOWN, "pos": "noun"}]
    with pytest.raises(ValidationError, match="implicit"):
        VocabPack.model_validate(_pack(entries=entries))


def test_rejects_template_with_unknown_gloss():
    bad = [{"pattern": ["ME", "PIZZA"], "output": "I want pizza."}]
    with pytest.raises(ValidationError, match="unknown glosses"):
        VocabPack.model_validate(_pack(templates=bad))


def test_rejects_unknown_placeholder():
    bad = [{"pattern": ["ME", "$WIDGET"], "output": "x"}]
    with pytest.raises(ValidationError, match="unknown placeholder"):
        VocabPack.model_validate(_pack(templates=bad))
