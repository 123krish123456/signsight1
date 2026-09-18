"""Signer-disjoint split enforcement (PRD rule 4) — the check M3 depends on."""

import pytest

from ml.data.manifest import Clip, assign_splits, verify


def make(signers=("a", "b", "c"), glosses=("ME", "YOU"), per=25) -> list[Clip]:
    return [
        Clip(clip=f"{s}_{g}_{i}.mp4", gloss=g, signer=s)
        for s in signers for g in glosses for i in range(per)
    ]


def test_splits_are_signer_disjoint():
    clips = assign_splits(make())
    by_signer = {}
    for c in clips:
        by_signer.setdefault(c.signer, set()).add(c.split)
    assert all(len(s) == 1 for s in by_signer.values())
    assert {c.split for c in clips} == {"train", "val", "test"}


def test_verify_rejects_a_leaked_signer():
    clips = assign_splits(make())
    clips[0].split = "test" if clips[0].split != "test" else "train"  # leak one clip
    assert any("signer-disjoint" in p for p in verify(clips))


def test_verify_flags_thin_classes():
    problems = verify(assign_splits(make(per=4)))
    assert any("below 25 clips" in p for p in problems)


def test_verify_flags_missing_vocab_coverage():
    problems = verify(assign_splits(make()), vocab_glosses={"ME", "YOU", "WATER"})
    assert any("'WATER' has no clips" in p for p in problems)


def test_too_few_signers_cannot_be_split():
    with pytest.raises(ValueError, match="need >= 3 signers"):
        assign_splits(make(signers=("a", "b")))


def test_clean_manifest_has_no_problems():
    assert verify(assign_splits(make()), vocab_glosses={"ME", "YOU"}) == []
