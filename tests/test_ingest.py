"""Public-dataset ingest and mixed-provenance splits.

Once data comes from public sets instead of our own recordings, two things can go wrong
quietly: classes with no signer identity get silently treated as signer-disjoint, or
they all land in one split and vanish from training. Both are tested here.
"""

import numpy as np
import pytest

from ml.data.ingest import gloss_key, gloss_map
from ml.data.manifest import UNLABELLED, Clip, assign_splits, disclosures, verify
from backend.config import ROOT
from backend.vocab.schema import load_pack

PACK = load_pack(ROOT / "backend" / "vocab" / "isl_v1.json")


# ---------------------------------------------------------------- name matching

@pytest.mark.parametrize("raw,expected", [
    ("THANK-YOU", "THANKYOU"),
    ("thank you", "THANKYOU"),
    ("3. thank_you", "THANKYOU"),
    ("01_Hello", "HELLO"),
    ("  hello  ", "HELLO"),
    ("A", "A"),
])
def test_gloss_key_normalises_dataset_names(raw, expected):
    assert gloss_key(raw) == expected


def test_aliases_map_dataset_names_onto_our_glosses():
    """ISL corpora label the greeting "namaste", not "hello". Adding a dataset must be
    a JSON edit, never a code change (PRD §4.6)."""
    from backend.vocab.schema import VocabPack

    pack = VocabPack.model_validate({
        "name": "t", "language": "ISL", "version": "1",
        "entries": [{"gloss": "HELLO", "pos": "interjection", "aliases": ["namaste", "namaskar"]}],
    })
    m = gloss_map(pack)
    assert m[gloss_key("namaste")] == "HELLO"
    assert m[gloss_key("2. Namaskar")] == "HELLO"
    assert m[gloss_key("HELLO")] == "HELLO", "the canonical gloss still resolves"


def test_entries_without_aliases_still_work():
    assert gloss_map(PACK)[gloss_key("HELLO")] == "HELLO"


def test_gloss_map_matches_real_vocabulary():
    m = gloss_map(PACK)
    assert m[gloss_key("thank you")] == "THANK-YOU"
    assert m[gloss_key("2. Hello")] == "HELLO"
    assert gloss_key("pizza") not in m


# ---------------------------------------------------------------- splits

def videos(signers=("s1", "s2", "s3"), glosses=("ME", "YOU"), per=25):
    return [Clip(clip=f"{s}/{g}/{i}.mp4", gloss=g, signer=s, source="include")
            for s in signers for g in glosses for i in range(per)]


def images(glosses=("A", "B"), per=30):
    return [Clip(clip=f"alphabet/{g}/{i}.jpg", gloss=g, signer=UNLABELLED, source="isl-alphabet")
            for g in glosses for i in range(per)]


def test_anonymous_samples_reach_every_split():
    """The bug this guards: one pseudo-signer means one split, so those classes are
    either never trained on or never tested."""
    clips = assign_splits(videos() + images())
    for gloss in ("A", "B"):
        splits = {c.split for c in clips if c.gloss == gloss}
        assert splits == {"train", "val", "test"}, f"{gloss} only reached {splits}"


def test_identified_signers_stay_signer_disjoint_alongside_anonymous_ones():
    clips = assign_splits(videos() + images())
    for signer in ("s1", "s2", "s3"):
        assert len({c.split for c in clips if c.signer == signer}) == 1


def test_most_anonymous_samples_go_to_training():
    clips = assign_splits(images(per=30))
    train = sum(c.split == "train" for c in clips)
    assert train > sum(c.split != "train" for c in clips)


def test_verify_does_not_call_anonymous_classes_signer_disjoint():
    problems = verify(assign_splits(videos() + images()))
    assert not any("multiple splits" in p and UNLABELLED in p for p in problems)


def test_anonymous_signers_do_not_count_toward_the_signer_minimum():
    clips = assign_splits(videos(signers=("s1", "s2", "s3")) + images())
    problems = verify(clips)
    assert not any("identified signer" in p for p in problems)

    # two real signers plus a big anonymous pool is still only two real signers
    only_two = videos(signers=("s1", "s2")) + images()
    for c in only_two:
        c.split = "train"
    assert any("identified signer" in p for p in verify(only_two))


# ---------------------------------------------------------------- disclosures

def test_disclosures_flag_the_random_split():
    notes = disclosures(assign_splits(videos() + images()))
    assert any("NOT a signer-disjoint" in n for n in notes)


def test_disclosures_flag_still_images():
    notes = disclosures(images())
    assert any("velocity feature is zero" in n for n in notes)


def test_disclosures_flag_the_absence_of_self_recorded_clips():
    notes = disclosures(videos())
    assert any("No self-recorded clips" in n for n in notes)


def test_a_self_recorded_manifest_has_nothing_to_disclose():
    own = [Clip(clip=f"{s}/{g}/{i}.mp4", gloss=g, signer=s, source="self")
           for s in ("a", "b", "c") for g in ("ME",) for i in range(25)]
    assert disclosures(assign_splits(own)) == []
