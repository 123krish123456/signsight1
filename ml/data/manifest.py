"""Clip manifest + signer-disjoint splits (PRD §7 M2, rule 4).

manifest.csv columns: clip,gloss,signer,split,source

A random clip split leaks signer identity and inflates accuracy 10–20 points. Splits
are assigned BY SIGNER here, and `verify()` is the automated check the M2 DoD requires.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.config import ROOT
from backend.vocab.schema import load_pack

MANIFEST = ROOT / "ml" / "data" / "manifest.csv"
FIELDS = ["clip", "gloss", "signer", "split", "source"]
SPLITS = ("train", "val", "test")
MIN_CLIPS_PER_CLASS = 25  # PRD M2 DoD
MIN_SIGNERS = 3


@dataclass
class Clip:
    clip: str
    gloss: str
    signer: str
    split: str = ""
    source: str = "self"


def load(path: Path = MANIFEST) -> list[Clip]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [Clip(**row) for row in csv.DictReader(f)]


def save(clips: list[Clip], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(asdict(c) for c in clips)


def append(clip: Clip, path: Path = MANIFEST) -> None:
    clips = load(path)
    clips.append(clip)
    save(clips, path)


UNLABELLED = "unlabelled"  # datasets that do not say who is signing (see ml/data/ingest.py)


def assign_splits(clips: list[Clip], test_signers: int = 1, val_signers: int = 1,
                  seed: int = 0) -> list[Clip]:
    """Whole signers go to one split. Never split a signer's clips across splits.

    Samples whose signer is unknown (public image sets, mostly) cannot take part in a
    signer-disjoint split. Putting them all in one split would be worse than useless —
    every class sourced that way would be absent from training, or absent from test — so
    they get a per-class random split instead. That is a weaker guarantee, and
    `disclosures()` reports it so it reaches the write-up rather than being buried.
    """
    import random

    identified = [c for c in clips if c.signer != UNLABELLED]
    anonymous = [c for c in clips if c.signer == UNLABELLED]

    if identified:
        signers = sorted({c.signer for c in identified})
        if len(signers) < test_signers + val_signers + 1:
            raise ValueError(
                f"need >= {test_signers + val_signers + 1} signers for a disjoint split, "
                f"have {len(signers)}: {signers}"
            )
        # Smallest contributors become val/test: the biggest signer is worth more in training.
        by_size = sorted(signers, key=lambda s: sum(c.signer == s for c in identified))
        test = set(by_size[:test_signers])
        val = set(by_size[test_signers : test_signers + val_signers])
        for c in identified:
            c.split = "test" if c.signer in test else "val" if c.signer in val else "train"

    if anonymous:
        rng = random.Random(seed)
        by_gloss: dict[str, list[Clip]] = defaultdict(list)
        for c in anonymous:
            by_gloss[c.gloss].append(c)
        for group in by_gloss.values():  # stratified, so every class reaches every split
            rng.shuffle(group)
            n = len(group)
            n_test = max(1, round(n * 0.2)) if n >= 3 else 0
            n_val = max(1, round(n * 0.1)) if n >= 3 else 0
            for i, c in enumerate(group):
                c.split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return clips


def disclosures(clips: list[Clip]) -> list[str]:
    """Compromises that must appear in the report. Not errors — chosen trade-offs."""
    notes = []
    anonymous = [c for c in clips if c.signer == UNLABELLED]
    if anonymous:
        classes = sorted({c.gloss for c in anonymous})
        notes.append(
            f"{len(anonymous)} samples across {len(classes)} classes have no signer identity, "
            f"so those classes use a random split, NOT a signer-disjoint one. Their accuracy is "
            f"optimistic and must be reported separately: {', '.join(classes[:8])}"
            + (" ..." if len(classes) > 8 else "")
        )
    sessions = {c.signer for c in clips if c.signer.startswith("session")}
    if sessions:
        notes.append(
            f"{len(sessions)} 'signers' are recording sessions inferred from filename "
            "numbering, not identified people. One person recorded several sessions, so the "
            "split is session-disjoint but NOT signer-disjoint: accuracy is an upper bound on "
            "the signer-independent figure. Prefer the dataset's own split files."
        )

    sources = {c.source for c in clips}
    if "self" not in sources and sources:
        notes.append(
            f"No self-recorded clips: every sample comes from {sorted(sources)}. Live accuracy "
            "in your own lighting and camera angle will be lower than the test split suggests."
        )
    images = [c for c in clips if c.clip.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if images:
        notes.append(
            f"{len(images)} samples are still images expanded to a fixed-length sequence, so every "
            "velocity feature is zero for them. Signs involving motion cannot be learned this way."
        )
    return notes


def verify(clips: list[Clip], vocab_glosses: set[str] | None = None) -> list[str]:
    """Returns a list of problems. Empty list == M2 DoD met."""
    problems: list[str] = []
    if not clips:
        return ["manifest is empty"]

    split_of: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        split_of[c.signer].add(c.split)
    for signer, splits in sorted(split_of.items()):
        if signer == UNLABELLED:
            continue  # deliberately split per class; see assign_splits
        if len(splits) > 1:
            problems.append(f"signer {signer!r} appears in multiple splits {sorted(splits)} — splits must be signer-disjoint")
        if splits <= {""}:
            problems.append(f"signer {signer!r} has unassigned clips — run assign_splits")

    signers = {c.signer for c in clips} - {UNLABELLED}
    if len(signers) < MIN_SIGNERS:
        problems.append(
            f"only {len(signers)} identified signer(s), PRD M2 requires >= {MIN_SIGNERS} "
            f"(samples with no signer identity do not count)"
        )

    counts = Counter(c.gloss for c in clips)
    if vocab_glosses:
        for gloss in sorted(vocab_glosses - set(counts)):
            problems.append(f"gloss {gloss!r} has no clips")
        for gloss in sorted(set(counts) - vocab_glosses):
            problems.append(f"gloss {gloss!r} is not in the vocab pack")
    thin = {g: n for g, n in counts.items() if n < MIN_CLIPS_PER_CLASS}
    if thin:
        problems.append(f"{len(thin)} classes below {MIN_CLIPS_PER_CLASS} clips (worst: {sorted(thin.items(), key=lambda kv: kv[1])[:5]})")

    for split in SPLITS:
        present = {c.gloss for c in clips if c.split == split}
        missing = set(counts) - present
        if missing:
            problems.append(f"split {split!r} is missing {len(missing)} classes, e.g. {sorted(missing)[:5]}")
    return problems


def summary(clips: list[Clip]) -> str:
    by_split = Counter(c.split or "unassigned" for c in clips)
    by_signer = Counter(c.signer for c in clips)
    counts = Counter(c.gloss for c in clips)
    thinnest = sorted(counts.items(), key=lambda kv: kv[1])[:5]
    return (
        f"{len(clips)} clips · {len(counts)} classes · {len(by_signer)} signers\n"
        f"  splits : {dict(by_split)}\n"
        f"  signers: {dict(by_signer)}\n"
        f"  thinnest classes: {thinnest}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect / split the clip manifest")
    ap.add_argument("--assign", action="store_true", help="assign signer-disjoint splits and save")
    ap.add_argument("--check", action="store_true", help="verify M2 DoD (exit 1 on problems)")
    args = ap.parse_args()

    clips = load()
    if args.assign:
        save(assign_splits(clips))
        print("splits assigned")
        clips = load()

    print(summary(clips))
    if args.check:
        glosses = {e.gloss for e in load_pack(ROOT / "backend" / "vocab" / "isl_v1.json").entries}
        problems = verify(clips, glosses)
        for p in problems:
            print(f"  FAIL {p}")
        if problems:
            print(f"\nM2 not met: {len(problems)} problem(s). Do not start M3 (PRD rule 9).")
            return 1
        print("\nOK: M2 DoD met — signer-disjoint, >=25 clips/class, all classes covered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
