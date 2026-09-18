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


def assign_splits(clips: list[Clip], test_signers: int = 1, val_signers: int = 1) -> list[Clip]:
    """Whole signers go to one split. Never split a signer's clips across splits."""
    signers = sorted({c.signer for c in clips})
    if len(signers) < test_signers + val_signers + 1:
        raise ValueError(
            f"need >= {test_signers + val_signers + 1} signers for a disjoint split, have {len(signers)}: {signers}"
        )
    # Smallest contributors become val/test: the biggest signer is worth more in training.
    by_size = sorted(signers, key=lambda s: sum(c.signer == s for c in clips))
    test = set(by_size[:test_signers])
    val = set(by_size[test_signers : test_signers + val_signers])
    for c in clips:
        c.split = "test" if c.signer in test else "val" if c.signer in val else "train"
    return clips


def verify(clips: list[Clip], vocab_glosses: set[str] | None = None) -> list[str]:
    """Returns a list of problems. Empty list == M2 DoD met."""
    problems: list[str] = []
    if not clips:
        return ["manifest is empty"]

    split_of: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        split_of[c.signer].add(c.split)
    for signer, splits in sorted(split_of.items()):
        if len(splits) > 1:
            problems.append(f"signer {signer!r} appears in multiple splits {sorted(splits)} — splits must be signer-disjoint")
        if splits <= {""}:
            problems.append(f"signer {signer!r} has unassigned clips — run assign_splits")

    signers = {c.signer for c in clips}
    if len(signers) < MIN_SIGNERS:
        problems.append(f"only {len(signers)} signer(s), PRD M2 requires >= {MIN_SIGNERS}")

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
