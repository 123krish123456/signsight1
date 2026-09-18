"""Turn a downloaded public dataset into manifest rows (PRD §7 M2, without recording).

Public datasets all have different layouts, so nothing here assumes one. You describe
where the class and the signer live, and the tool does the rest.

    # a folder-per-class video dataset (INCLUDE and most Kaggle mirrors)
    python -m ml.data.ingest videos D:/datasets/INCLUDE --source include

    # same, but the signer id is in the filename, e.g. "signer03_hello_07.mp4"
    python -m ml.data.ingest videos D:/data --source include --signer-pattern "(signer\\d+)"

    # a CSV-driven dataset (ASL Citizen ships train/val/test CSVs)
    python -m ml.data.ingest videos D:/ASL_Citizen --source asl-citizen \\
        --csv splits/train.csv --class-col Gloss --file-col "Video file" \\
        --signer-col "Participant ID" --split train

    # a folder-per-letter image dataset for the manual alphabet
    python -m ml.data.ingest images D:/datasets/ISL_alphabet --source isl-alphabet

Nothing is written until you drop --dry-run.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import re
from pathlib import Path

from backend.config import ROOT, settings
from backend.vocab.schema import VocabPack, load_pack
from ml.data.manifest import Clip, MANIFEST, load, save

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Signer id used when a dataset gives us no way to tell people apart. Deliberately
# recognisable: `manifest --check` refuses to call such classes signer-disjoint.
UNLABELLED = "unlabelled"


def gloss_key(name: str) -> str:
    """Reduce a dataset's class name to something comparable with our glosses.

    '1. Greetings/03_thank you' and 'THANK-YOU' both become 'THANKYOU'.
    """
    stem = re.sub(r"^[\d\W_]+", "", str(name)).strip()
    return re.sub(r"[^A-Z0-9]+", "", stem.upper())


def gloss_map(pack: VocabPack, include_letters: bool = True) -> dict[str, str]:
    """Every name a sign answers to → its canonical gloss.

    `include_letters=False` drops the manual alphabet. Word-level corpora contain
    single-letter class names that are not fingerspelling: INCLUDE's "40. I" is the
    pronoun, and it silently matched our letter I — two entirely different signs, which
    would have poisoned that class. Letters come from alphabet datasets; word corpora
    should be ingested with letters excluded.
    """
    return {
        gloss_key(name): e.gloss
        for e in pack.entries
        if include_letters or e.pos != "letter"
        for name in e.names
    }


def sessions_from_sequence(paths: list[Path], number_re: str, gap: int) -> dict[str, str]:
    """Group clips into recording sessions using the camera's sequential file numbers.

    Datasets recorded on a handheld camera (INCLUDE among them) name files MVI_4437,
    MVI_4438, ... and the counter runs continuously through a sitting: one person
    records word 46 four times, then word 47, and so on. A jump in the number therefore
    marks a new session, and clips within a block were shot by one person at one sitting.

    This is a HEURISTIC and it is WEAKER than a real signer split. INCLUDE has 7 signers
    but far more than 7 sessions, so one person certainly recorded several — meaning a
    session-disjoint split can still put the same person on both sides, which is exactly
    what PRD rule 4 forbids.

    It is still worth doing, because it stops consecutive takes of one person in one
    sitting from landing on both sides, and that is the leakage that inflates accuracy
    most. But a number measured this way is an upper bound on the signer-independent
    number, and must be reported as such. Prefer the dataset's own split files wherever
    it ships them — INCLUDE includes a Train_Test_Split folder.
    """
    numbered = []
    for p in paths:
        m = re.search(number_re, p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    if not numbered:
        return {}
    numbered.sort()
    mapping, session = {}, 0
    prev = numbered[0][0]
    for n, p in numbered:
        if n - prev > gap:
            session += 1
        mapping[p.as_posix()] = f"session{session:02d}"
        prev = n
    return mapping


def signers_from_passes(paths: list[Path], number_re: str, gap: int = 3) -> dict[str, str]:
    """Recover signer identity from recording passes within each class folder.

    A word's clips are not one continuous run: they come in tight blocks separated by
    large jumps, e.g. "loud" = 5177-5179, 5257-5259, 5335-5337, 9289-9291, 9368-9370,
    9448-9450, 9534-9536. Each block is one person's takes of that word in one sitting,
    and the whole corpus was recorded by each signer in turn — "loud" has exactly seven
    blocks, and INCLUDE documents exactly seven signers.

    So the Nth block of every word belongs to the same person, and clustering per class
    rather than globally is what makes the split work: every class then appears under
    every signer, instead of whole categories landing in one split because they were
    filmed in one sitting.

    Still a heuristic — it assumes the signers recorded in a consistent order — but a far
    better one than global session clustering, and it is checkable: a word should have as
    many blocks as there are signers.
    """
    per_class: dict[str, list[tuple[int, Path]]] = {}
    for p in paths:
        m = re.search(number_re, p.name)
        if m:
            per_class.setdefault(p.parent.name, []).append((int(m.group(1)), p))

    mapping: dict[str, str] = {}
    for items in per_class.values():
        items.sort()
        index = 0
        prev = items[0][0]
        for n, path in items:
            if n - prev > gap:
                index += 1
            mapping[path.as_posix()] = f"signer{index:02d}"
            prev = n
    return mapping


def _signer_from(rel: Path, pattern: str | None, fallback: str) -> str:
    if not pattern:
        return fallback
    m = re.search(pattern, rel.as_posix())
    return m.group(1) if m else fallback


def collect_files(root: Path, extensions: set[str]) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in extensions]


def from_tree(root: Path, extensions: set[str], pack: VocabPack,
              signer_pattern: str | None, default_signer: str, source: str,
              session_re: str | None = None, session_gap: int = 50,
              include_letters: bool = True,
              pass_re: str | None = None, pass_gap: int = 3) -> tuple[list[Clip], dict]:
    """Class comes from the containing folder name; signer from a regex or from
    sequential-filename session clustering."""
    mapping = gloss_map(pack, include_letters)
    files = collect_files(root, extensions)
    if pass_re:
        sessions = signers_from_passes(files, pass_re, pass_gap)
    elif session_re:
        sessions = sessions_from_sequence(files, session_re, session_gap)
    else:
        sessions = {}
    clips, unmatched = [], {}
    for path in files:
        gloss = mapping.get(gloss_key(path.parent.name))
        if gloss is None:
            unmatched[path.parent.name] = unmatched.get(path.parent.name, 0) + 1
            continue
        rel = path.relative_to(root)
        signer = sessions.get(path.as_posix()) or _signer_from(rel, signer_pattern, default_signer)
        clips.append(Clip(clip=path.as_posix(), gloss=gloss, signer=signer, source=source))
    return clips, unmatched


def from_csv(root: Path, csv_path: Path, pack: VocabPack, class_col: str, file_col: str,
             signer_col: str | None, split: str, source: str) -> tuple[list[Clip], dict]:
    mapping = gloss_map(pack)
    clips, unmatched = [], {}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csvmod.DictReader(f)
        missing = {c for c in (class_col, file_col) if c not in (reader.fieldnames or [])}
        if missing:
            raise SystemExit(f"{csv_path.name} has no column(s) {sorted(missing)}; "
                             f"it has {reader.fieldnames}")
        for row in reader:
            gloss = mapping.get(gloss_key(row[class_col]))
            if gloss is None:
                unmatched[row[class_col]] = unmatched.get(row[class_col], 0) + 1
                continue
            clips.append(Clip(
                clip=(root / row[file_col]).as_posix(),
                gloss=gloss,
                signer=(row.get(signer_col) or UNLABELLED) if signer_col else UNLABELLED,
                split=split,
                source=source,
            ))
    return clips, unmatched


MIN_USABLE_HAND_RATE = 0.30


def probe(clips: list[Clip], n: int = 6) -> bool:
    """Run the real extractor over a sample and report how often hands are found.

    Worth thirty seconds before ingesting anything. Signs are distinguished mostly by
    hand shape, so a dataset whose hands MediaPipe cannot see is useless no matter how
    many clips it has — and downscaled or long-shot footage fails exactly this way while
    still detecting the body perfectly, which makes it look fine until you train on it.
    """
    import random

    import numpy as np

    from ml.features.extract import HAND_DIM, POSE_DIM, extract_image, extract_video

    sample = random.Random(0).sample(clips, min(n, len(clips)))
    print(f"\nprobing {len(sample)} files for detectable landmarks...")
    print(f"  {'file':<28} {'frames':>6} {'pose':>6} {'hands':>6}")
    rates = []
    for c in sample:
        path = Path(c.clip)
        seq = (extract_image(str(path)) if path.suffix.lower() in IMAGE_EXT
               else extract_video(str(path), every_nth=1))
        if len(seq) == 0:
            print(f"  {path.name:<28} {'--':>6}  unreadable")
            rates.append(0.0)
            continue
        pose = np.any(seq[:, :POSE_DIM] != 0, axis=1).mean()
        hands = np.any(seq[:, POSE_DIM : POSE_DIM + 2 * HAND_DIM] != 0, axis=1).mean()
        rates.append(hands)
        print(f"  {path.name:<28} {len(seq):>6} {pose:>5.0%} {hands:>6.0%}")

    mean = sum(rates) / max(len(rates), 1)
    print(f"\n  hands visible in {mean:.0%} of frames")
    if mean < MIN_USABLE_HAND_RATE:
        print(f"  UNUSABLE: below {MIN_USABLE_HAND_RATE:.0%}. Hand shape is what separates one sign")
        print("  from another, so these clips carry almost no usable signal. Usually means the")
        print("  video is downscaled or the signer is too far from the camera. Find the")
        print("  original full-resolution release of this dataset.")
        return False
    print("  OK — hands are detectable often enough to train on.")
    return True


def report(clips: list[Clip], unmatched: dict, pack: VocabPack, kind: str) -> None:
    from collections import Counter

    by_gloss = Counter(c.gloss for c in clips)
    by_signer = Counter(c.signer for c in clips)
    wanted = [e.gloss for e in pack.entries]
    covered = [g for g in wanted if by_gloss[g]]

    print(f"\nmatched {len(clips)} {kind} across {len(covered)}/{len(wanted)} vocabulary classes")
    print(f"signers: {dict(by_signer) if len(by_signer) <= 8 else f'{len(by_signer)} distinct'}")

    thin = [(g, by_gloss[g]) for g in covered if by_gloss[g] < 25]
    if thin:
        print(f"\n{len(thin)} matched classes have fewer than 25 samples:")
        for g, n in sorted(thin, key=lambda kv: kv[1])[:10]:
            print(f"  {g:<14} {n}")
    missing = [g for g in wanted if not by_gloss[g]]
    if missing:
        print(f"\n{len(missing)} classes not covered at all:")
        print("  " + ", ".join(missing[:18]) + (" ..." if len(missing) > 18 else ""))
    if unmatched:
        top = sorted(unmatched.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print(f"\n{len(unmatched)} dataset classes matched nothing in the vocab pack, e.g.:")
        for name, n in top:
            print(f"  {name!r} ({n})")
        print("  (rename them in the pack, or ignore — they are simply not in our 50)")
    if UNLABELLED in by_signer:
        print(f"\nNOTE: {by_signer[UNLABELLED]} samples have no signer identity. Those classes "
              "cannot be split signer-disjoint, and `manifest --check` will say so.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["videos", "images"])
    ap.add_argument("root", type=Path)
    ap.add_argument("--source", required=True, help="tag recorded in the manifest, e.g. include / asl-citizen")
    ap.add_argument("--signer-pattern", default=None,
                    help="regex over the relative path; group 1 is the signer id")
    ap.add_argument("--session-re", default=None, metavar="REGEX",
                    help=r"derive sessions from sequential filenames, e.g. 'MVI_(\d+)'")
    ap.add_argument("--session-gap", type=int, default=50,
                    help="a jump larger than this in the sequence starts a new session")
    ap.add_argument("--pass-re", default=None, metavar="REGEX",
                    help=r"recover signers from per-class recording passes, e.g. 'MVI_(\d+)'. "
                         "Preferred over --session-re: it keeps every class present in every split.")
    ap.add_argument("--pass-gap", type=int, default=3,
                    help="a jump larger than this within one class starts a new pass")
    ap.add_argument("--no-letters", action="store_true",
                    help="ignore manual-alphabet classes; use for word corpora, whose "
                         "single-letter class names are words, not fingerspelling")
    ap.add_argument("--default-signer", default=None,
                    help=f"used when no signer can be determined (default: {UNLABELLED!r})")
    ap.add_argument("--csv", type=Path, default=None, help="CSV listing the clips instead of walking the tree")
    ap.add_argument("--class-col", default="Gloss")
    ap.add_argument("--file-col", default="Video file")
    ap.add_argument("--signer-col", default=None)
    ap.add_argument("--split", default="", choices=["", "train", "val", "test"],
                    help="pre-assign a split (use when the dataset ships its own signer-disjoint splits)")
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="report coverage without writing")
    ap.add_argument("--probe", type=int, nargs="?", const=6, default=None, metavar="N",
                    help="extract landmarks from N sampled files and report detection rates")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"not a directory: {args.root}")
    pack = load_pack(args.pack or settings.vocab_pack)
    default_signer = args.default_signer or UNLABELLED

    if args.csv:
        clips, unmatched = from_csv(args.root, args.csv, pack, args.class_col, args.file_col,
                                    args.signer_col, args.split, args.source)
        kind = "rows"
    else:
        ext = VIDEO_EXT if args.kind == "videos" else IMAGE_EXT
        clips, unmatched = from_tree(args.root, ext, pack, args.signer_pattern,
                                     default_signer, args.source,
                                     args.session_re, args.session_gap,
                                     include_letters=not args.no_letters,
                                     pass_re=args.pass_re, pass_gap=args.pass_gap)
        for c in clips:
            c.split = args.split
        kind = args.kind

    if not clips:
        print(f"no {args.kind} matched the vocabulary in {args.root}")
        if unmatched:
            report(clips, unmatched, pack, kind)
        return 1

    report(clips, unmatched, pack, kind)

    if args.probe:
        probe(clips, args.probe)

    if args.dry_run:
        print("\ndry run — nothing written. Drop --dry-run to add these to the manifest.")
        return 0

    existing = load()
    known = {c.clip for c in existing}
    added = [c for c in clips if c.clip not in known]
    save(existing + added)
    print(f"\nadded {len(added)} rows to {MANIFEST.relative_to(ROOT)} "
          f"({len(clips) - len(added)} already present)")
    print("next: python -m ml.data.manifest --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
