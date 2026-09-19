"""Guided capture tool (PRD §7 M2).

Plays the reference sign beside your camera so you can copy it, records a 3 s clip,
writes it to the manifest, and always offers the class you have the *fewest* clips of —
so coverage stays flat instead of 25 clips of HELLO and none of WHEN.

    python -m ml.data.record --signer eashan --target 10

The reference clip comes from the vocabulary pack's `reference_video`. Copy what it
shows: inventing a gesture and labelling it with a real word is worse than no data.

Keys:  SPACE record · N skip to next class · U undo last · Q quit
Needs the `ml` extra:  pip install -e ".[ml]"
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

from backend.config import ROOT, settings
from backend.vocab.schema import load_pack
from ml.data.manifest import Clip, load, save

CLIPS_DIR = ROOT / "ml" / "data" / "clips"
CLIP_SECONDS = 3.0
COUNTDOWN_SECONDS = 1.5
FPS = 30  # capture at 30, features decimate to 15 (PRD §4.1)


def next_gloss(glosses: list[str], counts: Counter, current: str | None) -> str:
    """Least-recorded class, ties broken by vocab order; never repeat `current`."""
    pool = [g for g in glosses if g != current] or glosses
    return min(pool, key=lambda g: (counts[g], glosses.index(g)))


class Reference:
    """Loops the reference clip for the current sign, beside the live camera.

    None of us knows ISL. The corpus we train on is also the only teacher available, so
    the recorder plays the sign you are about to copy: watch it, mirror it, record it.
    Without this you would be inventing gestures and labelling them with real words,
    which is worse than having no data at all.
    """

    def __init__(self) -> None:
        self.cap = None
        self.path = None

    def load(self, path: str | None) -> None:
        if path == self.path:
            return
        if self.cap is not None:
            self.cap.release()
        self.path = path
        full = ROOT / path if path and not Path(path).is_absolute() else path
        self.cap = cv2.VideoCapture(str(full)) if path and Path(str(full)).exists() else None

    def frame(self, height: int):
        """Next frame of the reference, looping. None if there is no reference."""
        if self.cap is None:
            return None
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if not ok:
            return None
        h, w = f.shape[:2]
        return cv2.resize(f, (int(w * height / h), height))

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


def side_by_side(live, ref):
    """Live camera on the left, reference sign on the right."""
    if ref is None:
        return live
    h = live.shape[0]
    if ref.shape[0] != h:
        ref = cv2.resize(ref, (int(ref.shape[1] * h / ref.shape[0]), h))
    strip = cv2.copyMakeBorder(ref, 0, 0, 6, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return cv2.hconcat([live, strip])


def draw(frame, lines: list[tuple[str, tuple[int, int, int], float]]) -> None:
    y = 40
    for text, colour, scale in lines:
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
        y += int(38 * scale + 14)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True, help="signer id — splits are BY SIGNER, so use a stable name")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pack", type=Path, default=settings.vocab_pack)
    ap.add_argument("--only", nargs="*", help="restrict to these glosses")
    ap.add_argument("--target", type=int, default=10,
                    help="clips per sign to aim for (default 10; the value of a new signer "
                         "is variety, not volume)")
    args = ap.parse_args()

    pack = load_pack(args.pack)
    glosses = [e.gloss for e in pack.entries]
    reference_of = {e.gloss: e.reference_video for e in pack.entries}
    if args.only:
        unknown = set(args.only) - set(glosses)
        if unknown:
            print(f"not in the pack: {sorted(unknown)}")
            return 1
        glosses = [g for g in glosses if g in set(args.only)]

    clips = load()
    counts = Counter(c.gloss for c in clips if c.signer == args.signer)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"cannot open camera {args.camera}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    reference = Reference()
    gloss = next_gloss(glosses, counts, None)
    total = sum(counts.values())
    print(f"signer={args.signer} · {len(glosses)} classes · {total} existing clips\nSPACE record · N next · U undo · Q quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            done = counts[gloss]
            reference.load(reference_of.get(gloss))
            draw(frame, [
                (f"SIGN: {gloss}", (80, 220, 255), 1.1),
                (f"{done}/{args.target} recorded · {sum(counts.values())} total", (200, 200, 200), 0.6),
                ("copy the clip on the right", (120, 200, 120), 0.55),
                ("SPACE record · N next · U undo · Q quit", (160, 160, 160), 0.5),
            ])
            cv2.imshow("SignSight capture", side_by_side(frame, reference.frame(frame.shape[0])))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                gloss = next_gloss(glosses, counts, gloss)
            elif key == ord("u") and clips and clips[-1].signer == args.signer:
                last = clips.pop()
                Path(last.clip).unlink(missing_ok=True)
                counts[last.gloss] -= 1
                save(clips)
                gloss = last.gloss
                print(f"undid {last.clip}")
            elif key == ord(" "):
                _countdown(cap, gloss)
                path = _record(cap, fourcc, args.signer, gloss, counts[gloss])
                clips.append(Clip(clip=str(path.relative_to(ROOT)), gloss=gloss, signer=args.signer))
                save(clips)
                counts[gloss] += 1
                print(f"saved {path.name}  ({counts[gloss]}/{args.target} {gloss})")
                gloss = next_gloss(glosses, counts, gloss)
    finally:
        cap.release()
        reference.release()
        cv2.destroyAllWindows()

    print(f"\n{sum(counts.values())} clips for {args.signer}. Next: python -m ml.data.manifest --assign --check")
    return 0


def _countdown(cap, gloss: str) -> None:
    end = time.monotonic() + COUNTDOWN_SECONDS
    while (left := end - time.monotonic()) > 0:
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        draw(frame, [(f"{gloss} in {left:.1f}", (80, 220, 255), 1.4)])
        cv2.imshow("SignSight capture", frame)
        cv2.waitKey(1)


def _record(cap, fourcc, signer: str, gloss: str, index: int) -> Path:
    out_dir = CLIPS_DIR / signer / gloss
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{signer}_{gloss}_{index:03d}.mp4"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h))
    end = time.monotonic() + CLIP_SECONDS
    try:
        while (left := end - time.monotonic()) > 0:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)  # unflipped: the model sees the true handedness
            shown = cv2.flip(frame.copy(), 1)
            draw(shown, [("● REC", (60, 60, 255), 1.2), (f"{left:.1f}s", (255, 255, 255), 0.8)])
            cv2.imshow("SignSight capture", shown)
            cv2.waitKey(1)
    finally:
        writer.release()
    return path


if __name__ == "__main__":
    raise SystemExit(main())
