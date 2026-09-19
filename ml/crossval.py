"""Leave-one-signer-out cross-validation.

    python -m ml.crossval --pack backend/vocab/isl_v2_words.json
    python -m ml.crossval --init-from ml/models/include_pretrain_long.keras

A single signer-disjoint split leaves one person as the whole test set — 75 clips, a
standard error above 5 points. That cannot tell a real improvement from a lucky run,
and comparing configurations on it is how you end up shipping noise.

This trains once per held-out signer and reports the mean and spread. It costs N times
as long and is the only honest way to say one change beat another on data this small.
Every fold reuses the same pre-trained checkpoint, so the comparison is of fine-tuning,
not of pre-training luck.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from backend.config import ROOT, settings
from backend.vocab.schema import load_pack
from ml.data.manifest import load as load_manifest
from ml.dataset import apply_velocity, augmented_batches, load_split

REPORTS = ROOT / "ml" / "reports"


def fold_arrays(clips, pack, held_out: str):
    """Everything except `held_out` trains; `held_out` is the test set."""
    labels = {g: i for i, g in enumerate(pack.labels)}
    from ml.dataset import features_for_clip
    from ml.features.extract import resample

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for c in clips:
        seq = features_for_clip(c.clip)
        if len(seq) == 0:
            continue
        v = resample(seq, settings.segment_resample_frames)
        y = labels.get(c.gloss)
        if y is None:
            continue
        if c.signer == held_out:
            te_x.append(v); te_y.append(y)
        else:
            tr_x.append(v); tr_y.append(y)
    return (np.asarray(tr_x, np.float32), np.asarray(tr_y, np.int64),
            np.asarray(te_x, np.float32), np.asarray(te_y, np.int64))


def subsample(x, y, fraction: float, seed: int):
    """Keep `fraction` of the training clips, stratified so no class disappears."""
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        keep.extend(idx[: max(1, int(round(len(idx) * fraction)))])
    keep = np.array(sorted(keep))
    return x[keep], y[keep]


def run_fold(pack, tr_x, tr_y, te_x, te_y, args, seed: int) -> float:
    import keras

    from ml.train import ARCHITECTURES, transfer_weights

    keras.utils.set_random_seed(seed)
    n_classes = len(pack.labels)
    te_xf = apply_velocity(te_x)
    model = ARCHITECTURES[args.arch](settings.segment_resample_frames, te_xf.shape[-1], n_classes)
    if args.init_from:
        transfer_weights(model, args.init_from)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    steps = max(1, len(tr_x) // args.batch_size)

    def stream():
        for xb, yb in augmented_batches(tr_x, tr_y, pack, args.batch_size, seed):
            yield xb, keras.utils.to_categorical(yb, n_classes)

    model.fit(
        stream(), steps_per_epoch=steps, epochs=args.epochs, verbose=0,
        callbacks=[keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=6, min_lr=1e-5)],
    )
    _, acc = model.evaluate(te_xf, keras.utils.to_categorical(te_y, n_classes), verbose=0)
    return float(acc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--arch", default="bilstm")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--min-clips", type=int, default=40,
                    help="skip signers with fewer clips than this; they make a meaningless fold")
    ap.add_argument("--tag", default="cv", help="name for the saved summary")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="use only this fraction of the training clips, stratified by class. "
                         "Sweeping it draws a learning curve, which is what answers "
                         "'would more data help?' with evidence instead of intuition.")
    ap.add_argument("--folds", type=int, default=None, help="use only the first N folds")
    args = ap.parse_args()

    pack = load_pack(args.pack or settings.vocab_pack)
    clips = [c for c in load_manifest() if c.gloss in {e.gloss for e in pack.entries}]
    counts = Counter(c.signer for c in clips)
    signers = sorted(s for s, n in counts.items() if n >= args.min_clips)
    if len(signers) < 3:
        raise SystemExit(f"need at least 3 usable signers, found {signers}")

    print(f"{len(clips)} clips, {len(pack.entries)} classes")
    print(f"folds: {len(signers)} — {', '.join(f'{s}({counts[s]})' for s in signers)}")
    if args.init_from:
        print(f"each fold fine-tunes from {args.init_from.name}")
    print()

    if args.folds:
        signers = signers[: args.folds]

    results = {}
    for i, held in enumerate(signers, 1):
        tr_x, tr_y, te_x, te_y = fold_arrays(clips, pack, held)
        if args.train_fraction < 1.0:
            tr_x, tr_y = subsample(tr_x, tr_y, args.train_fraction, args.seed + i)
        seen = len(set(te_y.tolist()))
        acc = run_fold(pack, tr_x, tr_y, te_x, te_y, args, args.seed + i)
        results[held] = acc
        print(f"  fold {i}/{len(signers)}  hold out {held:<10} "
              f"train {len(tr_x):>4}  test {len(te_x):>3} ({seen} classes)  acc {acc:.1%}", flush=True)

    vals = list(results.values())
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"\n  mean {mean:.1%}   sd {sd:.1%}   range {min(vals):.1%}-{max(vals):.1%}")
    print(f"  95% CI of the mean: {mean - 1.96*sd/len(vals)**0.5:.1%} - {mean + 1.96*sd/len(vals)**0.5:.1%}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"crossval_{args.tag}.json"
    out.write_text(json.dumps({
        "pack": pack.name, "arch": args.arch, "epochs": args.epochs,
        "init_from": args.init_from.name if args.init_from else None,
        "per_signer": {k: round(v, 4) for k, v in results.items()},
        "mean": round(mean, 4), "sd": round(sd, 4),
        "train_fraction": args.train_fraction,
        "train_clips_per_fold": int(len(tr_x)),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
