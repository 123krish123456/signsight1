"""Evaluation and failure analysis (PRD §7 M7).

    python -m ml.evaluate                          # test split, newest ONNX model
    python -m ml.evaluate --split val
    python -m ml.evaluate --ablate face            # drop the face-lite block
    python -m ml.evaluate --synthetic              # exercise the report path with no clips

Produces top-1 accuracy, per-class recall, a confusion matrix, the per-signer breakdown,
the sign pairs that confuse each other, and the inference latency distribution — the
material §7 M7 asks for. Runs on ONNX Runtime, the same engine the backend uses, so the
numbers describe what actually ships.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from backend.config import ROOT, settings
from backend.vocab.schema import load_pack
from ml.dataset import apply_velocity, load_split
from ml.features.extract import FACE_DIM, FEATURE_DIM

MODELS = ROOT / "ml" / "models"
REPORTS = ROOT / "ml" / "reports"
FACE_SLICE = slice(FEATURE_DIM - FACE_DIM, FEATURE_DIM)


def newest_onnx() -> Path:
    found = sorted(MODELS.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise SystemExit(f"no .onnx model in {MODELS} — run `python -m ml.export_onnx` first")
    return found[0]


def predict(onnx_path: Path, x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Per-sample softmax plus per-sample wall time in ms."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    probs, times = [], []
    for sample in x:
        batch = sample[None].astype(np.float32)
        t0 = time.perf_counter()
        out = session.run(None, {name: batch})[0]
        times.append((time.perf_counter() - t0) * 1000)
        probs.append(out[0])
    return np.asarray(probs), times


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)


def confused_pairs(cm: np.ndarray, labels: list[str], top: int = 12):
    """Off-diagonal hot spots — the 'which signs confuse, and why' table."""
    out = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j]:
                support = cm[i].sum()
                out.append((labels[i], labels[j], int(cm[i, j]), cm[i, j] / max(support, 1)))
    return sorted(out, key=lambda r: r[2], reverse=True)[:top]


def save_confusion_png(cm: np.ndarray, labels: list[str], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(8, len(labels) * 0.22)
    fig, ax = plt.subplots(figsize=(size, size), dpi=140)
    ax.imshow(norm, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("SignSight — confusion matrix (row-normalised)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def ablate(x: np.ndarray, which: str) -> np.ndarray:
    """Zero a feature block to measure what it contributes (PRD M7 ablations)."""
    if which == "none":
        return x
    if which == "face":
        out = x.copy()
        out[:, :, FACE_SLICE] = 0.0
        return out
    raise SystemExit(f"unknown ablation {which!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ablate", default="none", choices=["none", "face"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--top-confusions", type=int, default=12)
    args = ap.parse_args()

    pack = load_pack(settings.vocab_pack)
    labels = pack.labels
    model_path = args.model or newest_onnx()

    if args.synthetic:
        from ml.train import synthetic_split
        print("!! SYNTHETIC DATA — exercises the report path; the numbers mean nothing\n")
        x, y = synthetic_split(6, len(labels), settings.segment_resample_frames, 44)
        signers = ["synthetic"] * len(y)
    else:
        x, y, signers = load_split(args.split, pack)
        if len(x) == 0:
            raise SystemExit(
                f"the {args.split!r} split is empty — record clips, then\n"
                "  python -m ml.data.manifest --assign --check"
            )

    x = apply_velocity(ablate(x, args.ablate))
    probs, times = predict(model_path, x)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    cm = confusion(y, pred, len(labels))
    recall = per_class_recall(cm)
    top1 = float((pred == y).mean())

    REPORTS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}" + (f"_ablate-{args.ablate}" if args.ablate != "none" else "")

    print(f"model      {model_path.name}")
    print(f"split      {args.split}  ({len(y)} samples, {len(set(signers))} signers)")
    if args.ablate != "none":
        print(f"ablation   {args.ablate} block zeroed")
    print(f"\ntop-1 accuracy   {top1:.1%}")
    print(f"mean confidence  {conf.mean():.3f}")
    print(f"inference        p50 {np.percentile(times, 50):.1f} ms · p95 {np.percentile(times, 95):.1f} ms")

    # PRD §4.5: gating only emits above CONFIDENCE_THRESH, so report what survives it.
    gated = conf >= settings.confidence_thresh
    if gated.any():
        print(f"\nat the {settings.confidence_thresh} confidence gate:")
        print(f"  emitted          {gated.mean():.1%} of segments")
        print(f"  accuracy of those {(pred[gated] == y[gated]).mean():.1%}")
    else:
        print(f"\nnothing clears the {settings.confidence_thresh} confidence gate")

    print("\nper-signer accuracy:")
    for s in sorted(set(signers)):
        idx = [i for i, v in enumerate(signers) if v == s]
        print(f"  {s:<16} {(pred[idx] == y[idx]).mean():.1%}  ({len(idx)} clips)")

    weakest = [(labels[i], recall[i], int(cm[i].sum())) for i in np.argsort(np.nan_to_num(recall, nan=2.0))[:10]]
    print("\nweakest classes:")
    for name, r, sup in weakest:
        print(f"  {name:<14} recall {0.0 if np.isnan(r) else r:.0%}  ({sup} clips)")

    pairs = confused_pairs(cm, labels, args.top_confusions)
    if pairs:
        print("\nmost confused pairs (true -> predicted):")
        for t, p, n, frac in pairs:
            print(f"  {t:<14} -> {p:<14} {n:>3}  ({frac:.0%} of {t})")

    with (REPORTS / f"confusion_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for name, row in zip(labels, cm):
            w.writerow([name, *row.tolist()])

    png = REPORTS / f"confusion_{tag}.png"
    drew = save_confusion_png(cm, labels, png)

    summary = {
        "model": model_path.name,
        "split": args.split,
        "ablation": args.ablate,
        "synthetic": args.synthetic,
        "samples": int(len(y)),
        "top1": round(top1, 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "latency_ms": {"p50": round(float(np.percentile(times, 50)), 2),
                       "p95": round(float(np.percentile(times, 95)), 2)},
        "per_signer": {s: round(float((pred[[i for i, v in enumerate(signers) if v == s]]
                                       == y[[i for i, v in enumerate(signers) if v == s]]).mean()), 4)
                       for s in sorted(set(signers))},
        "per_class_recall": {labels[i]: (None if np.isnan(recall[i]) else round(float(recall[i]), 4))
                             for i in range(len(labels))},
        "top_confusions": [{"true": t, "predicted": p, "count": n} for t, p, n, _ in pairs],
    }
    (REPORTS / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nwrote {REPORTS / f'confusion_{tag}.csv'}")
    if drew:
        print(f"wrote {png}")
    print(f"wrote {REPORTS / f'summary_{tag}.json'}")

    if not args.synthetic and args.ablate == "none" and args.split == "test":
        print(f"\nReport {top1:.1%} as the headline number, whatever it is. Do not tune on this split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
