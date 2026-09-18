"""Keras → ONNX, with a numerical equivalence check (PRD §7 M3 DoD).

    python -m ml.export_onnx                                  # newest .keras in ml/models
    python -m ml.export_onnx --model ml/models/signsight_bilstm.keras

The backend runs ONNX Runtime, never TensorFlow, so this conversion sits on the path to
production. An export that silently changes the numbers would show up as a model that
scores well offline and behaves differently live — so the outputs are compared here and
the script fails if they drift past 1e-4.

Keras 3 is exported via SavedModel; tf2onnx's `from_keras` path targets Keras 2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from backend.config import ROOT

MODELS = ROOT / "ml" / "models"
TOLERANCE = 1e-4
OPSET = 15


def newest_keras_model() -> Path:
    found = sorted(MODELS.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise SystemExit(f"no .keras model in {MODELS} — run `python -m ml.train` first")
    return found[0]


def export(model_path: Path, out: Path, opset: int = OPSET) -> Path:
    import keras

    model = keras.saving.load_model(model_path)
    tmp = Path(tempfile.mkdtemp(prefix="signsight_sm_"))
    saved_model_dir = tmp / "sm"
    try:
        model.export(saved_model_dir)  # Keras 3 → TF SavedModel
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, "-m", "tf2onnx.convert",
             "--saved-model", str(saved_model_dir),
             "--output", str(out), "--opset", str(opset)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out.exists():
            raise SystemExit(f"tf2onnx failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def verify(model_path: Path, onnx_path: Path, n_samples: int = 8) -> float:
    """Run both on identical random input; return the max absolute difference."""
    import keras
    import onnxruntime as ort

    model = keras.saving.load_model(model_path)
    seq_len, n_features = model.input_shape[1], model.input_shape[2]
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype(np.float32)

    keras_out = model.predict(x, verbose=0)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: x})[0]

    assert onnx_out.shape == keras_out.shape, f"shape drift {onnx_out.shape} vs {keras_out.shape}"
    return float(np.abs(onnx_out - keras_out).max())


def benchmark(onnx_path: Path, runs: int = 50) -> float:
    """Median single-segment inference time in ms. PRD M3 DoD: < 25 ms on CPU."""
    import time

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    x = np.random.default_rng(0).normal(0, 1, shape).astype(np.float32)
    name = session.get_inputs()[0].name

    session.run(None, {name: x})  # warm up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=OPSET)
    args = ap.parse_args()

    model_path = args.model or newest_keras_model()
    out = args.out or model_path.with_suffix(".onnx")
    print(f"exporting {model_path.name} -> {out.name}")

    export(model_path, out, args.opset)
    drift = verify(model_path, out)
    ms = benchmark(out)
    size_mb = out.stat().st_size / 1e6

    print(f"\n  size          {size_mb:.1f} MB")
    print(f"  max drift     {drift:.2e}   (must be < {TOLERANCE:g})")
    print(f"  inference     {ms:.1f} ms median  (must be < 25 ms)")

    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update({"onnx": out.name, "onnx_drift": drift, "onnx_ms": round(ms, 2)})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if meta.get("synthetic"):
            print("\n  NOTE: this model was trained on synthetic data — not a real classifier.")

    ok = True
    if drift >= TOLERANCE:
        print(f"\nFAIL: ONNX output differs from Keras by {drift:.2e}. Do not ship this model.")
        ok = False
    if ms >= 25:
        print(f"\nFAIL: {ms:.1f} ms exceeds the 25 ms budget, which eats the 600 ms end-to-end target.")
        ok = False
    if ok:
        print("\nOK: ONNX matches Keras and fits the latency budget.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
