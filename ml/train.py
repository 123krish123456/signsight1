"""Train the sign classifier (PRD §4.4).

    python -m ml.train                       # BiLSTM, the shipped architecture
    python -m ml.train --arch transformer    # the v1.5 comparison, for the report
    python -m ml.train --epochs 5 --synthetic  # smoke test without real clips

Splits come from the manifest and are SIGNER-DISJOINT. This script refuses to run on a
manifest that fails that check: a random split leaks signer identity and inflates
accuracy by 10–20 points, which is wrong rather than merely optimistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.config import ROOT, settings
from backend.vocab.schema import load_pack
from ml.data.manifest import load as load_manifest, verify
from ml.dataset import apply_velocity, augmented_batches, load_split

MODELS = ROOT / "ml" / "models"


# ------------------------------------------------------------------ models

def build_bilstm(seq_len: int, n_features: int, n_classes: int):
    """~450k params. Trains in under 20 minutes on a free Colab tier."""
    import keras
    from keras import layers

    return keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        # Masking lets the net skip zero-padded / undetected frames rather than
        # treating "nothing detected" as a landmark at the origin.
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="bilstm")


def build_transformer(seq_len: int, n_features: int, n_classes: int):
    """PRD §4.4 v1.5 alternative. For the report's comparison — do not start here."""
    import keras
    from keras import layers

    d_model = 128
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.Dense(d_model)(inp)
    pos = layers.Embedding(seq_len, d_model)(keras.ops.arange(seq_len))
    x = x + pos
    for _ in range(4):
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(x, x)
        x = layers.LayerNormalization()(x + layers.Dropout(0.1)(attn))
        ff = layers.Dense(d_model, activation="relu")(x)
        ff = layers.Dense(d_model)(ff)
        x = layers.LayerNormalization()(x + layers.Dropout(0.1)(ff))
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="transformer")


def build_bilstm_small(seq_len: int, n_features: int, n_classes: int):
    """A quarter of the baseline's width, and heavier dropout.

    With roughly 14 training clips per class the 587k-parameter baseline reaches 92%
    on train and half that on test — it is memorising. Less capacity is the cheapest
    thing to try against that.
    """
    import keras
    from keras import layers

    return keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(48, return_sequences=True)),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="bilstm_small")


# Registry, so architectures are selected by config rather than code edits (PRD §4.4).
ARCHITECTURES = {
    "bilstm": build_bilstm,
    "bilstm_small": build_bilstm_small,
    "transformer": build_transformer,
}


# ------------------------------------------------------------------ data

def synthetic_split(n_per_class: int, n_classes: int, seq_len: int, seed: int):
    """Class-separable noise, for exercising the pipeline with no clips recorded.

    Accuracy on this means the code runs. It says NOTHING about sign recognition.

    The class centres come from a fixed seed so every split describes the SAME classes;
    only the per-sample noise varies. Seeding the centres per split instead gives each
    split unrelated classes, and validation sits at chance forever.
    """
    centres = np.random.default_rng(1234).normal(0, 1, (n_classes, 261)).astype(np.float32)
    rng = np.random.default_rng(seed)
    x, y = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            base = centres[c] + rng.normal(0, 0.35, 261).astype(np.float32)
            ramp = np.linspace(0.6, 1.4, seq_len, dtype=np.float32)[:, None]
            x.append(base[None, :] * ramp)
            y.append(c)
    return np.asarray(x, np.float32), np.asarray(y, np.int64)


def transfer_weights(model, source_path: Path) -> int:
    """Copy weights from a pre-trained model, skipping layers whose shape differs.

    The point is training on ASL and fine-tuning on ISL. The 261-d features describe
    body geometry, not a language, so the recurrent layers transfer; only the final
    classifier is language-specific, and it has a different width per vocabulary, so it
    is left at its initial values. Returns how many layers were copied.
    """
    import keras

    source = keras.saving.load_model(source_path, compile=False)
    copied = 0
    for dst, src in zip(model.layers, source.layers):
        dst_w, src_w = dst.get_weights(), src.get_weights()
        if dst_w and len(dst_w) == len(src_w) and all(a.shape == b.shape for a, b in zip(dst_w, src_w)):
            dst.set_weights(src_w)
            copied += 1
    return copied


SHORTFALL = "below"  # the clips-per-class problem — the one thing a corpus can simply cap


def require_signer_disjoint_manifest(accept_shortfall: bool = False) -> list[str]:
    """Refuse to train on a manifest that fails M2. Returns the accepted deviations.

    PRD rule 9 forbids lowering a criterion that cannot be met, so the clips-per-class
    target is never relaxed silently: accepting it takes an explicit flag, and the
    shortfall is written into the model's metadata so any number produced carries the
    caveat with it.

    Leakage problems — a signer present in two splits, a class missing from a split —
    are never waivable. Those make a number wrong rather than merely limited.
    """
    problems = verify(load_manifest(), None)
    if not problems:
        return []

    waivable = [p for p in problems if SHORTFALL in p]
    blocking = [p for p in problems if SHORTFALL not in p]

    if blocking or not accept_shortfall:
        hint = ""
        if waivable and not blocking:
            hint = ("\n\nThe clips-per-class shortfall can be accepted with "
                    "--accept-shortfall when the corpus has no more; it is recorded "
                    "with the model.")
        raise SystemExit(
            "Refusing to train — the manifest does not meet M2:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + hint
            + "\n\nThen: python -m ml.data.manifest --assign --check"
        )

    print("ACCEPTED DEVIATION FROM M2 (recorded in the model metadata):")
    for p in waivable:
        print(f"  - {p}")
    print()
    return waivable


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="bilstm", choices=sorted(ARCHITECTURES))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accept-shortfall", action="store_true",
                    help="train despite fewer clips per class than the target, when the "
                         "corpus has no more. Recorded with the model.")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated data to verify the pipeline (no clips needed)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="pre-trained .keras to transfer weights from (e.g. an ASL model)")
    ap.add_argument("--pack", type=Path, default=None,
                    help="vocabulary pack to train against (default: the configured one)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import keras

    keras.utils.set_random_seed(args.seed)
    pack = load_pack(args.pack or settings.vocab_pack)
    n_classes = len(pack.labels)
    seq_len = settings.segment_resample_frames

    deviations: list[str] = []
    if args.synthetic:
        print("!! SYNTHETIC DATA — verifies the pipeline only, not recognition accuracy\n")
        x_train, y_train = synthetic_split(24, n_classes, seq_len, args.seed)
        x_val, y_val = synthetic_split(6, n_classes, seq_len, args.seed + 1)
        x_test, y_test = synthetic_split(6, n_classes, seq_len, args.seed + 2)
    else:
        deviations = require_signer_disjoint_manifest(args.accept_shortfall)
        x_train, y_train, s_train = load_split("train", pack)
        x_val, y_val, s_val = load_split("val", pack)
        x_test, y_test, _ = load_split("test", pack)
        print(f"train {len(x_train)} clips from signers {sorted(set(s_train))}")
        print(f"val   {len(x_val)} clips from signers {sorted(set(s_val))}")
        assert not (set(s_train) & set(s_val)), "train and val share a signer"

    x_val_f, x_test_f = apply_velocity(x_val), apply_velocity(x_test)
    n_features = x_val_f.shape[-1]

    model = ARCHITECTURES[args.arch](seq_len, n_features, n_classes)
    if args.init_from:
        copied = transfer_weights(model, args.init_from)
        print(f"transferred {copied} layers from {args.init_from.name}; "
              f"the {n_classes}-class head starts fresh")
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        # Label smoothing 0.1: with few clips per class, hard targets overfit fast.
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()

    steps = max(1, len(x_train) // args.batch_size)
    history = model.fit(
        _one_hot_stream(augmented_batches(x_train, y_train, pack, args.batch_size, args.seed), n_classes),
        steps_per_epoch=steps,
        epochs=args.epochs,
        validation_data=(x_val_f, keras.utils.to_categorical(y_val, n_classes)),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ],
        verbose=2,
    )

    out = args.out or MODELS / f"signsight_{args.arch}.keras"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)

    test_loss, test_acc = model.evaluate(
        x_test_f, keras.utils.to_categorical(y_test, n_classes), verbose=0
    )
    meta = {
        "arch": args.arch,
        "classes": pack.labels,
        "seq_len": seq_len,
        "n_features": n_features,
        "synthetic": args.synthetic,
        "train_clips": int(len(x_train)),
        "pack": pack.name,
        "init_from": args.init_from.name if args.init_from else None,
        "m2_deviations": deviations,
        "test_top1": round(float(test_acc), 4),
        "best_val_acc": round(float(max(history.history["val_accuracy"])), 4),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nsaved {out}")
    print(f"test top-1: {test_acc:.1%}" + ("  [SYNTHETIC — not a real result]" if args.synthetic else ""))
    if not args.synthetic and test_acc < 0.85:
        print(f"\nBelow the 85% target. Report {test_acc:.1%} honestly; do not tune on the test split.")
    return 0


def _one_hot_stream(batches, n_classes: int):
    import keras

    for xb, yb in batches:
        yield xb, keras.utils.to_categorical(yb, n_classes)


if __name__ == "__main__":
    raise SystemExit(main())
