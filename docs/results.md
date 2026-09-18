# Results

Every number here was produced by running the system. Reproduce with `make all`, or the
commands at the end. **No number here meets the 85% target**, and none has been
massaged: tuning was done on validation, and the test split was read once per model
after training finished.

## Headline

**57.3% top-1** over 24 Indian Sign Language words, on a test signer the model never saw.
Chance is 4%.

That is the model **validation** selected. A second configuration scored 69.3% on test —
see "Which number to report" below for why the lower one is the honest headline.

At the 0.75 confidence gate the system actually used live, it speaks for **43% of
segments and is right 72% of the time** — the more honest number for a demo, because
below-threshold segments surface as "…" rather than as a wrong word.

## Which number to report

| Configuration | Val | Test |
|---|---|---|
| Baseline | 70.1% | 52.0% |
| **Pre-trained, 50 epochs** | **80.5%** | **57.3%** |
| Pre-trained, 140 epochs | 79.2% | **69.3%** |
| Lower-capacity BiLSTM (158k params) | 63.6% | 48.0% |

Validation ranks the 50-epoch model highest; test ranks the 140-epoch model highest, by
12 points. They disagree, and that disagreement is the whole point:

- **Model selection must use validation.** Picking the 140-epoch run because it scored
  better on test is tuning on the test split, which the spec forbids and which would make
  the number meaningless as an estimate of unseen performance.
- **Neither set can resolve a 12-point gap.** Test is 75 clips from one signer, validation
  77 from another. At n=75 the standard error is ±5.3pp, so 57.3% and 69.3% carry 95%
  intervals of 46–68% and 59–80% — heavily overlapping.

So: **report 57.3%, and report the range.** The truthful statement is that this system
scores somewhere around 50–70% on unseen signers over 24 classes, and that the evaluation
sets are too small to pin it down further. Quoting 69.3% alone would be picking the
luckiest of four runs.

Splitting by signer is what makes the evaluation small — there are only ~8 signers, so
one of them is the entire test set. That is the right trade: a larger random split would
report a higher, wronger number.

## Ablations (PRD §7 M7)

| Configuration | Test | Best val |
|---|---|---|
| Baseline — 261-d features, trained from scratch | 52.0% | 70.1% |
| **+ velocity features (522-d)** | 52.0% | 68.8% |
| **+ pre-training on 179 other INCLUDE classes** | **57.3%** | **80.5%** |
| + pre-training, 140 epochs instead of 50 | 69.3% | 79.2% |
| Lower-capacity BiLSTM, 158k parameters | 48.0% | 63.6% |
| Pre-trained, face-lite block zeroed at test time | 48.0% | — |

**Less capacity does not help.** Quartering the width and raising dropout cost 4 points
of test and 6.5 of validation. The failure mode is not purely over-parameterisation —
the model needs capacity *and* representations it cannot learn from 14 clips per class,
which is exactly what pre-training supplies.

**Velocity features earn nothing.** Identical test accuracy, slightly worse validation,
double the input width. `use_velocity_features` stays `false`, as the spec's default
already had it — now with a measurement behind the choice rather than an assumption.

**The face-lite block is worth 9.3 points.** Zeroing those 60 dimensions at test time
drops 57.3% to 48.0%. The spec's argument for keeping 20 face landmarks — that ISL uses
mouthing and brow position grammatically — is supported by the data.

**Pre-training is the biggest single win.** The 24 target words have only 496 clips
between them, and no other INCLUDE category adds to that. Its other 179 classes and
3,317 clips still help as a source of representation: same signers, same camera, no
domain gap. Three of four weight layers transfer; only the classifier head is
vocabulary-specific. Validation moved 10.4 points, test 5.3.

## Setup

| | |
|---|---|
| Vocabulary | 24 words (`isl_v2_words.json`) |
| Clips | 496, 20–21 per class |
| Split | 344 train / 77 val / 75 test, **signer-disjoint** |
| Architecture | BiLSTM 128→64, 587k parameters |
| Inference | 7.2 ms median, ONNX Runtime CPU |
| ONNX vs Keras | 1.19e-07 max drift |

## Failure analysis

The systematic confusions are linguistically real rather than random:

| True | Predicted | Why |
|---|---|---|
| HE | SHE | Same handshape and motion; the distinction is subtle |
| PLEASED | YOU | Both are chest-directed |
| SCHOOL | FRIEND | Two-handed contact signs with similar trajectories |
| HEALTHY | BIG | Both are broad two-handed outward movements |

Around eight classes sit at 0% recall. Several have only 1–4 test clips, so those recalls
are noise as much as signal — a caveat that belongs in the write-up rather than a
conclusion about those signs.

## What limits this

**Roughly 14 training clips per class.** Training accuracy reaches 92% against 52–57% on
test: the model memorises what little it sees. This is a data-volume problem, not an
architecture one, which is why pre-training helped and velocity features did not.

**Signer identity is inferred, not given.** INCLUDE's filenames carry no signer label.
Each word's takes arrive in blocks separated by large jumps in the camera's counter —
"loud" has exactly seven blocks of three, and INCLUDE documents exactly seven signers —
so the Nth block of every word is taken to be the same person. It is checkable and it is
consistent, but it is a heuristic, and the split is only as trustworthy as it is.

**Nothing was recorded in our own conditions.** Every clip comes from INCLUDE's camera,
lighting and framing. Live accuracy on a laptop webcam will be lower than these numbers.

**Fingerspelling is untrained.** The 26 letters need alphabet image datasets, which are
not yet ingested. Any letter accuracy will additionally not be signer-disjoint, because
those datasets carry no signer labels, and must be reported separately.

## Reproducing

```bash
python -m ml.data.download_include --fetch E:/datasets/INCLUDE   # 56.75 GB, verified
python -m ml.data.prepare --archives E:/datasets/INCLUDE          # extract
python -m ml.data.ingest videos E:/datasets/INCLUDE/extracted \
    --source include --pass-re 'MVI_(\d+)' --no-letters
python -m ml.data.manifest --assign --check
python -m ml.data.precompute --workers 6

# pre-train, then fine-tune
SIGNSIGHT_VOCAB_PACK=backend/vocab/isl_include_all.json \
    python -m ml.train --epochs 50 --accept-shortfall --out ml/models/include_pretrain.keras
SIGNSIGHT_VOCAB_PACK=backend/vocab/isl_v2_words.json \
    python -m ml.train --epochs 80 --batch-size 16 --accept-shortfall \
    --init-from ml/models/include_pretrain.keras --out ml/models/isl_words_pretrained.keras

python -m ml.export_onnx --model ml/models/isl_words_pretrained.keras
python -m ml.evaluate --split test
python -m ml.evaluate --split test --ablate face
```
