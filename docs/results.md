# Results

Every number here was produced by running the system. Reproduce with `make all`, or the
commands at the end. Tuning used cross-validation; the held-out signer of each fold was
scored once, after that fold finished training.

## Headline

**76.1% ± 7.2%** top-1 over 24 Indian Sign Language words, measured by leave-one-signer-out
cross-validation across eight folds — 1,001 clips from ten people. Chance is 4%. The
target is 85%.

At the 0.75 confidence gate the system actually uses, it speaks for about half of segments
and is right **82%** of the time — the number that matters for a demo, because
below-threshold segments surface as "…" rather than as a wrong word.

That mean hides the finding below, which matters more than the mean: **the two folds
recorded on our own laptop webcams only work because there are two of them.**

## The camera the clip was shot on decides everything

Eight of the ten signers come from the INCLUDE corpus: 1920x1080, studio lighting, a
tripod. Two are ours: 640x480 laptop webcams, at a desk. Holding out each of ours in turn,
and then removing the other one from training:

| Held out | Other webcam signer in training | Only studio signers in training |
|---|---|---|
| krish (webcam) | **68.2%** | **38.9%** |
| arpit (webcam) | **71.3%** | **29.4%** |
| the six studio signers, mean | 78.2% | 79.2 – 80.3% |

Removing one webcam signer costs the other **29 and 42 points**. Removing the same clips
costs the studio folds nothing — if anything they improve slightly, because 505 poorly
tracked clips dilute that domain.

For scale, every other lever measured on this project is worth single digits: pre-training
on 179 extra classes bought 11 points, doubling the clip count 2.9, rotation augmentation
0.6. **Domain match is worth 30 to 40.** Eight studio signers at 1080p do not teach the
model to read a person on a laptop; one other person on a laptop does.

The practical consequence is blunt. Anyone who will sign at the demo must have recorded,
and must have recorded on the machine they will demo on. This is not about the model
having seen more people — it is about it having seen that kind of picture at all.

### Why our own footage is weak, and what was done about it

`python -m ml.data.precompute --report` gives the tracking rate per signer:

| signer | clips | frames with a hand tracked |
|---|---|---|
| the eight INCLUDE signers | 496 | 86 – 92% |
| krish | 240 | 61% |
| arpit | 265 | 35% |

The cause was measured, not guessed: for **92% of arpit's and 95% of krish's** undetected
hands, the pose model still reported a wrist, and that wrist was at or past the edge of
the frame. They sat at ordinary laptop distance, so the shot is head-and-shoulders and
their hands leave the bottom of the picture as they sign. Not lighting, not motion blur,
not resolution.

Two bugs were found alongside it and fixed, neither of which changed accuracy measurably:

- `extract_video` decimated every clip by a fixed stride of 2, correct for a 25 FPS corpus
  but arpit recorded at 15 FPS, so half of every one of his clips was discarded. The
  stride now derives from the clip's own frame rate.
- MediaRecorder writes WebM that reports 1000 FPS for a three-second clip, which fed
  MediaPipe's video tracker timestamps one millisecond apart. Implausible metadata is now
  rejected in favour of a sane default.

The recorder now draws the safe area on the camera preview, shows exactly what it
captures rather than a cropped version of it, and asks for 720p instead of VGA.

## How it is measured, and why that changed the answer

The first reported figure was 52.0%, from a single signer-disjoint split. That split holds
one person out as the entire test set: 75 clips, standard error above 5 points. It cannot
separate a real improvement from a lucky run, and it also trains on less data because
validation is withheld.

Six-fold cross-validation over signers costs six times the compute and answers honestly:

| Configuration | Single split | **6-fold CV** |
|---|---|---|
| Baseline, no pre-training | 52.0% | **64.5% ± 3.1%** |
| + pre-training on 179 INCLUDE classes | 57.3% | **75.5% ± 6.5%** |
| + rotation augmentation | — | **76.1% ± 6.4%** |

The single split understated the baseline by 12 points. It also happened to hold out the
hardest signer, who remains the worst fold at 69.3%.

The pre-training gain is real rather than noise: the 95% intervals, 62.0–67.0 against
70.3–80.7, do not overlap.

## What was tried

| Change | CV mean | Verdict |
|---|---|---|
| Baseline BiLSTM | 64.5% ± 3.1% | — |
| Pre-train 140 epochs on 179 classes | 75.5% ± 6.5% | **+11.0, the one big win** |
| **+ rotation augmentation** | **76.1% ± 6.4%** | kept, though within noise |
| Fine-tune 160 epochs instead of 80 | 75.6% ± 5.8% | no gain |
| Pre-train 320 epochs instead of 140 | 73.2% ± 6.9% | worse — over-pre-training hurts |
| Velocity features (522-d) | — | no gain on the single split; not pursued |
| Lower-capacity BiLSTM (158k params) | — | worse on the single split; not pursued |
| Face-lite block zeroed at test | — | 9.3 points worse; the block earns its place |

**Pre-training is the only intervention that clearly helped.** Capacity reduction hurt,
extra fine-tuning did nothing, and more pre-training passed a peak and declined. All three
point the same way: the constraint is how little each class is seen — roughly 14 clips —
not the architecture.

**Rotation augmentation is kept on a judgement call, not a measurement.** INCLUDE was shot
on a fixed camera, so cross-validating over it contains almost no camera-angle variation
for the augmentation to earn its keep. Our normalisation is invariant to distance and
horizontal position but not to angle, and a laptop lid at a different tilt is exactly what
a webcam demo will meet. The benefit, if real, lands where this corpus cannot see it.

## Would more data help?

Answered by a learning curve rather than by intuition: the same cross-validation, run on
a stratified fraction of each fold's training clips.

| Training data | Clips per class | CV mean | Gain |
|---|---|---|---|
| 25% | ~5 | 66.0% | — |
| 50% | ~10 | 73.8% | **+7.8** |
| 75% | ~15 | 74.0% | +0.2 |
| 100% | ~20 | 76.7% | +2.7 |

**Yes, but with sharply diminishing returns.** Doubling from 5 to 10 clips per class bought
7.8 points. Doubling again, from 10 to 20, bought 2.9. Each doubling returns roughly half
the last, so the next doubling — 20 to 40 clips — is worth perhaps 1 to 2 points, and the
whole remaining series converges somewhere around **80%**.

More clips of the same kind will therefore not reach the 85% target. Anyone planning to
get there by collecting more video of these signers should know that before they start.

### What the remaining error actually is

| Held-out signer | Accuracy |
|---|---|
| signer05 | 67.3% |
| signer03 | 69.3% |
| signer02 | 77.9% |
| signer01 | 78.0% |
| signer00 | 81.4% |
| signer04 | 82.9% |

The spread across signers is **15.7 points**, against 2.9 points for doubling the clips.
Which person the model is tested on matters roughly five times more than how many clips it
trained on.

That reframes the requirement. The shortage is not clips, it is **people**: eight signers
is too few for the model to learn what varies between signers and what is the sign itself.
Twenty clips each from ten more signers would be worth far more than eighty clips each from
the same eight — which is the argument for FDMSE-ISL (20 signers) over simply recording
more of INCLUDE's seven.

## Shipping model

`ml/models/signsight_isl24.onnx`, trained with the winning recipe.

| | |
|---|---|
| Vocabulary | 24 words (`isl_v2_words.json`) |
| Clips | 496, 20–21 per class, 8 signers |
| Architecture | BiLSTM 128→64, 587k parameters |
| Estimated accuracy | **76.1% ± 6.4%** (cross-validated) |
| Held-out split score | 68.0%, 81.6% of what it chooses to say |
| Inference | 6.8 ms median, ONNX Runtime CPU |
| ONNX vs Keras | 1.04e-07 max drift |

Quote the cross-validated figure. The single-split score is one fold of the same thing and
is noisier.

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

**Roughly 14 training clips per class.** The model memorises what little it sees. Shown
three ways: reducing capacity made it worse, velocity features changed nothing, and
pre-training — which adds representation rather than parameters — was the only clear win.
More clips per class, not a better architecture, is what would close the gap to 85%.

**Signer identity is inferred, not given.** INCLUDE's filenames carry no signer label.
Each word's takes arrive in blocks separated by large jumps in the camera's counter —
"loud" has exactly seven blocks of three, and INCLUDE documents exactly seven signers —
so the Nth block of every word is taken to be the same person. It is checkable and it is
consistent, but it is a heuristic, and the split is only as trustworthy as it is.

**Only two people have recorded in our own conditions, and both framed it badly.** Hands
are tracked in 35% and 61% of their frames against 86-92% for the studio corpus, because
their hands leave the bottom of the shot. Their folds, 71.3% and 68.2%, are the two
lowest of the eight, and they only reach that because each supports the other — see the
domain-transfer table above. Live accuracy for a third person on a laptop is unmeasured
and, on this evidence, would be far lower until they too have recorded.

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

# our own clips, recorded in the browser and handed over as folders
python -m ml.data.ingest videos ml/data/clips --source self \
    --signer-pattern '^([^/]+)/' --no-letters
python -m ml.data.precompute --workers 6
python -m ml.data.precompute --report     # tracking rate per signer — check this first

# the reported figure: eight folds, one per signer
python -m ml.crossval --pack backend/vocab/isl_v2_words.json \
    --init-from ml/models/include_pretrain_long.keras

# the domain-transfer result: drop one webcam signer, watch the other collapse
python -m ml.crossval --pack backend/vocab/isl_v2_words.json \
    --init-from ml/models/include_pretrain_long.keras --exclude arpit --tag no_arpit
python -m ml.crossval --pack backend/vocab/isl_v2_words.json \
    --init-from ml/models/include_pretrain_long.keras --exclude krish --tag no_krish

# the shipping model
python -m ml.train --accept-shortfall     --init-from ml/models/include_pretrain_long.keras --out ml/models/signsight_isl24.keras
python -m ml.export_onnx --model ml/models/signsight_isl24.keras
python -m ml.evaluate --model ml/models/signsight_isl24.onnx --split test
python -m ml.evaluate --model ml/models/signsight_isl24.onnx --split test --ablate face
```
