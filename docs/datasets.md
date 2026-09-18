# Data strategy — public datasets instead of self-recorded clips

## The decision

Self-recording the 1,250 clips M2 asks for was not possible for this team. v1 is
therefore trained entirely on public data, in three parts:

1. **ISL words** — from public ISL video datasets, which carry signer identity and so
   support the signer-disjoint splits the spec requires.
2. **The manual alphabet** — from public ISL *image* datasets, because no public ISL
   fingerspelling **video** set of usable size exists. Stills are expanded to a
   fixed-length sequence.
3. **ASL pre-training** — the large ASL corpora are used to pre-train, and the ISL data
   fine-tunes on top. The 261-d feature vector describes body geometry, not a language,
   so the recurrent layers learn "how signing moves" from ASL and only the classifier
   head is language-specific.

The shipped language is still **ISL**. ASL is used as a source of pre-training signal,
never as the output vocabulary.

## What this costs, stated plainly

These are real compromises. `python -m ml.data.manifest --check` prints them after every
run so they reach the write-up instead of being quietly forgotten.

- **Alphabet classes are not signer-disjoint.** The image datasets do not say who is
  signing, so those classes fall back to a per-class random split. Their accuracy is
  optimistic and **must be reported separately from the word classes.** Quoting one
  blended number across both would be the kind of inflated figure the spec forbids.
- **Alphabet classes have no motion.** A still expanded to 45 identical frames has zero
  velocity throughout. Letters that involve movement cannot be represented at all.
- **No clips in our own conditions.** Every sample comes from someone else's camera,
  lighting and background. Live accuracy in the demo room will be lower than the test
  split suggests — the spec names this as a High risk, and our mitigation for it
  (recording in the demo lighting) is exactly the thing we could not do.

## Datasets surveyed

### ISL — words

| Dataset | Content | Signers | Per class | Access |
|---|---|---|---|---|
| **FDMSE-ISL** | 40,033 videos, 2,002 words | 20 | ~20 | On request — <https://cs.rkmvu.ac.in/~isl/> |
| **INCLUDE / INCLUDE-50** | 4,287 videos, 263 words | 7 | 25 in the -50 subset | Free (Zenodo, Kaggle mirrors) |
| CISLR | 7,050 videos, 4,765 words | 71 | ~1.5 | Free — too few per class |
| iSign | 118K video–sentence pairs | — | — | Sentence-level; out of scope |

### ISL — alphabet (all still images)

| Dataset | Content | Signers | Access |
|---|---|---|---|
| ISL-dataset (ananyaarya22) | 36 classes (A–Z, 0–9), 1,000 images each | not labelled | Kaggle |
| ISL character-level (prathumarikeri) | A–Z | not labelled | Kaggle |
| IEEE DataPort ISL fingerspelling | 35 classes, ~400 images each | not labelled | IEEE DataPort |

> A thousand images per letter sounds generous, but many are near-duplicate frames from
> one session. Deduplicate, and do not read a high letter accuracy as meaningful.

### ASL — pre-training only

| Dataset | Content | Signers | Per class | Access |
|---|---|---|---|---|
| **ASL Citizen** | 83,912 videos, 2,731 signs | 52 | ~31 | Free; ships signer-independent splits |
| WLASL | 21,000 clips, 2,000 words | 119 | ~11 | Free |
| MS-ASL | 25,000 videos, 1,000 words | 222 | ~25 | Free; some source links have rotted |

## How to load them

`ml/data/ingest.py` assumes nothing about layout — you say where the class and the
signer live. Always dry-run first; it reports coverage without writing.

```bash
# 1. ISL words. Use --signer-pattern if the signer appears in the path.
python -m ml.data.ingest videos D:/datasets/INCLUDE --source include \
    --signer-pattern "(signer\d+)" --dry-run

# 2. Alphabet images. No signer to extract, so they are marked unlabelled.
python -m ml.data.ingest images D:/datasets/ISL_alphabet --source isl-alphabet --dry-run

# 3. ASL for pre-training, into its own manifest so it never mixes with ISL.
python -m ml.data.ingest videos D:/ASL_Citizen --source asl-citizen \
    --csv splits/train.csv --class-col Gloss --file-col "Video file" \
    --signer-col "Participant ID" --split train --pack backend/vocab/asl_v1.json

# then
python -m ml.data.manifest --assign --check
```

Coverage will not be 100%. Where a dataset uses a different word, either add that name
to the vocab pack or drop the class — the spec explicitly allows cutting the vocabulary
to what is actually available.

**Download the videos, not anyone's pre-extracted landmarks.** Our 261-d vector has a
specific composition and normalisation, locked to `normalise.ts` by a test. Foreign
landmarks will not match and will quietly poison the feature space.

## Training with pre-training

```bash
# 1. pre-train on ASL
python -m ml.train --pack backend/vocab/asl_v1.json --out ml/models/asl_pretrain.keras

# 2. fine-tune on ISL, reusing everything except the classifier head
python -m ml.train --init-from ml/models/asl_pretrain.keras

# 3. export and evaluate
python -m ml.export_onnx
python -m ml.evaluate
```

Report both numbers — with and without pre-training. The comparison is worth more to the
write-up than either number alone, and it is the kind of ablation §7 M7 asks for.
