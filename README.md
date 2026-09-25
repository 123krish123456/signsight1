# SignSight

Translates **Indian Sign Language into English text and speech, live from a webcam.**

A deaf or mute person signs at their laptop. The browser tracks their hands, face and
upper body, sends only the coordinates to a local server, and the server works out which
sign was made and builds an English sentence from it. Two front ends use that: a
**Speaker app** the signer uses to be heard aloud, and a **Chrome extension** that
captions a signer during a video call.

Minor project. Specification: [`docs/SignSight_PRD.md`](docs/SignSight_PRD.md).
Proposal, background, literature review and planning: [`docs/first-review.md`](docs/first-review.md).

---

## Where it stands

**76.1% ± 7.2%** top-1 over 24 ISL signs, measured by leave-one-signer-out
cross-validation — that is, tested on people the model has never seen. Chance is 4%; the
project's target is 85%. At the confidence threshold the system actually uses, **82% of
what it chooses to say is correct.**

> That figure is from eight folds over 1,001 clips. The corpus is now **1,245 clips from
> eleven signers** — all three of us have recorded — and the nine-fold re-run is in
> progress. The shipping model is already retrained on the full set.

The mean is the least interesting number we have. **A signer recorded on a laptop webcam
scores 29-39% if the training set contains only studio footage, and 68-71% if it contains
one other laptop webcam.** Domain match is worth 30-40 points; every other change measured
on this project is worth single digits. Whoever signs at the demo has to have recorded.

Full numbers, ablations and failure analysis: [`docs/results.md`](docs/results.md).

| Milestone | State |
|---|---|
| M0 Skeleton | done — backend, app, extension shell, CI |
| M1 Landmark pipeline | done — browser and Python agree to 8.9e-16, 15 FPS sustained |
| M2 Data | done — 1,245 clips, 24 signs, 11 signers: the INCLUDE corpus plus all three of us |
| M3 Model | done — exported to ONNX at 13.3 ms. **Short of the 85% target** |
| M4 Live recognition | done — classifier, confidence gating and sentence assembly wired in |
| M5 Speaker app | done — transcript, speech and reference sheet; the extension is M6 |
| M6 Listener extension | shell only — screen capture not written |
| M7 Evaluation | done bar the final write-up — ablations, failure analysis, latency measured |

The system runs end to end today: sign at the camera and it speaks an English sentence.
**409 ms p50** from the end of a sign to the gloss appearing, against a 600 ms budget.

What remains is the Chrome extension (M6) and a third signer's recordings.

---

## Getting set up

You need **Python 3.11+** and **Node 22.18+**. Windows, macOS and Linux are all fine.

```bash
git clone https://github.com/5C3PT3R/signsight
cd signsight

pip install -e ".[dev]"      # backend and tests
pip install -e ".[ml]"       # + recording and feature extraction (large: mediapipe, opencv)

cd app
npm install
npm run fetch-assets         # downloads the tracking models, ~46 MB, once
cd ..
```

`npm run fetch-assets` is not optional — the browser cannot track hands without it.

### Check it works

```bash
pytest                       # 60 tests, a few seconds
```

---

## Running it

Two terminals.

```bash
# terminal 1 — the backend
uvicorn backend.main:app --reload

# terminal 2 — the web app
cd app && npm run dev
```

Then open **http://127.0.0.1:5173** and press **Start camera**.

You should see roughly 15 FPS and the hand indicators (`L ● R ●`) light up as you move
each hand into frame. The **SIGNING / IDLE** badge responds to you in real time.

Now sign one of [the 24 signs](#the-24-signs) — the reference sheet is in the app, under
**Signs it knows**. A recognised sign appears as a gloss, a finished sentence appears in
the transcript, and the browser speaks it. Untick **Speak** to silence it.

**Expect it to be wrong sometimes.** 76.1% means roughly one sign in four. A `…` means a
sign was detected but the model was not confident enough to name it, which is deliberate:
the system says nothing rather than guessing.

> Accuracy is far lower for someone who has not recorded training clips — measured at
> 29–39%. If you are demonstrating this, record yourself first.

To exercise the interface without signing at all, start the backend with the mock
recogniser:

```bash
SIGNSIGHT_MOCK_RECOGNITION=true uvicorn backend.main:app --reload
#   PowerShell: $env:SIGNSIGHT_MOCK_RECOGNITION="true"; uvicorn backend.main:app --reload
```

It fabricates realistic glosses and sentences from your actual movements, so the
transcript and speech can be built and demonstrated now. **Turn it off for any real
demo** — it does not look at the camera.

---

## Recording clips

All three of us have done this. It is written up because anyone else who wants the
system to work for *them* has to do it too — see the numbers below for why.

### Why

Holding out each of the two who recorded first, the model reads them at 68–71% — but take
the *other* one's clips out of training and that drops to **29–39%**, because the
remaining eight signers were all filmed at 1080p on a tripod and the model has then never
seen what a laptop webcam looks like. Nothing else measured here moves accuracy by more
than single digits: cross-corpus pre-training was worth 11 points, doubling the clip count
2.9, rotation augmentation 0.6. **Domain match is worth 30 to 40.**

So this is not "more data would be nice". **If you are going to sign in front of the demo,
your clips have to be in the training set**, recorded on the machine you will demo on.

### Who records what

**All three of us record all 24 signs.** Do not split the vocabulary between us: the
evaluation works by holding one person out, so a held-out person needs examples of every
sign.

**All three of us have recorded. 749 clips.**

| Who | Clips | Hands tracked |
|---|---|---|
| Eashan | 244 | **98%** |
| Krish | 240 | 61% |
| Arpit | 265 | 35% |

For comparison the INCLUDE studio corpus manages 86–92%, so the last batch is the best
data this project has.

The difference is one thing. Krish and Arpit sat at normal laptop distance, so their hands
spend most of each clip below the bottom of the frame — 92% of every failed hand detection
had the wrist at or past an edge, a median of a third of a frame-height outside the
picture. **Push your chair back until your waist is in shot.** The recorder now draws the
safe area, refuses to save a clip where nothing moved, draws the tracked skeleton over
your hands, and reports the hand-visible percentage after every take.

Check any batch before trusting it:

```bash
python -m ml.data.precompute --report    # tracking rate per signer; below 70% is a problem
```

### How — in the browser (Node only, no Python, no backend)

```bash
cd app && npm install && npm run dev
```

Open **http://127.0.0.1:5173/record**, click **your own tab**, then **Choose a folder and
start**. Clips are written into a folder you pick on your own laptop — nothing is uploaded.
Zip that folder afterwards and send it over.

> Use `localhost`, not your machine's network address. Browsers only grant camera access
> on a secure origin, so `http://192.168.x.x:5173` will not work at all.

Press **Record** or the spacebar: 1.5 s countdown, 3 s clip, saved automatically, then it
moves to whichever sign you have fewest of. `u` undoes, `n` skips, and **keep going
automatically** records continuously. Pick the same folder next session and it resumes.

Full detail, including how Eashan merges everyone's folders:
[`docs/recording.md`](docs/recording.md).

### How — on the desktop instead

Same thing, needs the `ml` extra installed:

```bash
python -m ml.data.record --signer eashan --target 10
```

### Getting it right

**None of us knows ISL, so copy the reference exactly.** A wrong gesture labelled with a
real word is worse than no clip at all — it teaches the model something false and nothing
in the pipeline can detect it.

- **Mirror the reference.** Your camera view is flipped, so if the reference signer uses
  their right hand, use yours. It will look like the opposite side on screen. That is right.
- **Frame yourself from the waist up**, both hands able to move without leaving frame, and
  **keep your face visible** — removing face landmarks costs 9 points of accuracy.
- **Change something halfway.** Five clips, then move to a different room or change the
  lighting, then five more. Variety is the entire point.
- **Record some of it in the room we will demo in.**
- **Sign naturally.** Signs take half a second to two seconds; do not hold a pose stiffly
  for the full three.

### After everyone has recorded

```bash
python -m ml.data.ingest videos ml/data/clips --source self \
    --signer-pattern '^([^/]+)/' --no-letters
python -m ml.data.manifest --assign --check
python -m ml.data.precompute --workers 6       # ~4 minutes for 500 clips
python -m ml.data.precompute --report          # tracking rate — check before training
python -m ml.crossval --pack backend/vocab/isl_v2_words.json \
    --init-from ml/models/include_pretrain_long.keras
```

Read the webcam folds, not the mean. They are the only ones that predict what the demo
will do.

---

## The 24 signs

```
HELLO  THANK-YOU  GOOD-MORNING  HOW-ARE-YOU  ALRIGHT  PLEASED
ME  YOU  HE  SHE  WE
MOTHER  FATHER  FRIEND  MAN  WOMAN
HAPPY  SICK  HEALTHY  BIG  SMALL  COLD
HOUSE  SCHOOL
```

These are not the words the specification originally chose. The public corpus covered only
3 of those 50, against the spec's own threshold of 15, so the vocabulary was rebuilt from
signs the data actually contains — same budget of 24 words, chosen so sentences still work
("I am happy", "Mother is sick"). The reasoning is in
[`docs/datasets.md`](docs/datasets.md).

---

## Who owns what

| Who | Area | Files |
|---|---|---|
| **Eashan** | Features, training, evaluation | `ml/`, `backend/config.py` |
| **Krish** | Backend recognition path | `backend/pipeline/`, `backend/storage/` |
| **Arpit** | Both front ends | `app/src/`, `extension/` |

**What is left, in priority order:**

1. **Arpit — the Chrome extension (M6).** The only substantial build remaining. The
   service worker owns the WebSocket and the content script draws the overlay, both
   written. What is missing is the middle: `manifest.json` requests the `offscreen`
   permission but there is no `offscreen.html` or `offscreen.js`, so nothing captures the
   screen or runs MediaPipe on it. Those files are the job. Step-by-step walkthrough:
   [`docs/extension-guide.md`](docs/extension-guide.md).
2. **Everyone — the final report.** M0-M5 and M7 are done; M6 is the gap.

Fingerspelling stays out of v1: 26 of the 50 planned classes have no training data, and a
letter branch that can never fire is untested code rather than a feature.

**Interfaces that are settled** — build against these rather than renegotiating them:

- Wire events: `docs/SignSight_PRD.md` §5.1
- Classifier: in `(45, 261)` float32, out softmax over `VocabPack.labels`
- Vocabulary and sentence templates are data, in `backend/vocab/*.json`
- **The 261-value feature spec is frozen.** Changing it means changing `extract.py`,
  `normalise.ts` and `test_parity.py` in one commit — talk to Eashan first.

Run `pytest` before you push. CI runs it on every push anyway.

---

## How it works

**Landmarks on the wire, never video.** The browser extracts 261 numbers per frame
describing the position of the body, hands and face, and sends only those — about
0.33 Mbps, against 4–8 Mbps for video. That single decision is what makes it real time.

```
browser: camera → MediaPipe → 261 numbers → WebSocket
                                               ↓
backend:  ring buffer → segmenter → classifier → sentence templates → transcript
```

- **`backend/`** — FastAPI server, sign boundary detection, vocabulary validation
- **`ml/`** — dataset ingest, feature extraction, training, evaluation
- **`app/`** — React speaker app and the browser recorder
- **`extension/`** — Chrome extension for captioning video calls
- **`docs/`** — specification, results, data strategy, front-end guide

Two implementations of the feature spec exist, one in Python for training and one in
TypeScript for the browser. They must agree exactly, so `tests/test_parity.py` runs both
on identical input and fails if they drift. If they ever disagree, accuracy collapses in
production while every other test stays green.

---

## Useful commands

```bash
make help                    # everything below, listed
make test                    # tests and module self-checks
make data                    # extract and ingest downloaded corpora
make ingest                  # newly recorded clips → manifest → features → report
make train                   # train the classifier
make crossval                # the honest accuracy: leave one signer out, every signer
make evaluate                # accuracy, confusion matrix, ablations
make latency                 # sign-end → gloss, p50/p95 (needs `make serve` running)
```

Without `make` (Windows), this is what `make ingest` and `make crossval` run:

```bash
python -m ml.data.ingest videos ml/data/clips --source self \
    --signer-pattern '^([^/]+)/' --no-letters
python -m ml.data.manifest --assign --check
python -m ml.data.precompute --workers 6
python -m ml.data.precompute --report        # tracking rate per signer — read this first
python -m ml.crossval --pack backend/vocab/isl_v2_words.json \
    --init-from ml/models/include_pretrain_long.keras
```

Others worth knowing:

```bash
python -m ml.data.catalogue --dir <dir>          # what signs a dataset contains
python -m ml.data.ingest videos <dir> --probe    # is a dataset usable at all?
```

Every tunable — thresholds, frame rates, timeouts — lives in `backend/config.py` and can
be overridden with a `SIGNSIGHT_*` environment variable, so nothing needs a code change
during a demo. See [`.env.example`](.env.example).

---

## Honest limitations

- **v1 recognises isolated signs, not continuous signing.** Continuous sign language
  translation is an unsolved research problem and this project does not attempt it.
- **76.1% is short of the 85% target**, and a learning curve shows more clips of the same
  kind converge around 80%. More *signers* would help; more footage of the same eight
  would not.
- **It does not transfer across cameras.** A laptop-webcam signer scores 29-39% against a
  training set of studio footage alone. The three of us have recorded, so the system works
  for the three of us; it has no claim to working for a fourth person.
- **Signer identity in the public corpus is inferred**, not labelled — recovered from the
  camera's file numbering. Consistent and checkable, but a heuristic.
- **Two of the three home recordings are poorly framed** — hands tracked in 35% and 61%
  of frames against 86-92% for the studio corpus, because hands leave the bottom of the
  shot. They are kept regardless: they are load-bearing for cross-camera transfer, and
  re-recording them would cost more than it gained.
- **Fingerspelling is untrained.** 26 of the 50 planned classes have no data.
- Normalisation is invariant to distance and horizontal position, **not** to camera angle.
