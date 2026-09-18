# SignSight

Indian Sign Language → English text and speech, live from a webcam. Two modes: a
**Speaker App** for the signer, and a **Listener Chrome extension** that captions a
signer during a video call.

Spec: [`docs/SignSight_PRD.md`](docs/SignSight_PRD.md). Engineering rules: [`PROJECT_RULES.md`](PROJECT_RULES.md).

## Status — honest

| Milestone | State |
|---|---|
| M0 skeleton | ✅ done |
| M1 landmark pipeline + JS/Python parity | ✅ done (parity drift 8.9e-16) |
| M2 data | ✅ INCLUDE ingested — 496 clips, 24 classes, 8 signers ([strategy](docs/datasets.md)) |
| M3 model | ✅ trained — 57.3% top-1, signer-disjoint ([results](docs/results.md)) |
| M4 live recognition | 🔨 segmenter done and wired in; classifier blocked on M3 |
| M5 speaker app (assembler, TTS) | 🔨 buildable now against the mock recogniser |
| M6 listener extension | 🔨 shell only; buildable now against the mock recogniser |
| M7 evaluation | 🔨 ablations done (velocity, face-lite); latency harness pending |

**57.3% top-1 over 24 ISL words on a signer-disjoint test split** (chance 4%; target 85%).
At the 0.75 confidence gate the system speaks for 43% of segments and is right 72% of the
time. Full numbers, ablations and failure analysis: [`docs/results.md`](docs/results.md).

Short of the target, and reported as measured. The binding constraint is ~14 training
clips per class, not the architecture.

## Setup

```bash
pip install -e ".[dev]"          # backend + tests
pip install -e ".[ml]"           # + mediapipe/opencv, for recording and training

cd app && npm install
npm run fetch-assets             # vendors MediaPipe WASM + holistic model (~46 MB)
```

## Run

```bash
uvicorn backend.main:app --reload        # backend on :8000
cd app && npm run dev                    # speaker app on :5173
```

Open http://127.0.0.1:5173 and hit **Start camera**. You should see ~15 FPS and the
backend logging `{'received': ..., 'evicted': ..., 'lost': 0, 'fps': 15.0}`.

**Building the UI before the model exists?** Run the backend with
`SIGNSIGHT_MOCK_RECOGNITION=true` and it fabricates realistic glosses and sentences, so
the transcript, speech output and overlay can all be built and demoed today — see
[`docs/frontend-guide.md`](docs/frontend-guide.md). Turn it off for real demos: it does
not look at the camera.

Load the extension with `chrome://extensions` → Developer mode → Load unpacked →
`extension/`. It shows the overlay on Google Meet; capture arrives in M6.

## Test

```bash
pytest                                   # 38 tests; parity test needs node >= 22.18
python -m ml.features.extract            # normalisation self-check
python -m backend.pipeline.buffer        # ring-buffer self-check
```

## Team and work split

| Who | Owns | Files |
|---|---|---|
| **Eashan** | Feature pipeline, model training, evaluation, integration | `ml/`, `backend/config.py`, parity spec |
| **Krish** | Backend inference path — everything between a landmark frame and a gloss | `backend/pipeline/`, `backend/storage/` |
| **Arpit** | Both front ends — speaker app and listener extension | `app/src/`, `extension/` |

### Step 1 — load the public datasets (blocks everything else)

Self-recording is not happening, so v1 trains on public data. The strategy, the
datasets surveyed, and the compromises that must reach the report are in
[`docs/datasets.md`](docs/datasets.md). Short version: ISL words from INCLUDE/FDMSE-ISL,
the alphabet from image sets, and ASL corpora for pre-training — ISL stays the shipped
language.

```bash
python -m ml.data.ingest videos <dir> --source include --signer-pattern "(signer\d+)" --dry-run
python -m ml.data.ingest images <dir> --source isl-alphabet --dry-run
python -m ml.data.manifest --assign --check
```

`--check` prints the compromises (alphabet classes are not signer-disjoint, stills have
no motion, nothing is recorded in our own conditions). Those go in the write-up.

<details>
<summary>The original plan — recording it ourselves (not happening)</summary>


The splits are **signer-disjoint**, so the training set needs several different people
signing. We are three people, which is the bare minimum the spec allows.

> **Do not divide the vocabulary between us.** Each of us records *all 50 signs*.
> Splitting the word list three ways makes signer-disjoint splits impossible to form —
> the test signer would have no examples of the signs the other two recorded, and the
> whole evaluation falls apart.

```bash
python -m ml.data.record --signer eashan     # use your own name; SPACE record · N next · U undo · Q quit
python -m ml.data.record --signer krish
python -m ml.data.record --signer arpit
python -m ml.data.manifest --assign --check  # signer-disjoint splits + M2 gate
```

25 clips × 50 signs each ≈ 1,250 clips total, varying lighting and background.
`--check` exits non-zero until that holds; nobody starts M3 before it passes.

</details>

### Step 2 — three parallel tracks

Nobody waits for the model. The interfaces below are already fixed and committed, so
Krish and Arpit can both build against stubs while clips are still being recorded.

**Eashan — ML (M3, M7)**
`ml/train.py`, `ml/evaluate.py`, `ml/export_onnx.py`. BiLSTM per the spec, the full
augmentation stack, signer-disjoint evaluation, ONNX export. Then the confusion matrix,
latency percentiles, per-signer breakdown and ablations for the report.

**Krish — backend (M4, M5 server side)**
`classifier.py` (ONNX Runtime wrapper behind a registry so architectures swap via
config), `smoother.py` (k-of-n voting, confidence gate, repeat cooldown),
`assembler.py` (templates + fingerspelling), `session_log.py` (SQLite).
The segmenter is done and wired in — read it for the house style.
Inference goes in a `ThreadPoolExecutor`; the event loop must never block.

**Arpit — front ends (M5 client side, M6)** — start with [`docs/frontend-guide.md`](docs/frontend-guide.md)
Speaker app: transcript view, TTS via the Web Speech API, sign reference sheet rendered
from `GET /vocab`. Extension: `chrome.desktopCapture` region selection, the offscreen
document that does the cropping and landmark extraction, and overlay polish.
The MediaPipe bundling problem the spec warns about is already solved — assets are
vendored by `npm run fetch-assets`.

### Interfaces — settled, do not renegotiate mid-sprint

- **Wire format and events** — `docs/SignSight_PRD.md` §5.1. Arpit can mock every
  server event today; Krish can emit them without a model.
- **Classifier contract** — in `(45, 261)` float32, out `(51,)` softmax, class order is
  `VocabPack.labels` (50 signs then `UNKNOWN`). Krish can build and test the whole
  inference path against a stub that returns random softmax.
- **Vocabulary and templates** — `backend/vocab/isl_v1.json`, validated by
  `vocab/schema.py`. Sentence templates are data, not code.
- **The 261-d feature spec is frozen.** Changing it means changing `extract.py`,
  `normalise.ts` and `test_parity.py` in one commit — talk to Eashan first.

Everyone: `pytest` passes before you push.

## Recording data (M2 — the long pole)

```bash
python -m ml.data.record --signer <yourname>  # SPACE record · N next · U undo · Q quit
python -m ml.data.manifest --assign --check   # signer-disjoint splits + M2 DoD gate
```

Needs **25 clips × 50 classes across ≥3 signers**, varying lighting and background.
`--check` exits non-zero until that holds; do not start M3 before it passes.
Optional public data: `python -m ml.data.download_include --include-dir <path>`.

## Design notes

- **Landmarks on the wire, never video** — 261 floats/frame ≈ 0.35 Mbps instead of
  4–8 Mbps of JPEG. This is what makes it real time.
- **`normalise.ts` and `extract.py` are one spec in two languages.** Drift between them
  silently destroys live accuracy while every other test stays green, so
  `tests/test_parity.py` runs both on the same fixtures.
- **Splits are signer-disjoint**, enforced in code. Random clip splits inflate accuracy
  by 10–20 points by leaking signer identity.

## Known limitations

- Normalisation is invariant to distance and horizontal position, **not** to camera angle.
- Shoulder-width scaling uses x,y only — MediaPipe's pose `z` is too noisy to divide by.
- v1 recognises **isolated** signs with boundary detection, not continuous signing.
  Continuous sign language translation is an open research problem; this project does
  not attempt it, and says so.
