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
| M2 data | 🔨 tooling done, **0 clips recorded** — needs a human and a webcam |
| M3 model | ⛔ blocked on M2 |
| M4 live recognition | 🔨 segmenter done and wired in; classifier blocked on M3 |
| M5 speaker app (assembler, TTS) | ⛔ blocked on M4 |
| M6 listener extension | ⛔ shell only |
| M7 evaluation | ⛔ |

**No accuracy number is claimed yet, because no model has been trained.** When one is,
it goes here, measured on a signer-disjoint test split, whatever it turns out to be.

## Setup

```bash
pip install -e ".[dev]"          # backend + tests
pip install -e ".[ml]"           # + mediapipe/opencv, for recording and training

cd app && npm install
npm run fetch-assets             # vendors MediaPipe WASM + holistic model (~15 MB)
```

## Run

```bash
uvicorn backend.main:app --reload        # backend on :8000
cd app && npm run dev                    # speaker app on :5173
```

Open http://127.0.0.1:5173 and hit **Start camera**. You should see ~15 FPS and the
backend logging `{'received': ..., 'dropped': 0, 'fps': 15.0}`.

Load the extension with `chrome://extensions` → Developer mode → Load unpacked →
`extension/`. It shows the overlay on Google Meet; capture arrives in M6.

## Test

```bash
pytest                                   # 27 tests; parity test needs node >= 22.18
python -m ml.features.extract            # normalisation self-check
python -m backend.pipeline.buffer        # ring-buffer self-check
```

## Recording data (M2 — the long pole)

```bash
python -m ml.data.record --signer eashan     # SPACE record · N next · U undo · Q quit
python -m ml.data.manifest --assign --check  # signer-disjoint splits + M2 DoD gate
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
