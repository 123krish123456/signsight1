# SignSight — Product Requirements Document

**Author:** Eashan
**Version:** 1.0 (v1 scope)
**Status:** Ready for implementation
**Audience:** implementing developer + project evaluators

---

## 0. How to use this document

This PRD is written to be handed directly to whoever implements it. Read sections 1–6 in full before writing any code. Section 7 is the build order — **implement milestones strictly in order** and do not begin milestone N+1 until milestone N's acceptance criteria pass.

Section 11 contains hard rules. Violating them will break the project's real-time budget or its academic scope. If a requirement in this document appears impossible, stop and report rather than silently substituting an approach.

---

## 1. Product summary

### 1.1 One-line

SignSight translates Indian Sign Language captured from a webcam into English text and speech in real time, in two modes: a **Speaker App** used by the signer, and a **Listener Extension** that captions a signer during a video call.

### 1.2 The problem

Deaf and mute individuals cannot participate in spontaneous conversation with hearing people who do not sign. Existing solutions are asymmetric: they require the hearing party to learn sign language, or require a human interpreter to be present. Video calling has made this worse, not better — a signer on Google Meet is visible but unintelligible to most participants.

### 1.3 The two modes

**Speaker Mode (desktop web app)** — used by the signer.
The signer's webcam captures gestures. The system recognises signs, assembles them into English sentences, and speaks them aloud through the laptop speakers via TTS. Functionally a "voice" for the signer in an in-person conversation.

**Listener Mode (Chrome extension)** — used by the hearing party.
During a video call, the hearing user selects a screen region containing the signer's video tile. The extension tracks landmarks from that region and displays a live translated transcript in an overlay beside the call. The signer needs no special software installed.

### 1.4 Target users

| User | Mode | Need |
|---|---|---|
| Deaf/mute signer | Speaker | Be understood by hearing people without an interpreter |
| Hearing participant | Listener | Understand a signer on a video call without learning ISL |
| Evaluator / demo audience | Both | See a working, honest, live demonstration |

### 1.5 Success criteria for v1

- Live isolated-sign recognition at **≥85% top-1 accuracy** on a held-out test split of the shipped vocabulary
- **End-to-end latency ≤600 ms** from sign completion to text appearing on screen
- Sustained **≥15 FPS** landmark extraction on the target hardware without frame backlog
- Both modes demonstrable in a single 5-minute live demo without restart

---

## 2. Scope

### 2.1 In scope for v1

- Isolated sign recognition over a **fixed vocabulary of 50 ISL signs** (list in Appendix A)
- Automatic **sign boundary segmentation** from a continuous webcam stream
- **Template-based gloss → English** sentence construction
- **Fingerspelling fallback** for out-of-vocabulary proper nouns (A–Z ISL manual alphabet)
- Text-to-speech output via the Web Speech API
- Chrome Extension (Manifest V3) with region capture and transcript overlay
- Language-agnostic pipeline architecture with ISL as the shipped vocabulary pack

### 2.2 Explicitly out of scope for v1 (non-goals)

These are deferred to v2. **Do not implement them.** Do not add stub files, config keys, or abstractions "in preparation" for them beyond what section 4.6 specifies.

- ❌ **Continuous sign language translation** (natural, unsegmented signing with full grammar). This is an unsolved research problem; published systems score in the teens on BLEU. v1 does isolated signs with boundary detection, and the report must say so.
- ❌ **Seq2seq / transformer gloss-to-English model.** No parallel ISL-gloss↔English corpus exists at usable scale. v1 uses a deterministic template layer.
- ❌ Multilingual output translation (Hindi, Tamil, etc.) — one API call, add later if time permits
- ❌ Firebase / cloud database. v1 uses SQLite locally, and only for session logs
- ❌ Electron packaging. v1 ships as a web app served by the local backend
- ❌ User accounts, authentication, multi-user sessions
- ❌ Mobile support
- ❌ Signer-independent generalisation guarantees (v1 is tuned on a small signer pool; state this limitation honestly)

### 2.3 Why this scope

The classic failure mode for this project is attempting continuous translation, getting 30% accuracy, and having nothing demonstrable. A tightly scoped isolated-sign system that works reliably in a live demo is a **better** project than an ambitious one that does not run. The gap between v1 and true translation is the most interesting part of the written report — do not paper over it.

---

## 3. Architecture

### 3.1 Key architectural decision: landmarks on the edge, not video on the wire

**The single most important design decision in this project.**

Both clients extract MediaPipe landmarks locally and send only **landmark vectors** to the backend over WebSocket. Raw video frames are never transmitted.

| Approach | Bandwidth @15 FPS | Verdict |
|---|---|---|
| Stream 720p JPEG frames to backend | ~4–8 Mbps | ❌ Unusable |
| Stream 261-float landmark vectors as JSON | ~45 KB/s (~0.35 Mbps) | ✅ Ship this |

This is roughly a 15–25× reduction and it is the difference between a real-time demo and a slideshow. It also means the Chrome extension does not need to ship a neural network — it ships MediaPipe Tasks (WASM), which is a solved problem.

### 3.2 System diagram

```
┌─────────────────────────────┐     ┌──────────────────────────────┐
│  SPEAKER APP (React, local) │     │  LISTENER EXT (Chrome MV3)   │
│                             │     │                              │
│  getUserMedia (webcam)      │     │  getDisplayMedia + crop      │
│         ↓                   │     │         ↓                    │
│  MediaPipe Tasks (WASM)     │     │  MediaPipe Tasks (WASM)      │
│         ↓                   │     │         ↓                    │
│  normalise → 261-d vector   │     │  normalise → 261-d vector    │
│         ↓                   │     │         ↓                    │
│  Web Speech API (TTS out)   │     │  Transcript overlay (DOM)    │
└──────────┬──────────────────┘     └──────────────┬───────────────┘
           │      WebSocket (JSON landmark frames) │
           └───────────────┬───────────────────────┘
                           ↓
        ┌──────────────────────────────────────────┐
        │      BACKEND — Python / FastAPI          │
        │                                          │
        │  1. Frame buffer (ring, 45 frames)       │
        │  2. Segmenter (motion-energy FSM)        │
        │  3. Classifier (BiLSTM, ONNX Runtime)    │
        │  4. Temporal smoother (k-of-n vote)      │
        │  5. Gloss buffer → template assembler    │
        │  6. Emit transcript event                │
        └──────────────────────────────────────────┘
```

### 3.3 Repository layout

Monorepo. Create exactly this structure.

```
signsight/
├── PROJECT_RULES.md             # engineering rules (see §11)
├── README.md
├── pyproject.toml
├── .env.example
│
├── backend/
│   ├── main.py                  # FastAPI app + WebSocket endpoint
│   ├── config.py                # pydantic-settings, all tunables
│   ├── pipeline/
│   │   ├── buffer.py            # ring buffer
│   │   ├── segmenter.py         # motion-energy state machine
│   │   ├── classifier.py        # ONNX inference wrapper
│   │   ├── smoother.py          # k-of-n temporal voting
│   │   └── assembler.py         # gloss → English templates
│   ├── vocab/
│   │   ├── isl_v1.json          # 50-sign vocabulary pack
│   │   └── schema.py            # vocab pack pydantic model
│   └── storage/
│       └── session_log.py       # SQLite session logging
│
├── ml/
│   ├── data/
│   │   ├── download_include.py  # fetch INCLUDE dataset
│   │   ├── record.py            # self-recording capture tool
│   │   └── manifest.csv         # clip → label → split
│   ├── features/
│   │   └── extract.py           # video → 261-d sequences (§4.2)
│   ├── train.py
│   ├── evaluate.py              # confusion matrix, per-class recall
│   ├── export_onnx.py
│   └── models/                  # .onnx artefacts (gitignored)
│
├── app/                         # Speaker Mode — React + Vite
│   ├── src/
│   │   ├── App.tsx
│   │   ├── landmarks.ts         # MediaPipe Tasks wrapper
│   │   ├── normalise.ts         # MUST match ml/features/extract.py
│   │   ├── socket.ts
│   │   └── tts.ts
│   └── vite.config.ts
│
├── extension/                   # Listener Mode — Chrome MV3
│   ├── manifest.json
│   ├── background.js            # service worker
│   ├── offscreen.html/.js       # media processing (§6.3)
│   ├── content.js               # transcript overlay
│   ├── popup.html/.js           # region selection UI
│   └── vendor/                  # bundled MediaPipe WASM (§6.2)
│
└── tests/
    ├── test_segmenter.py
    ├── test_assembler.py
    ├── test_parity.py           # JS vs Python normalisation parity
    └── fixtures/
```

---

## 4. Core technical specification

### 4.1 Landmark extraction

Use **MediaPipe Holistic** (Python, training) and **MediaPipe Tasks — Holistic Landmarker** (JS, clients). Target capture rate: **15 FPS**. Do not run at 30 FPS; it doubles compute for no accuracy gain on signs lasting 0.5–2 s.

### 4.2 Feature vector — 261 dimensions

This spec is binding. `ml/features/extract.py` and `app/src/normalise.ts` must produce **bit-identical** output for the same input landmarks (enforced by `tests/test_parity.py`).

| Component | Landmarks | Dims each | Total |
|---|---|---|---|
| Upper-body pose (indices 0–24) | 25 | 3 (x, y, z) | 75 |
| Left hand | 21 | 3 | 63 |
| Right hand | 21 | 3 | 63 |
| Face-lite (lips + brows, indices in Appendix B) | 20 | 3 | 60 |
| **Total** | | | **261** |

**Why face-lite and not full face mesh:** full FaceMesh adds 1404 dims of mostly-noise and triples inference cost. But ISL uses mouthing and eyebrow position as grammatical markers (e.g. brow raise marks questions), so dropping the face entirely loses real signal. 20 landmarks is the compromise.

**Normalisation — apply in this exact order:**

1. Compute shoulder midpoint `M = (pose[11] + pose[12]) / 2`
2. Compute shoulder width `S = ||pose[11] − pose[12]||`
3. If `S < 1e-6`, mark frame invalid and emit a zero vector
4. For every landmark `p`: `p' = (p − M) / S`
5. If a hand is not detected, fill its 63 dims with zeros (do **not** interpolate — absence is informative; one-handed signs exist)

This makes features invariant to the signer's distance from camera and horizontal position. It does not correct for camera angle — document that as a limitation.

**Optional delta features:** `config.use_velocity_features` (default `false`) appends frame-to-frame differences, giving 522 dims. Evaluate both in `ml/evaluate.py` and ship whichever wins. Report both numbers.

### 4.3 Segmentation — motion-energy state machine

The hardest unglamorous problem: knowing where one sign ends and the next begins. Without this nothing works live, no matter how good the classifier is.

Compute per-frame **motion energy** as the mean L2 velocity of the two wrist landmarks (pose indices 15, 16) over the normalised coordinates, smoothed with a 5-frame moving average.

State machine with hysteresis:

```
IDLE ──(energy > ENTER_THRESH for 3 consecutive frames)──> SIGNING
SIGNING ──(energy < EXIT_THRESH for 4 consecutive frames)──> IDLE (emit segment)
SIGNING ──(frame count > MAX_SEGMENT_FRAMES)──> IDLE (emit segment, flag truncated)
```

Defaults in `config.py`, all tunable without code changes:

```python
ENTER_THRESH = 0.08          # normalised units/frame
EXIT_THRESH  = 0.04          # deliberately lower — hysteresis prevents flicker
ENTER_FRAMES = 3
EXIT_FRAMES  = 4
MIN_SEGMENT_FRAMES = 8       # discard shorter — noise
MAX_SEGMENT_FRAMES = 45      # 3 s at 15 FPS
```

**Two thresholds, not one.** A single threshold causes rapid IDLE/SIGNING oscillation at the boundary and produces dozens of garbage segments per second. This is the most common bug in student implementations of this pipeline.

Emitted segments are resampled to exactly **45 frames** by linear interpolation before classification.

### 4.4 Classifier

**Input:** `(45, 261)` float32 tensor
**Output:** `(51,)` softmax — 50 vocabulary classes + 1 `UNKNOWN` class

**v1 architecture (build this first):**

```
Input (45, 261)
  → Masking (skip zero-padded frames)
  → BiLSTM(128, return_sequences=True)
  → Dropout(0.3)
  → BiLSTM(64)
  → Dropout(0.3)
  → Dense(128, relu)
  → Dense(51, softmax)
```

~450k parameters. Trains in under 20 minutes on Colab's free tier. Runs in <15 ms on CPU.

**v1.5 alternative (only after v1 meets accuracy targets):** a 4-layer transformer encoder, `d_model=128`, 4 heads, learned positional encoding. Report the comparison — it is good material for the paper. Do not start here.

`classifier.py` must load models through a registry interface so architectures are swappable via config, not code edits.

**Training:**

- Loss: categorical cross-entropy with label smoothing 0.1
- Optimiser: Adam, lr 1e-3, ReduceLROnPlateau
- Augmentation (critical given small data): random temporal crop/stretch ±15%, mirror left↔right hands (relabel handedness-sensitive signs — see vocab pack `mirror_safe` flag), Gaussian landmark jitter σ=0.01, random frame dropout 10%
- Splits: **split by signer, not by clip.** A random clip split leaks signer identity and inflates accuracy by 10–20 points. This is a correctness requirement, not a nicety.
- Export to ONNX; backend inference uses ONNX Runtime, not TensorFlow

**`UNKNOWN` class:** train on segments from out-of-vocabulary signs and non-sign hand movement (scratching, gesturing, adjusting hair). Without this the model confidently misclassifies every random motion as a vocabulary sign, which destroys the live demo.

### 4.5 Temporal smoothing and confidence gating

A single classification is never emitted directly. Maintain a rolling window of the last 3 segment predictions:

- Emit a gloss only if `max(softmax) ≥ CONFIDENCE_THRESH` (default 0.75)
- Suppress a repeat of the immediately preceding gloss within `REPEAT_COOLDOWN_MS` (default 1200 ms), unless the vocab pack marks the sign as `repeatable`
- Anything below threshold emits `UNKNOWN` and is shown to the user as `…` rather than being silently dropped — the user must be able to tell "not understood" from "not signing"

### 4.6 Gloss → English assembly (template layer)

Deterministic, rule-based. No ML.

ISL grammar is broadly Subject-Object-Verb with topic-comment structure and no articles or copula. The assembler applies ordered rewrite rules from the vocabulary pack:

```json
{
  "templates": [
    { "pattern": ["ME", "NAME", "$FINGERSPELL"], "output": "My name is {2}." },
    { "pattern": ["YOU", "NAME", "WHAT"],        "output": "What is your name?" },
    { "pattern": ["ME", "$ADJ"],                  "output": "I am {1}." },
    { "pattern": ["$NOUN", "WHERE"],              "output": "Where is the {0}?" },
    { "pattern": ["ME", "$NOUN", "WANT"],         "output": "I want {1}." }
  ],
  "fallback": "join_with_spaces_and_capitalise"
}
```

Rules are matched longest-first against the gloss buffer. Unmatched glosses flush through the fallback after `ASSEMBLY_TIMEOUT_MS` (default 2500 ms) so the user is never left waiting on a template that will not match.

**Language-agnostic requirement:** the assembler must read all templates, vocabulary, and grammar hints from the vocab pack JSON. Swapping `isl_v1.json` for `asl_v1.json` and retraining must require **zero Python changes**. `vocab/schema.py` defines and validates the pack format.

### 4.7 Fingerspelling

Proper nouns cannot be in a 50-sign vocabulary. Detect the ISL manual-alphabet mode: when 3+ consecutive segments classify to letter classes, buffer them, and flush the concatenated string as a `$FINGERSPELL` token on the next non-letter sign or 1500 ms timeout.

Letter classes are part of the 50-sign budget — Appendix A allocates 26 to letters and 24 to words. This is a deliberate trade: fingerspelling makes the demo feel far more capable than 50 arbitrary words would.

---

## 5. Backend API

### 5.1 WebSocket `/ws/stream`

Both clients use the same endpoint. Bidirectional JSON.

**Client → Server, landmark frame:**

```json
{
  "type": "frame",
  "session_id": "uuid",
  "seq": 1247,
  "t_client_ms": 1723987654321,
  "landmarks": [0.123, -0.456, ...],
  "hands_present": [true, false]
}
```

`landmarks` is exactly 261 floats (or 522 if velocity features are enabled), pre-normalised client-side per §4.2.

**Client → Server, control:**

```json
{ "type": "control", "action": "start" | "stop" | "reset_buffer" }
```

**Server → Client, events:**

```json
{ "type": "state",      "value": "IDLE" | "SIGNING" }
{ "type": "gloss",      "value": "HELLO", "confidence": 0.91, "segment_ms": 830 }
{ "type": "transcript", "text": "My name is Eashan.", "is_final": true }
{ "type": "error",      "code": "BAD_DIMENSION", "message": "..." }
```

The `state` event drives a live UI indicator so the signer can see the system knows they are signing. This dramatically improves usability and costs almost nothing.

### 5.2 REST endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + loaded model name/version |
| `GET` | `/vocab` | Active vocabulary pack (clients render a sign reference sheet) |
| `GET` | `/sessions/{id}` | Session transcript log from SQLite |

### 5.3 Backpressure

If the frame queue exceeds 90 frames (6 s), drop the **oldest** frames and log a warning. Never block the WebSocket read loop on inference. Run inference in a `ThreadPoolExecutor`; the event loop must stay free.

---

## 6. Chrome Extension (Listener Mode)

### 6.1 Manifest V3 permissions

```json
{
  "manifest_version": 3,
  "permissions": ["desktopCapture", "offscreen", "storage", "activeTab"],
  "host_permissions": ["ws://localhost:8000/*"],
  "background": { "service_worker": "background.js" },
  "content_scripts": [{ "matches": ["https://meet.google.com/*"], "js": ["content.js"] }]
}
```

### 6.2 MV3 remote-code restriction — read this carefully

**Manifest V3 forbids loading and executing remote code.** MediaPipe's JS examples fetch WASM binaries and `.task` model files from a CDN at runtime. Doing that in an extension will get it rejected and will fail under the default CSP.

**Requirement:** vendor the MediaPipe WASM bundle and the `holistic_landmarker.task` file into `extension/vendor/` and load them from `chrome.runtime.getURL()`. Add a build step that copies them from `node_modules`. Budget for roughly 10–15 MB of extension package size.

This will cost an afternoon of debugging if discovered late. It is called out here so it does not.

### 6.3 Region capture

1. Popup calls `chrome.desktopCapture.chooseDesktopMedia` → user picks the tab/window/screen
2. The resulting stream goes to an **offscreen document** (MV3 service workers have no DOM and cannot process media)
3. The user drags a rectangle over a preview to select the signer's video tile; store `{x, y, w, h}` as fractions of the source dimensions, not pixels, so it survives window resizing
4. Offscreen document draws each frame to an `OffscreenCanvas`, crops to the region, feeds the crop to MediaPipe
5. Landmarks post to the service worker, which owns the WebSocket connection

### 6.4 Transcript overlay

Content script injects a fixed-position panel, default bottom-right, draggable, with an opacity control. Show the last 4 lines. Include a visible **"SignSight — automated, may contain errors"** label. Do not present machine output as a verified interpretation; this matters both ethically and for the report's discussion of responsible deployment.

---

## 7. Build order

Implement in this order. Each milestone has a verifiable definition of done. **Do not proceed past a failing milestone.**

### M0 — Skeleton (target: week 1)
- Repo structure per §3.3, `pyproject.toml`, `PROJECT_RULES.md`, `.env.example`
- FastAPI serving `/health`; Vite React app; extension loading unpacked
- **DoD:** `GET /health` returns 200; app dev server runs; extension appears in `chrome://extensions` without errors

### M1 — Landmark pipeline and parity (week 1–2)
- `ml/features/extract.py` and `app/src/normalise.ts` implementing §4.2
- `tests/test_parity.py`: same fixture landmarks through both paths, assert max abs difference < 1e-6
- WebSocket streaming live from the app; server logs frame rate
- **DoD:** parity test passes; sustained 15 FPS end to end for 60 s with no queue growth

### M2 — Data (week 2–4) ⚠️ longest pole
- `download_include.py` pulls the INCLUDE dataset, filters to the Appendix A vocabulary
- `record.py`: guided capture tool — shows the target sign, records a 3 s clip, writes to manifest, tracks per-class counts
- **Record 25 clips per sign across at least 3 signers**, varying lighting and background
- Build `manifest.csv` with a **signer-disjoint** train/val/test split
- **DoD:** ≥25 clips × 50 classes in the manifest; split verified signer-disjoint by an automated check; feature extraction runs over the full set without error

### M3 — Model (week 4–6)
- `train.py` with the §4.4 augmentation stack; `evaluate.py` producing a confusion matrix and per-class recall; `export_onnx.py`
- **DoD:** ≥85% top-1 on the signer-disjoint test split; ONNX output matches Keras output within 1e-4; inference <25 ms/segment on CPU

### M4 — Live recognition (week 6–7)
- Segmenter, smoother, ONNX classifier wired into the WebSocket loop
- **DoD:** signing 10 signs in sequence in front of the webcam produces ≥8 correct glosses with no spurious segments during pauses; latency from sign end to gloss event <600 ms (measured and logged, not estimated)

### M5 — Speaker App complete (week 7–8)
- Assembler, fingerspelling, TTS, live state indicator, transcript view, vocab reference sheet
- **DoD:** signing `ME` `NAME` then fingerspelling E-A-S-H-A-N produces spoken output "My name is Eashan."

### M6 — Listener Extension (week 8–10)
- Region capture, offscreen processing, vendored WASM, overlay
- **DoD:** in a live Google Meet call with a second machine showing a signer, the overlay produces a transcript matching the Speaker App's output for the same signs

### M7 — Evaluation and report (week 10–12)
- Confusion matrix, latency distribution (p50/p95), per-signer accuracy breakdown, ablation on velocity features and on the face-lite block
- Documented failure analysis: which sign pairs confuse, and why
- **DoD:** results reproducible from a single `make evaluate`

---

## 8. Configuration

Every tunable in §4 lives in `backend/config.py` via `pydantic-settings`, overridable by environment variable. No magic numbers in pipeline code. The demo will need threshold tuning in the room on the day — that must not require a code change or restart of anything but the server.

---

## 9. Testing

| Test | What it protects |
|---|---|
| `test_parity.py` | JS/Python normalisation drift — silently destroys accuracy in production |
| `test_segmenter.py` | Synthetic energy traces → expected segment boundaries; explicitly test the hysteresis flicker case |
| `test_assembler.py` | Gloss sequences → expected sentences, including fallback and timeout paths |
| `test_vocab_schema.py` | Vocab pack validates; `asl_v1.json` stub loads without Python changes |
| Latency harness | Replays a recorded session, asserts p95 end-to-end < 600 ms |

CI: GitHub Actions running pytest on push. Model training is not in CI.

---

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| INCLUDE dataset coverage of Appendix A vocabulary is poor | High | M2 self-recording is the primary data source; treat public data as augmentation. Verify coverage in week 2 — if <15 signs match, cut the vocabulary to what you can record |
| Live accuracy far below test accuracy | High | Signer-disjoint splits, aggressive augmentation, record in the actual demo lighting |
| Segmenter fires on non-sign motion | Medium | `UNKNOWN` class + confidence gate + `MIN_SEGMENT_FRAMES` |
| MV3 WASM bundling blocks the extension | Medium | §6.2 — resolve in the first week of M6, not the last |
| Two-handed ISL signs occlude each other | Medium | Accept and document. Zero-fill on non-detection lets the model learn occlusion patterns |
| Latency budget blown by Python GIL | Low | ONNX Runtime releases the GIL; inference in a thread pool |

---

## 11. Engineering rules

1. **Do not implement anything in §2.2.** If a task seems to need it, stop and ask.
2. **Do not stream video frames over the WebSocket.** Landmarks only. This is not negotiable (§3.1).
3. **Do not change the 261-dim feature spec** without updating `extract.py`, `normalise.ts`, and `test_parity.py` together in the same change.
4. **Never use a random train/test split.** Splits are signer-disjoint. An accuracy number from a random split is wrong, not merely optimistic.
5. **No magic numbers in pipeline code.** Everything tunable goes in `config.py`.
6. **Prefer boring, working code.** This is a graded student project with a live demo, not a research artefact. A reliable BiLSTM beats a fragile transformer.
7. **Report honest numbers.** If accuracy is 71%, the README says 71%. Do not tune on the test split.
8. **Write `PROJECT_RULES.md` first**, containing rules 1–7 plus the build/test commands, so they are on record before any code is written.
9. When a milestone's DoD cannot be met, **stop and report** with the measured numbers. Do not lower the criterion.

---

## Appendix A — v1 vocabulary (50 classes)

**Manual alphabet (26):** A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

**Words (24):** ME · YOU · NAME · HELLO · THANK-YOU · PLEASE · SORRY · YES · NO · HELP · WANT · NEED · WHAT · WHERE · WHO · WHEN · HOW · GOOD · BAD · WATER · FOOD · HOME · WORK · UNDERSTAND

Plus the `UNKNOWN` class (§4.4), giving 51 output units.

Selected so that the template layer (§4.6) can build genuinely useful sentences — questions, requests, and self-introduction — rather than disconnected words. Each entry in `isl_v1.json` carries `{ gloss, pos, mirror_safe, repeatable, reference_video }`.

## Appendix B — Face-lite landmark indices

Outer lip contour (12): 61, 291, 39, 181, 0, 17, 269, 405, 270, 314, 13, 14
Eyebrows (8): 70, 63, 105, 66, 300, 293, 334, 296

Indices refer to the MediaPipe FaceMesh 468-point topology.

## Appendix C — Reading list for the report

- Joze & Koller, *MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language*
- Li et al., *Word-level Deep Sign Language Recognition from Video (WLASL)*
- Sridhar et al., *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*
- Camgoz et al., *Neural Sign Language Translation* — the continuous-translation baseline you are explicitly not attempting; cite it when justifying scope
