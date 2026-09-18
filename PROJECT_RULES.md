# Engineering rules — SignSight

Source of truth is [`docs/SignSight_PRD.md`](docs/SignSight_PRD.md). These rules are
binding for anyone working on this codebase.

## Hard rules

1. **Do not implement anything listed as out of scope** (continuous translation,
   seq2seq gloss→English, multilingual output, Firebase, Electron, auth, mobile).
2. **Never stream video frames over the WebSocket.** Landmarks only (261 floats/frame).
   This is the decision that makes the system real time; it is not negotiable.
3. **Do not change the 261-dim feature spec** without changing `ml/features/extract.py`,
   `app/src/normalise.ts` and `tests/test_parity.py` in the same commit.
4. **Never use a random train/test split.** Splits are signer-disjoint, enforced by
   `ml/data/manifest.py`. A random-split accuracy number is wrong, not merely optimistic.
5. **No magic numbers in pipeline code.** Everything tunable lives in `backend/config.py`
   and is overridable by `SIGNSIGHT_*` env vars, so thresholds can be retuned during a
   live demo without restarting anything but the server.
6. **Prefer boring, working code.** A reliable BiLSTM beats a fragile transformer.
7. **Report honest numbers.** If accuracy is 71%, the README says 71%. Never tune on the
   test split.
8. When a milestone's definition of done cannot be met, **stop and report** the measured
   numbers. Do not lower the bar.

## Feature vector (binding)

261 floats, in this exact order:

| Slice | Component | Dims |
|---|---|---|
| `[0:75]`    | pose landmarks 0–24 (x,y,z) | 75 |
| `[75:138]`  | left hand 21× (x,y,z)       | 63 |
| `[138:201]` | right hand 21× (x,y,z)      | 63 |
| `[201:261]` | face-lite 20× (x,y,z)       | 60 |

Normalisation order: shoulder midpoint `M=(pose11+pose12)/2`, shoulder width
`S=||pose11-pose12||` **over x,y only** (MediaPipe pose `z` is a noisy per-frame depth
estimate; dividing all 261 features by it spreads that noise everywhere — a deliberate,
documented deviation from a literal reading of the spec, and both implementations must
agree), `S < 1e-6` → the whole frame is a 261-zero vector, else `p' = (p-M)/S`.
A missing hand or face leaves its slice zeroed. Never interpolate: absence is
informative, because one-handed signs exist.

The face mesh arrives with **478** points, not the 468 the spec assumes — MediaPipe
appends 10 iris points. Accept ≥468 and take the first 468.

Face-lite indices live in `FACE_LITE_IDX`, duplicated in `normalise.ts`. Touch one,
touch the other, and run the parity test.

## Commands

```bash
# setup
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev]"
cd app && npm install && npm run fetch-assets        # vendors MediaPipe (~46 MB)

# run
uvicorn backend.main:app --reload                    # backend  :8000
cd app && npm run dev                                # speaker app :5173

# test
pytest                                               # parity test needs node >= 22.18
pytest tests/test_parity.py -v                       # JS/Python normalisation drift

# data
python -m ml.data.record --signer <name>             # guided capture
python -m ml.data.manifest --assign --check          # signer-disjoint splits + gate
```

## Status

- M0 skeleton — done
- M1 landmark pipeline + parity — done (drift 8.9e-16 against a 1e-6 budget)
- M2 data — tooling done, **recording is a human task**, blocked on clips
- M3 model — blocked on M2. Do not start before `manifest --check` passes.
- M4 live recognition — segmenter done and wired in; classifier blocked on M3
- M5–M7 — not started
