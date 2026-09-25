# What went wrong, and what it taught us

An honest account of the bugs, dead ends and wrong turns behind SignSight. It is kept
because the finished numbers in [`results.md`](results.md) make the work look tidier than
it was, and because most of what we learned came from things that did not work.

Three categories: **bugs that silently corrupted data**, **hypotheses we tested and had to
abandon**, and **measurements that flattered us until we looked harder**.

---

## 1. Bugs that were invisible until measured

The dangerous defects on this project were never crashes. Every one of them let the
system keep running while quietly degrading it.

### MediaPipe returns 478 face points, not 468

The feature spec selects 20 face landmarks by index from a 468-point mesh. MediaPipe's
Tasks API returns 478 — the extra ten are irises. Our guard checked for exactly 468, did
not match, and silently wrote zeros into 60 of the 261 features.

Nothing failed. Every test passed. The model simply trained on a quarter of its input
missing, and we only found it by asserting on the shape of real output rather than
fixture output.

**Lesson:** a validation check that silently does nothing when it does not match is worse
than no check.

### Frame decimation with a hardcoded stride

`extract_video` took every second frame to convert 30 FPS footage to the 15 FPS the
clients run at. Correct for the studio corpus. Our own recorder writes 15 FPS, so half of
every clip one team member recorded was thrown away before it reached the model — his
clips were processed at 7.5 FPS for weeks.

The stride is now derived from each clip's own frame rate.

### A frame rate of 1000

MediaRecorder writes WebM that reports 1000 FPS and 2,912 frames for a three-second clip
that actually holds 88. That nonsense fed MediaPipe's video tracker timestamps one
millisecond apart. Implausible metadata is now rejected rather than trusted.

### An integrity check that always passed

The archive verifier ended with `return z.testzip() is None or True`. That expression is
unconditionally true. It approved a corrupt download that was 130% of its expected size,
and we only noticed when extraction failed much later.

### A resume flag that corrupted what it resumed

`curl -C -` on an already-complete file appends rather than skipping. It silently
corrupted two archives, then four more. Downloads now write to a `.part` file and verify
size before being moved into place.

### A missing dependency nobody could have hit locally

`backend/capture.py` accepts file uploads, which FastAPI cannot build a route for without
`python-multipart`. It was never declared. It happened to be installed on the maintainer's
machine, so everything worked; **a fresh clone could not start the backend at all.** CI
found it the first time a test imported the app.

### Wrist jitter read as signing

The pose model reports a wrist position whether or not the hand is in shot. Out of shot
that estimate is unconstrained and jitters hard. The segmenter looked only at wrist
velocity, so someone sitting perfectly still with their hands in their lap was read as
SIGNING — and the classifier then named a sign from a segment containing no hand data at
all. A frame with no detected hand now contributes no motion.

### A mock that demonstrated a system that could not exist

The development mock named fourteen signs the shipped vocabulary does not contain, because
its scripts were written against a vocabulary pack we had abandoned. Anyone demonstrating
with it was showing output the real system could never produce. It now emits glosses only
and the real assembler builds the sentence, so it cannot disagree with the system.

### Others, briefly

- `model_path` pointed at `signsight_v1.onnx`, a filename from the plan that nothing has
  ever written.
- The default vocabulary pack was the abandoned one, so the recorder offered 24 signs the
  backend rejected 20 of.
- Clip numbering counted files instead of taking the highest index, so deleting a clip
  made the next upload overwrite an existing one.
- The recorder's auto mode never advanced, because `advance()` was called only when auto
  was *off*.
- Auto mode had no motion detection at all: it recorded on a timer, so a pause to read the
  reference was saved as a clip of someone sitting still, labelled with a real word.
- OpenCV's `mp4v` writes MPEG-4 Part 2, which no browser plays. The reference panel was
  black and nothing reported an error.
- Malformed JSON on the WebSocket raised out of the handler, killing the session with a
  stack trace and no message to the client.

---

## 2. Six hypotheses we had to abandon

The third signer's held-out accuracy came in at **37.3%** against 72.7–87.0% for everyone
else, despite having the best-tracked footage in the corpus. Six explanations were tested.
All six were wrong.

| # | Hypothesis | Test | Outcome |
|---|---|---|---|
| 1 | The badly-tracked clips poison the webcam domain | retrain without them | **Worse.** 37.3% → 32.8%, and → 20.5% with both other home signers removed |
| 2 | The clips are mirrored | score them flipped | **Worse.** 11.2% → 6.2% |
| 3 | Reclining or unusual framing | shoulder tilt and torso geometry per signer | **Closest of the three to the studio corpus** |
| 4 | The signs blur into each other | between-sign distance over within-clip motion | **Highest ratio of anyone**, 4.12 |
| 5 | Both hands visible confuses one-handed signs | blank the idle hand | **Worse.** 14.3% → 7.8% |
| 6 | The movement is too small and slow | amplify it ×1.5, ×2, ×3 | **No change.** 14.3% → 14.8% |

A seventh idea — keep only the clips a studio-trained model already agrees with — also
made things worse, 37.3% → 25.4%.

What survived is a pattern rather than a mechanism. **Every time home-recorded data was
removed, accuracy fell; every time it was added, accuracy rose.** The signer is not badly
recorded; there is simply nobody else like him in the corpus. His 244 clips improved six
of the other eight folds, one by 12 points, while remaining unreadable by a model trained
without them: data that teaches well and cannot be read back.

It is reported unresolved. Dropping an inconvenient fold would have bought a tidier mean
at the cost of every other number in the document.

---

## 3. Measurements that flattered us

The most persistent hazard on this project was not broken code. It was numbers that looked
fine.

### 52.0% against 76.1%

The first accuracy figure came from a single signer-disjoint split: one person held out,
75 clips, a standard error above five points. It reported 52.0%. Leave-one-signer-out
cross-validation of the same system returned 64.5%, and the split had happened to hold out
the hardest signer. A random split would have reported higher still and meant nothing.

Every figure in `results.md` is now cross-validated, and the training script refuses to
run on a manifest whose splits are not signer-disjoint.

### 7 milliseconds against 409

The first latency harness measured from the last frame *sent* to the arrival of the gloss
and reported 7 ms — eighty times inside budget. It was measuring the server's response
time, not what a signer waits. The clock has to start when the hands stop, because the
segmenter cannot know a sign has ended until it has seen several frames of stillness. The
honest figure is **409 ms p50**, which still passes, and is a number that means something.

### 53.2% against 37.7%

A per-signer score was computed over "the first 80 clips", which the manifest happens to
order by sign — so it covered a handful of classes rather than a sample. Over all 239 it
was 37.7%. Convenience sampling, caught only by rerunning it.

### A test that passed by skipping

A regression test for a real bug was written so that it skipped whenever the synthetic
input was not recognised — which was always. It reported success for days without ever
executing its assertion. It now stubs the classifier so it runs.

### The wrong sign to test with

When investigating the low fold, we asked the signer to re-record `HELLO` and compare.
Both attempts scored 0/10 — but so did another signer's `HELLO`. The sign is one the model
fails for *everyone*, so the experiment could not have distinguished anything. The right
test is a sign one person gets right and another does not.

### Read the field, not the flag

A check on repository permissions printed the first `true` entry in GitHub's permissions
object. `pull` sorts before `push`, so full write access was reported as read-only, and
the team documentation was briefly wrong about it. `role_name` says `write` and is
unambiguous.

---

## What we would tell someone starting this

1. **Measure the thing you will be judged on.** Almost every mistake here came from
   measuring something adjacent — a single split instead of cross-validation, server
   latency instead of user-perceived latency, a convenient slice instead of a sample.
2. **A check that can silently pass is not a check.** The 468-vs-478 landmark bug, the
   archive verifier and the skipping test all reported success while doing nothing.
3. **Look at the data as pictures, not only as numbers.** The framing problem that cost
   two thirds of our hand tracking was invisible in every summary statistic and obvious in
   a single contact sheet of video frames.
4. **Keep the result you cannot explain.** The 37.3% fold is the most interesting thing in
   the project, and the temptation to drop it was strongest exactly when it mattered most.
