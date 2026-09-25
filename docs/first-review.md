# SignSight — First Review

**Real-time Indian Sign Language to English text and speech, from an ordinary webcam.**

| | |
|---|---|
| **Team** | Eashan Singh · Krish · Arpit |
| **Review** | First review — proposal, background and planning |
| **Repository** | https://github.com/5C3PT3R/signsight |
| **Status at review** | Working prototype: 24 signs, live recognition end to end |

---

## 1. Title of the Project

**SignSight: A Landmark-Based Real-Time Indian Sign Language Recognition and
Translation System for Commodity Webcams**

The short form used throughout is **SignSight**.

Two words in that title carry the contribution. *Landmark-based* — the system never
transmits or stores video; it extracts 261 geometric coordinates per frame in the browser
and sends only those. *Commodity webcam* — the target is the laptop a student already
owns, not a depth camera, not instrumented gloves, and not a studio.

---

## 2. Background and Domain

### 2.1 Domain

The project sits at the intersection of three areas: **computer vision** (real-time human
pose and hand landmark estimation), **sequence modelling** (classifying a gesture that
unfolds over time), and **assistive technology / HCI** (a system a deaf user would
actually choose to use).

### 2.2 The language

Indian Sign Language is a complete natural language, not a manual encoding of Hindi or
English. It has its own grammar, its own syntax and its own vocabulary [1]. Three
properties shape the engineering:

- **Broadly Subject–Object–Verb**, with topic–comment structure. It has no copula and no
  articles, so the gloss sequence `ME HAPPY` is a well-formed sentence meaning "I am
  happy." A translation layer, not a word-for-word substitution, is required.
- **Non-manual markers matter.** Facial expression and head position carry grammatical
  information — negation, question marking, intensity. A system that tracks only hands
  discards part of the language. Our own ablation confirms the cost empirically:
  removing face landmarks costs 9 percentage points of accuracy.
- **Two-handed and spatially organised.** Signs use both hands, and meaning depends on
  position relative to the signer's own body, which is why features must be normalised
  against the signer's frame rather than the camera's.

### 2.3 Why this matters

India's deaf population is estimated at between 1.8 million and 7 million; the census does
not count deafness separately, which is why the range is so wide [2]. Against that
population, ISLRTC records a total of **339 certified Level-C / DISLI interpreters**, and
other sources put the working figure at around 250 [2, 3].

That ratio — on the order of one interpreter per several thousand deaf people — is the
problem statement. Interpreters are not a resource that can be scheduled for an everyday
conversation at a bank counter, a clinic reception or a video call.

### 2.4 Why this is feasible now

Until recently, gesture recognition needed either a depth sensor (Kinect), instrumented
gloves, or a GPU running a video model. Three things changed:

1. **On-device pose estimation.** Google's MediaPipe Holistic runs face, hand and body
   landmark extraction simultaneously in a browser at real-time rates [4].
2. **Public ISL data.** The INCLUDE corpus (2020) was the first large-scale public ISL
   dataset [5]; CISLR (2022) followed with a far larger vocabulary [6].
3. **Skeleton-based recognition.** Once landmarks are available, the classifier operates
   on a few hundred numbers per frame rather than on pixels, so a model small enough to
   run on a CPU becomes sufficient.

---

## 3. Literature Review and Existing System

### 3.1 Academic work

| Work | Contribution | Limitation for our purpose |
|---|---|---|
| **Sridhar et al., ACM MM 2020 — INCLUDE** [5] | First large-scale public ISL dataset: 4,287 videos, 263 signs, 15 categories, recorded with experienced signers. INCLUDE-50 subset for fast evaluation. | Studio recordings — 1920×1080, fixed tripod, controlled lighting. Signer identity is not labelled in the filenames. |
| **Joshi et al., EMNLP 2022 — CISLR** [6] | ~4,700-word ISL vocabulary. Introduces a prototype-based **one-shot learner that transfers features from resource-rich ASL** to improve ISL prediction. | Mostly one example per word; addresses vocabulary breadth rather than deployment. |
| **Velmathi & Goyal, 2023** [7] | Applies MediaPipe Holistic to ISL; compares CNN against LSTM, finding CNN better for static letters and the sequence model better once hands, face and pose are tracked together. | Closest prior work to our architecture; does not address signer-independent evaluation or deployment. |
| **Damdoo et al., IET Image Processing 2025** [8] | Integrative survey of ISL recognition and translation. | Confirms the field's own summary: **isolated sign work greatly outweighs continuous sentence interpretation.** |
| **Survey of SLR systems, 2025** [9] | Identifies the field-wide blockers: limited annotated datasets, global imbalance in representation, difficulty of modelling multimodality. | — |
| **Signer-dependence in evaluation, 2026** [10] | Shows that the standard protocol lets the same signer appear in train and test, so models exploit hand morphology and personal articulation habits instead of the sign. Reports an **11.19-point drop** when Bangla SL evaluation is made signer-independent. | Directly motivates our evaluation protocol (§5.4). |

### 3.2 Existing systems

**SignAll** is the most mature commercial effort. Its original product used a Kinect depth
sensor with coloured gloves; it later released a gloveless SDK built on MediaPipe in
partnership with Google [11]. It remains ASL-focused and hardware- or SDK-dependent.

**Glove-based systems** instrument the hand with flex and inertial sensors. Reviews of a
decade of such work [12] report recurring limitations: they cannot resolve occlusion
between fingers, they ignore the face entirely, and — the decisive objection — they
require the deaf person to wear equipment in order to be understood.

**Vision-based research prototypes** are numerous but, as the surveys note, sensitive to
lighting, background clutter, occlusion and camera angle, and largely unvalidated outside
the conditions they were recorded in.

**Google MediaPipe** itself provides the hand and pose tracking layer but no sign
recognition: it is infrastructure that this project builds on, not a competitor.

### 3.3 The gap

No system we found offers **ISL recognition, on an unmodified laptop webcam, evaluated on
people the model has never seen.** Each existing approach relaxes one of those three:
special hardware, a different sign language, or an evaluation protocol that flatters the
result.

---

## 4. Problem Identification and Limitations of the Existing System

### 4.1 Problem statement

> A deaf ISL user cannot be understood by a hearing person who does not sign, in everyday
> unscheduled situations, because interpreters are scarce and existing automatic systems
> need hardware, a different sign language, or conditions the user does not have.

### 4.2 Limitations of existing systems

**L1 — Hardware dependence.** Depth cameras and sensor gloves make the deaf person carry
the burden of being understood. Any system requiring them will not be adopted.

**L2 — Evaluation that does not predict deployment.** Where the same signer appears in
both training and test data, reported accuracy includes credit for recognising the person
rather than the sign [10]. This is the single most common way sign-recognition results
overstate what a system will do for a new user.

**L3 — A domain gap that the literature does not quantify.** Public corpora are recorded
in studios. We measured what this costs, and it is the central finding of our work so far:

| Held-out signer | With another webcam signer in training | With only studio signers in training |
|---|---|---|
| Signer A (laptop webcam) | **68.2%** | **38.9%** |
| Signer B (laptop webcam) | **71.3%** | **29.4%** |
| Six studio signers (mean) | 78.2% | 79.2 – 80.3% |

Eight studio signers at 1080p do not teach a model to read a person sitting at a laptop.
Adding a single other webcam signer is worth **29 and 42 points** to the other one, while
costing the studio folds nothing. For scale, every other change we measured on this
project is worth single digits: cross-corpus pre-training 11 points, doubling the number
of clips 2.9, rotation augmentation 0.6.

**L4 — Privacy and bandwidth.** Any design that streams webcam video of a person's face
and home to a server is both a privacy problem and a bandwidth problem (4–8 Mbps).

**L5 — Continuous signing is unsolved.** Segmenting and translating fluent connected
signing remains an open research problem [8, 9]. A minor project that claims to solve it
is not being honest.

**L6 — Vocabulary availability.** Of the 50 signs our own specification originally chose,
the public corpus covered **3**. Vocabulary has to be chosen from what data exists, not
from what would be convenient.

### 4.3 Scope boundary

SignSight v1 recognises **isolated signs**, one at a time, and assembles them into
sentences with deterministic templates. It does not attempt continuous sign language
translation. This is stated as a limitation rather than discovered as a failure.

---

## 5. Project Planning and Proposal

### 5.1 Objectives

| # | Objective | Measure of success |
|---|---|---|
| O1 | Recognise a working ISL vocabulary live from a webcam | ≥85% top-1, signer-independent (current: 76.1%) |
| O2 | Never transmit video | Only 261 floats per frame leave the browser |
| O3 | Respond fast enough to feel live | <600 ms from end of sign to displayed gloss |
| O4 | Produce grammatical English, not glosses | Template assembly from the vocabulary pack |
| O5 | Speak the output aloud | Browser text-to-speech |
| O6 | Caption a signer in a video call | Chrome extension overlay |
| O7 | Report accuracy honestly | Leave-one-signer-out cross-validation, always |

### 5.2 Proposed architecture

```
Browser                              Local server
------------------------------       ---------------------------------
camera -> MediaPipe Holistic         ring buffer
       -> 261 numbers / frame   ->   motion-energy segmenter
       -> WebSocket                  BiLSTM classifier (ONNX)
                                     confidence gate + repeat cooldown
       <- gloss / transcript    <-    template assembler
```

**The design decision the whole project rests on:** landmarks are extracted in the browser
and only the coordinates travel. That is about **0.33 Mbps against 4–8 Mbps for video**,
it removes the privacy objection entirely, and it is what makes real-time performance
achievable on a laptop CPU.

**Feature vector (261 values per frame, frozen specification):**

| Block | Points | Values |
|---|---|---|
| Upper-body pose | 25 | 75 |
| Left hand | 21 | 63 |
| Right hand | 21 | 63 |
| Face (reduced set) | 20 | 60 |

Normalised against shoulder width and midpoint, making the representation invariant to how
far the signer sits from the camera and where they sit in frame.

**Classifier:** bidirectional LSTM, 128→64 units, 587k parameters, input a 45-frame
resampled segment, exported to ONNX. Measured **12.3 ms per inference** on CPU, 2.4 MB.

### 5.3 Milestones and current state

| Milestone | Definition of done | State |
|---|---|---|
| M0 Skeleton | Backend, app, extension shell, CI | **Done** |
| M1 Landmark pipeline | Browser and Python agree to <1e-6; 15 FPS sustained | **Done** — agree to 8.9e-16 |
| M2 Data | Signer-disjoint manifest, coverage per class | **Done** — 1,001 clips, 24 signs, 10 signers |
| M3 Model | ≥85% signer-disjoint; ONNX matches Keras; <25 ms | **Partial** — 76.1%, ONNX drift 1.6e-07, 12.3 ms |
| M4 Live recognition | Signs recognised live from the camera | **Done** — classifier, gating and assembly wired in |
| M5 Speaker app | Transcript, speech, reference sheet | In progress |
| M6 Listener extension | Overlay captions in a video call | Shell only |
| M7 Evaluation | Confusion matrix, latency, ablations | Mostly done |

### 5.4 Evaluation protocol — the methodological commitment

Every accuracy figure in this project is produced by **leave-one-signer-out
cross-validation**. The model is trained once per person in the corpus, each time with
that person removed entirely, and scored on them.

This is not the convenient choice. Our first reported figure, from a single
signer-disjoint split, was 52.0%; proper cross-validation of the same system returned
64.5%. A random split would have reported higher still and meant nothing [10].

**Hard rule, enforced by an automated check:** the training script refuses to run on a
manifest whose splits are not signer-disjoint.

### 5.5 Current results

| Metric | Value |
|---|---|
| Vocabulary | 24 ISL signs |
| Corpus | 1,001 clips · 10 signers · INCLUDE + our own recordings |
| **Accuracy (leave-one-signer-out, 8 folds)** | **76.1% ± 7.2%** (chance = 4%) |
| Precision at the 0.75 confidence gate | 82% of what the system chooses to say |
| Inference latency | 12.3 ms per segment, CPU |
| Feature parity, browser vs Python | 8.9e-16 max absolute difference |

### 5.6 Division of work

| Member | Area |
|---|---|
| **Eashan Singh** | Feature extraction, training, evaluation, data pipeline |
| **Krish** | Backend recognition path, streaming, storage |
| **Arpit** | Speaker app and Chrome extension front ends |

### 5.7 Risks and mitigation

| Risk | Mitigation |
|---|---|
| Accuracy short of the 85% target | Identified as a **data** limit, not an architecture limit — four ablations (capacity, velocity features, longer fine-tuning, more pre-training) each failed to help, while cross-corpus pre-training gained 11 points |
| The demo signer is unseen by the model | Quantified in §4.2: all three members must record. Two of three complete |
| Recording quality | A measured failure — hands leaving frame cost two-thirds of hand tracking in the first 505 clips. The recorder now draws the safe area and reports the tracking rate per signer |
| None of the team signs ISL | Every clip is recorded copying a reference video from the corpus; the vocabulary is restricted to signs with reference footage |
| Fingerspelling untrained | 26 letter classes have no data. Declared out of scope for v1 rather than faked |

### 5.8 Remaining plan

1. Complete the third signer's recordings, re-run cross-validation.
2. Finish the Speaker app: transcript view, text-to-speech, reference sheet (M5).
3. Chrome extension screen capture and overlay (M6).
4. End-to-end latency measurement, p50 and p95 (M7).
5. Final report and demonstration.

---

## 6. References

1. Indian Sign Language Research and Training Centre (ISLRTC), Department of
   Empowerment of Persons with Disabilities, Government of India.
   https://islrtc.nic.in/about-us/

2. "With a deaf community of millions, hearing India is only just beginning to sign,"
   *The World* (PRX), 2017.
   https://theworld.org/stories/2017/01/03/deaf-community-millions-hearing-india-only-just-beginning-sign

3. Indian Sign Language Research and Training Centre, certified interpreter statistics.
   https://islrtc.nic.in/about-department/about-islrtc/

4. I. Grishchenko and V. Bazarevsky, "MediaPipe Holistic — Simultaneous Face, Hand and
   Pose Prediction, on Device," Google AI Blog, December 2020. Provides a unified
   topology of 540+ keypoints (33 pose, 21 per hand, 468 facial) in real time on device.
   https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/

5. A. Sridhar, R. G. Ganesan, P. Kumar and M. Khapra, "INCLUDE: A Large Scale Dataset for
   Indian Sign Language Recognition," in *Proc. 28th ACM International Conference on
   Multimedia (MM '20)*, Seattle, 2020. https://dl.acm.org/doi/10.1145/3394171.3413528

6. A. Joshi, A. Bhat, et al., "CISLR: Corpus for Indian Sign Language Recognition," in
   *Proc. EMNLP 2022*, Abu Dhabi, pp. 10357–10366.
   https://aclanthology.org/2022.emnlp-main.707/

7. G. Velmathi and K. Goyal, "Indian Sign Language Recognition Using Mediapipe Holistic,"
   arXiv:2304.10256, 2023. https://arxiv.org/abs/2304.10256

8. U. Damdoo et al., "An integrative survey on Indian sign language recognition and
   translation," *IET Image Processing*, 2025.
   https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70000

9. "A comprehensive survey on recent advances and challenges in sign language recognition
   systems," *Discover Artificial Intelligence*, Springer, 2025.
   https://link.springer.com/article/10.1007/s44163-025-00629-7

10. "Rethinking Sign Language Translation: The Impact of Signer Dependence on Model
    Evaluation," arXiv:2609.07965, 2026. https://arxiv.org/abs/2609.07965

11. "SignAll SDK: Sign language interface using MediaPipe is now available for
    developers," Google Developers Blog.
    https://developers.googleblog.com/signall-sdk-sign-language-interface-using-mediapipe-is-now-available-for-developers/

12. M. J. Ahmed et al., "A Review on Systems-Based Sensory Gloves for Sign Language
    Recognition State of the Art between 2007 and 2017," *Sensors*, 2018.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6069389/

---

### Project documents

- Specification: [`SignSight_PRD.md`](SignSight_PRD.md)
- Full results, ablations and failure analysis: [`results.md`](results.md)
- Data strategy and corpus selection: [`datasets.md`](datasets.md)
- Recording protocol: [`recording.md`](recording.md)
