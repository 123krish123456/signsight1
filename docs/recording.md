# Recording plan — Eashan, Krish, Arpit

## Why we are recording — now with the measurement behind it

Krish and Arpit have recorded. Their 505 clips made the answer to this question much
sharper than it was when the plan was written.

Held out one at a time, the model reads them at **68.2%** and **71.3%**. Remove the
*other* one's clips from training and those become **38.9%** and **29.4%**. The six
remaining signers — 1920x1080, studio lighting, a tripod — are not enough on their own to
read a person sitting at a laptop, whatever their number.

Nothing else measured on this project comes close. Pre-training on 179 extra classes was
worth 11 points, doubling the clip count 2.9, rotation augmentation 0.6. **Being in the
same domain as the training data is worth 30 to 40.**

So this is not about the signer pool growing from 8 to 11. It is that whoever signs at the
demo, on the laptop they demo with, has to be in the training set — otherwise the model is
in the 30% regime for them and no amount of public data fixes it.

## The split

**All three of us record all 24 signs.** Do not divide the vocabulary between us — that
makes signer-disjoint evaluation impossible, because the held-out person would have no
examples of the signs the other two recorded.

| Who | Signer id | Signs | Clips | State |
|---|---|---|---|---|
| Krish | `krish` | all 24 | 240 | done — hands tracked in 61% of frames |
| Arpit | `arpit` | all 24 | 265 | done — hands tracked in 35% of frames |
| Eashan | `eashan` | all 24 | 240 | **outstanding** |

Both finished batches are usable but weak, for one reason covered in the next section.
Five clips per sign — about 20 minutes — captures most of the value if time is short: the
gain comes from being a new person on a new camera, not from clip count.

Use your own name as the signer id and keep it identical every session. Splits are
assigned by signer, so a typo creates a phantom twelfth person and quietly weakens the
evaluation.

## The 24 signs

```
HELLO  THANK-YOU  GOOD-MORNING  HOW-ARE-YOU  ALRIGHT  PLEASED
ME  YOU  HE  SHE  WE
MOTHER  FATHER  FRIEND  MAN  WOMAN
HAPPY  SICK  HEALTHY  BIG  SMALL  COLD
HOUSE  SCHOOL
```

## How to do it

You need **Node only** — no Python, no backend. Clone the repo and:

```bash
cd app
npm install
npm run dev
```

Open **http://127.0.0.1:5173/record**, click **your own tab**, then **Choose a folder and
start**. Pick any folder on your laptop; clips are written straight into it and nothing
is uploaded anywhere.

> **It has to be `localhost` or `127.0.0.1`.** Browsers only expose the camera on a secure
> origin, and a plain-HTTP address like `http://192.168.1.3:5173` is not one —
> `navigator.mediaDevices` there is not merely blocked, it does not exist. So you cannot
> record by pointing your laptop at someone else's machine over wifi.

The reference sign plays on the left, your camera on the right.

| Key | Action |
|---|---|
| `SPACE` or **Record** | 1.5 s countdown, records 3 s, saves, moves on |
| `u` | undo the last clip (deletes the file) |
| `n` | skip to another sign |

Tick **keep going automatically** to record continuously with a pause between clips. Stop
whenever — pick the same folder next time and it carries on from where you left off, since
it counts the files already there.

The tabs exist so nobody records half a session under a different spelling of their name:
the signer id is the evaluation's split key, and a typo invents a phantom person.

### Sending your clips over

Your folder ends up looking like this:

```
<the folder you picked>/
  HELLO/        eashan_HELLO_000.webm, eashan_HELLO_001.webm, ...
  THANK-YOU/    ...
  ...
```

Zip it and send it to Eashan however you like — Drive, WhatsApp, a USB stick. Roughly
100–250 MB for 240 clips.

Eashan: drop each person's folder into `ml/data/clips/<their-name>/` and rebuild the
manifest from the folder layout. Nothing needs merging by hand.

```bash
python -m ml.data.ingest videos ml/data/clips --source self \
    --signer-pattern '^([^/]+)/' --no-letters
```

### On the machine running the project

If you are on Eashan's laptop with the backend running, the recorder can write into the
repository's clips directory directly — it is behind "Send to the project backend
instead". Firefox and Safari have no folder API, so they always use this route.

### Or the desktop recorder

```bash
pip install -e ".[ml]"
python -m ml.data.record --signer <yourname> --target 10
```

## Getting it right

### Push your chair back. This is the one that actually went wrong.

Of the 505 clips recorded so far, two thirds of all hand tracking was lost, and the cause
is not subtle: for **92% of Arpit's and 95% of Krish's** undetected hands, the body model
still found a wrist and that wrist was at or past the edge of the picture. Both sat at
ordinary laptop distance, so the shot is head-and-shoulders and their hands drop out of
the bottom of the frame as they sign.

It is invisible while recording — you can see your own hands perfectly well, because your
eyes are not cropped to the webcam's field of view. So the recorder now draws the safe
area on the preview. **Both hands must stay inside that box for the whole clip.**

Check yourself after the first few:

```bash
python -m ml.data.precompute --report
```

Below 70% means re-frame and redo them. The studio corpus manages 86–92%.

**Watch the reference two or three times before your first take of a sign.** A wrong
gesture labelled with a real word is worse than no clip at all — it teaches the model
something false, and nothing in the pipeline can detect it.

**Mirror the reference, do not copy it left-to-right.** Your camera view is flipped like
a mirror, so if the reference signer uses their right hand, you use yours. It will look
like the opposite side on screen. That is correct.

**Keep your face visible.** Removing those landmarks costs 9 points of accuracy, and it
is easy to lose them by tilting the laptop screen back to get your hands in.

**Change something halfway.** Do five clips, then move to a different room or turn a light
on, and do the other five. Variation across conditions is the entire point of this
exercise; ten identical clips are worth much less than ten varied ones.

**Record at least some of it in the room we will demo in.** That is the only way we ever
learn what our real-world accuracy is, rather than our accuracy on someone else's camera.

**Be natural about speed.** Signs run 0.5–2 seconds. Do not hold a pose stiffly for the
full 3 seconds; sign it, then relax.

## Handing clips over

Clips are not committed to git — a few hundred megabytes of video would be baked into
every future clone. Krish and Arpit: zip your own folder and send it to Eashan.

```bash
# on your machine, after recording
ml/data/clips/<yourname>/          <- zip this folder and send it
```

Eashan: drop those folders into `ml/data/clips/` alongside your own, then rebuild the
manifest rows from the folder layout. There is nothing to merge by hand — the signer name
is the folder name.

```bash
python -m ml.data.ingest videos ml/data/clips --source self \
    --signer-pattern '^([^/]+)/' --no-letters
```

## When everyone is done

```bash
python -m ml.data.manifest --assign --check
python -m ml.data.precompute --workers 6
python -m ml.data.precompute --report             # tracking rate per signer
python -m ml.crossval --pack backend/vocab/isl_v2_words.json \
    --init-from ml/models/include_pretrain_long.keras
```

Check `--report` before anything else. A signer below 70% has a framing problem, and no
training run recovers what the camera never saw.

Then read the webcam folds rather than the mean. They are the only ones that predict what
the demo will do; the six studio folds measure a camera we will not be using.
