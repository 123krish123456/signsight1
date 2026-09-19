# Recording plan — Eashan, Krish, Arpit

## Why we are recording, given we already have 496 clips

Not for volume. The learning curve says another doubling of clips is worth 1–2 points
([`results.md`](results.md)). What it also says is that accuracy swings **15.7 points**
depending on which signer is held out, against 2.9 points for doubling the data.

The model has seen eight people. It cannot yet tell what varies between signers from what
is the sign itself. **Three new people is the most valuable thing we can add**, and it is
the one thing no dataset download gives us, because it also fixes the other known gap:
every clip we currently train on comes from INCLUDE's camera, lighting and room. None
comes from ours.

## The split

**All three of us record all 24 signs.** Do not divide the vocabulary between us — that
makes signer-disjoint evaluation impossible, because the held-out person would have no
examples of the signs the other two recorded.

| Who | Signer id | Signs | Clips each | Total |
|---|---|---|---|---|
| Eashan | `eashan` | all 24 | 10 | 240 |
| Krish | `krish` | all 24 | 10 | 240 |
| Arpit | `arpit` | all 24 | 10 | 240 |

**720 clips, roughly 45 minutes each.** That takes the signer pool from 8 to 11 and
nearly doubles the training set.

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
python -m ml.data.ingest videos ml/data/clips --source self     --signer-pattern '^([^/]+)/' --no-letters
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

**Watch the reference two or three times before your first take of a sign.** A wrong
gesture labelled with a real word is worse than no clip at all — it teaches the model
something false, and nothing in the pipeline can detect it.

**Mirror the reference, do not copy it left-to-right.** Your camera view is flipped like
a mirror, so if the reference signer uses their right hand, you use yours. It will look
like the opposite side on screen. That is correct.

**Frame yourself from the waist up**, both hands free to move without leaving frame, face
visible. The face matters: removing those landmarks costs 9 points of accuracy.

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
python -m ml.data.ingest videos ml/data/clips --source self     --signer-pattern '^([^/]+)/' --no-letters
```

## When everyone is done

```bash
python -m ml.data.manifest --assign --check       # 11 signers now
python -m ml.data.precompute --workers 6          # ~6 min for 720 new clips
python -m ml.crossval --pack backend/vocab/isl_v2_words.json \
    --init-from ml/models/include_pretrain_long.keras
```

Two numbers are worth watching, and the second matters more:

1. **Does the cross-validated mean rise** above 76.1%? Probably a little.
2. **What accuracy do the three new folds get?** Those are the first measurements ever
   taken on our own camera and lighting. If they come in far below the INCLUDE folds, the
   domain gap is real and quantified — which is a genuine finding for the report, not a
   failure.
