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

## How to do it — in the browser

No Python needed. Start both servers, then open **http://127.0.0.1:5173/record** and
click your own tab.

```bash
uvicorn backend.main:app --reload     # terminal 1
cd app && npm run dev                 # terminal 2
```

The tabs exist so nobody records half a session under a different spelling of their name
— the signer id is the evaluation's split key, and a typo invents a phantom person.

Press **Record** or the spacebar: 1.5 s countdown, 3 s clip, saved automatically, then it
moves to the sign you have fewest of. `u` undoes, `n` skips. Tick **keep going
automatically** to record continuously.

## Or on the desktop

```bash
pip install -e ".[ml]"
python -m ml.data.record --signer <yourname> --target 10
```

The window shows **your camera on the left and the reference sign on the right**, looping.
None of us knows Indian Sign Language, so copy what the reference shows. It comes from
INCLUDE — the same corpus the model trains on — so copying it is exactly right.

| Key | Action |
|---|---|
| `SPACE` | 1.5 s countdown, then records 3 s |
| `N` | skip to another sign |
| `U` | undo the last clip (deletes the file too) |
| `Q` | quit — progress is saved as you go |

The tool always offers whichever sign you have fewest clips of, so just keep pressing
SPACE and coverage stays even. Stop and resume whenever; it picks up where you left off.

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
