# Front-end guide — Arpit

Everything you need to build your part without waiting for the model.

## What SignSight is, in four lines

A deaf or mute person signs at their webcam. The browser tracks their hands, face and
upper body, and sends **only the coordinates** — never the video — to a local Python
server. The server works out where each sign starts and stops, recognises it, and builds
an English sentence. Two front ends use that: a **Speaker app** the signer uses to be
heard out loud, and a **Chrome extension** that captions a signer during a video call.

Both front ends are yours.

## The one thing to understand first

**You are not blocked by the missing model.** The server already sends the exact events
the finished system will send — there is a mock recogniser that fabricates realistic
glosses and sentences. Build against those today; when the real model lands, nothing in
your code changes, because the event shapes are identical.

## Setup (15 minutes)

```bash
pip install -e ".[dev]"
cd app && npm install && npm run fetch-assets      # downloads ~46 MB of tracking models
```

Two terminals, and note the environment variable — it is what turns the mock on:

```bash
# terminal 1 — backend with fake recognition
SIGNSIGHT_MOCK_RECOGNITION=true uvicorn backend.main:app --reload
#   Windows PowerShell:
#   $env:SIGNSIGHT_MOCK_RECOGNITION="true"; uvicorn backend.main:app --reload

# terminal 2 — the app
cd app && npm run dev
```

Open http://127.0.0.1:5173, click **Start camera**, allow the webcam, then **move your
hands and stop**. Each time you move and settle, the server decides a sign happened and
sends you a gloss. After a few, you get a full sentence:

```
gloss       ME     conf 0.94
gloss       NAME   conf 0.83
gloss       E · A · S · H · A · N
TRANSCRIPT  "My name is Eashan."
```

You don't need to know any sign language to test — any hand movement triggers it.

## The events you receive

The socket is already wired up in `app/src/socket.ts`. You handle these four:

| Event | When | What you do with it |
|---|---|---|
| `{type:"state", value:"IDLE"\|"SIGNING"}` | Movement starts/stops | The badge. Already built. |
| `{type:"gloss", value, confidence, segment_ms}` | A sign was recognised | Append to the gloss strip |
| `{type:"transcript", text, is_final}` | A sentence completed | **Show it, and speak it** |
| `{type:"error", code, message}` | Something broke | Show the message |

`value: "UNKNOWN"` means a sign was detected but not understood — the mock returns it
about 8% of the time on purpose. Render it as `…`, never hide it: the user has to be
able to tell "I wasn't understood" from "it didn't see me". That distinction is a
requirement, not a nicety.

## Your tasks

**The speaker app is done** — transcript view, speech output and the sign reference sheet
are all in `app/src/App.tsx` and working. Tasks 1 to 3 of the original list came off your
plate on 25 September.

**What is left is the Chrome extension (M6), and it is the last substantial piece of the
whole project.** It has its own walkthrough, written for someone who has not touched
Manifest V3 before:

> **[`docs/extension-guide.md`](extension-guide.md)**

Short version: a service worker has no DOM and cannot touch video, so capture happens in
a hidden "offscreen document". `offscreen.html` and `offscreen.js` do not exist yet and
are the work. Everything around them — the socket, the overlay, the vendored MediaPipe
files — is already written.

## Rules

- **Never send video over the socket.** Coordinates only. It is the decision that makes
  this run in real time — 0.33 Mbps instead of 8.
- **Do not touch `app/src/normalise.ts`.** It is mathematically locked to the Python
  training code and a test fails if they disagree by more than a rounding error. If you
  genuinely need a change there, talk to Eashan.
- `npm run build` and `pytest` both pass before you push.
- Turn the mock **off** for any real demo — it does not look at the camera at all.

## When you're stuck

`docs/SignSight_PRD.md` is the full specification. §5.1 is the event contract, §6 is the
whole extension design, §4.5 explains the confidence rules. Your parts are M5 and M6 in
the build order at §7.
