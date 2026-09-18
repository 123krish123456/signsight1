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

## Your tasks, in the order I'd do them

### 1. Transcript view — `app/src/App.tsx`

Right now finished sentences are thrown away. Keep a running list and show it.

**Done when:** you sign a few times and see a growing list of sentences, newest at the
bottom, with the gloss strip still visible above it.

### 2. Speech output — `app/src/tts.ts` (new file)

The signer's actual voice. Use the browser's built-in `speechSynthesis` — no library,
no API key.

```ts
const u = new SpeechSynthesisUtterance(text);
u.rate = 1.0;
speechSynthesis.speak(u);
```

Three things that will bite you:
- Speak **only** when `is_final` is true, or you'll stutter over partial sentences.
- Call `speechSynthesis.cancel()` when the user hits Stop, or it keeps talking.
- Voices load asynchronously — `getVoices()` is empty on first call. Listen for
  `voiceschanged` if you offer a voice picker.

**Done when:** signing produces "My name is Eashan." out of the laptop speakers, and
pressing Stop shuts it up mid-sentence.

### 3. Sign reference sheet — new component

Users can't use a 50-sign vocabulary they can't see. `GET http://127.0.0.1:8000/vocab`
returns every sign with its part of speech. Render it as a panel or modal, grouped by
`pos` — the 26 letters are the manual alphabet, the other 24 are words.

**Done when:** a first-time user can open the sheet and see what the system understands.
Fetch it once on load, not per render.

### 4. Extension: region selection — `extension/popup.js`

Now the Chrome half. The hearing person picks which part of their screen has the signer
in it.

- `chrome.desktopCapture.chooseDesktopMedia(["tab","window","screen"], cb)` gets a stream
- Show a still preview, let them drag a rectangle over the signer's video tile
- **Store the rectangle as fractions** (`{x:0.25, y:0.1, w:0.3, h:0.4}`), not pixels —
  pixels break the moment the window is resized

**Done when:** you can pick a region and it survives resizing the window.

### 5. Extension: offscreen document — `extension/offscreen.html` + `.js` (new)

This one has a trap. Chrome extensions' background workers **have no DOM**, so they
cannot touch video at all. Everything media-related happens in an "offscreen document" —
a hidden page the worker creates.

Flow: crop each frame to the chosen region on an `OffscreenCanvas` → feed the crop to
MediaPipe → send the landmarks to `background.js`, which owns the socket.

Copy the tracking setup from `app/src/landmarks.ts`; it already works. Load the model
files with `chrome.runtime.getURL("vendor/models/holistic_landmarker.task")` — **never
from a URL on the internet.** Extensions are forbidden from running remote code and will
silently fail. The files are already vendored in `extension/vendor/` for you.

**Done when:** the overlay captions a signer in a real Google Meet call.

### 6. Overlay polish — `extension/content.js`

Basic version exists: draggable, opacity slider, last 4 lines. Make it good.

**Keep the "SignSight — automated, may contain errors" label visible.** It is not
decoration — we are putting words in a deaf person's mouth, and a viewer has to know a
machine wrote them. This is discussed in the report.

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
