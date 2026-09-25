# Building the Listener extension — a walkthrough for Arpit

This is the last substantial piece of SignSight. Everything else is finished.

**What you are building:** a hearing person is on a Google Meet call with a deaf person
who signs. They click your extension, point it at the signer's video tile, and captions
appear over the call.

Budget a weekend. Most of it is one file.

---

## Before anything: what already exists

Less is missing than it looks. Open `extension/` and you will find:

| File | State |
|---|---|
| `manifest.json` | Done. Permissions already declared, including `offscreen` |
| `background.js` | Done. Owns the WebSocket to the backend, relays events |
| `content.js` | Done. Draggable caption overlay with opacity slider |
| `popup.html` | Done. Two buttons |
| `popup.js` | **Half.** "Connect" works. "Select signer's video…" is a stub |
| `vendor/` | Done. MediaPipe wasm, the model file, and the library |
| **`offscreen.html`** | **Missing — you write this** |
| **`offscreen.js`** | **Missing — you write this. It is the real work** |

So: finish one button, and write the piece in the middle that turns pixels into
landmarks.

### Get set up

```bash
git fetch origin && git reset --hard origin/main   # history was rewritten; do this once
cd app && npm install && npm run fetch-assets
```

`fetch-assets` is not optional. It downloads the 13.7 MB tracking model and copies
MediaPipe into `extension/vendor/`. Those files are gitignored, so cloning does not give
them to you.

Then load the extension: `chrome://extensions` → **Developer mode** on → **Load
unpacked** → pick the `extension/` folder.

And start the backend in a terminal, which you will need running throughout:

```bash
uvicorn backend.main:app --reload
```

Click the extension icon, hit **Connect to backend**. If it says
`Connected · vocab isl_v2_words · model signsight_isl24.onnx`, everything on the other
side of your work is functioning.

---

## The one concept you need

**A Chrome MV3 service worker has no DOM.** No `document`, no `<video>`, no `<canvas>`.
It cannot touch media at all. This is not a limitation you can work around — it is how
MV3 works.

So Chrome gives you an **offscreen document**: a hidden HTML page your service worker
creates on demand, purely so something in your extension has a DOM. That is the only
reason `offscreen.html` exists.

The whole data path:

```
popup.js         user picks a screen/tab  →  a MediaStream
    ↓            (hands the stream id over)
offscreen.js     video → crop to region → MediaPipe → 261 numbers
    ↓            (chrome.runtime.sendMessage)
background.js    already written — pushes them down the WebSocket
    ↓
backend          already written — segments, classifies, builds sentences
    ↓
content.js       already written — draws the captions
```

You are writing the second row. That is it.

---

## Step 1 — Finish the popup

Right now `popup.js` has:

```js
document.getElementById("capture").onclick = () => {
  status.textContent = "Region capture arrives in M6.";
};
```

Replace it with a real picker. `chooseDesktopMedia` gives you a **stream id**, not a
stream — you pass that id to the offscreen document, which turns it into video.

```js
document.getElementById("capture").onclick = () => {
  chrome.desktopCapture.chooseDesktopMedia(
    ["screen", "window", "tab"],
    async (streamId) => {
      if (!streamId) return;                       // user cancelled
      await chrome.runtime.sendMessage({ type: "start-capture", streamId });
      status.textContent = "Capturing. Pick the signer's tile.";
    },
  );
};
```

Then in `background.js`, create the offscreen document when that message arrives:

```js
async function ensureOffscreen() {
  const existing = await chrome.offscreen.hasDocument?.();
  if (existing) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Runs hand and pose tracking on the captured video.",
  });
}
```

and route `start-capture` to it. Add a branch to the existing
`chrome.runtime.onMessage.addListener` — do not write a second listener.

> **Trap.** `chooseDesktopMedia` only works from the popup or the service worker, and the
> stream id it returns is **single-use and expires in seconds**. Hand it over
> immediately; do not stash it for later.

---

## Step 2 — `offscreen.html`

Tiny. It exists to host a script and two elements nobody sees.

```html
<!doctype html>
<meta charset="utf-8" />
<video id="src" autoplay muted playsinline></video>
<canvas id="crop"></canvas>
<script type="module" src="offscreen.js"></script>
```

`type="module"` matters — you are importing MediaPipe as an ES module.

---

## Step 3 — `offscreen.js`, the real work

**Copy the tracking setup from `app/src/landmarks.ts`.** It works, it is tested, and the
only differences are how you load files and where frames come from.

### Turning the stream id into video

```js
const media = await navigator.mediaDevices.getUserMedia({
  video: { mandatory: { chromeMediaSource: "desktop", chromeMediaSourceId: streamId } },
});
document.getElementById("src").srcObject = media;
```

That `mandatory` block is old non-standard syntax. It is still the only thing that works
for desktop capture. Do not modernise it.

### Loading MediaPipe — where people lose an afternoon

**MV3 forbids remote code.** The MediaPipe examples all fetch wasm and the model from a
CDN. Do that and it fails silently — no error you would recognise, just nothing
happening. PRD §6.2 calls this out for a reason.

Everything is already vendored locally. Load it from disk:

```js
import { FilesetResolver, HolisticLandmarker } from "./vendor/vision_bundle.mjs";

const fileset = await FilesetResolver.forVisionTasks(
  chrome.runtime.getURL("vendor/wasm"),
);
const landmarker = await HolisticLandmarker.createFromOptions(fileset, {
  baseOptions: {
    modelAssetPath: chrome.runtime.getURL("vendor/models/holistic_landmarker.task"),
    delegate: "GPU",
  },
  runningMode: "VIDEO",
});
```

If you ever see a URL starting `https://cdn.jsdelivr.net` in your code, that is the bug.

### The crop

The user picked a rectangle over the signer's tile. Store it as **fractions of the source
size, never pixels** — a captured window changes size and pixel coordinates silently
start pointing at the wrong place.

```js
// region = { x: 0.25, y: 0.10, w: 0.30, h: 0.40 }
const sx = region.x * video.videoWidth;
const sy = region.y * video.videoHeight;
const sw = region.w * video.videoWidth;
const sh = region.h * video.videoHeight;

canvas.width = sw;
canvas.height = sh;
ctx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
```

Feed `canvas` to MediaPipe, not `video`. Cropping first means the hands fill more of the
frame, and hand detection depends heavily on that — see below.

### The loop

Copy the rAF gating from `app/src/landmarks.ts`. **15 FPS, not 60.** The backend expects
it, and 30 doubles the work for no gain on signs lasting half a second to two seconds.

```js
const result = landmarker.detectForVideo(canvas, performance.now());
const vector = normaliseFrame(pose, leftHand, rightHand, face);   // see below
chrome.runtime.sendMessage({ type: "frame", landmarks: Array.from(vector),
                             hands_present: [!!leftHand, !!rightHand] });
```

`background.js` already handles `type: "frame"` and adds `seq` and `t_client_ms`. You do
not need to.

### Normalisation — do not write your own

The 261 numbers are a frozen specification. `app/src/normalise.ts` implements it, and a
test (`tests/test_parity.py`) fails if it drifts from the Python training code by more
than a rounding error. If your extension normalises even slightly differently, accuracy
collapses and **every test still passes.**

So do not reimplement it. Copy `app/src/normalise.ts` to
`extension/vendor/normalise.js` — strip the TypeScript types, change nothing else — or
add a small build step that compiles it. Either is fine. Writing a second implementation
is not.

---

## How to tell it is working

In order. Do not skip ahead — each step tells you something different.

1. **`chrome://extensions` → your extension → "Inspect views: offscreen.html".** That
   gives you a console for the offscreen document. You will live here.
2. **Log the video dimensions** once the stream arrives. If they are `0x0`, the stream id
   expired before you used it.
3. **Log `result.leftHandLandmarks`.** If it is always empty, MediaPipe loaded but is not
   finding hands — usually the crop region is wrong. Draw the canvas somewhere visible.
4. **Watch the backend terminal.** It logs the frame rate every five seconds. You want
   about 15 FPS with `lost: 0`.
5. **Sign at it.** The terminal prints `SIGNING` / `IDLE` and then a gloss per segment.
6. **Open a Meet call** and check captions appear over it.

### Test without a second person

You do not need a real call to develop this. Open the speaker app
(`http://127.0.0.1:5173`) in one window, capture that window with the extension, and sign
at your own webcam. The extension will caption you.

**Or** play a reference clip and capture that:

```bash
cd app && npm run dev
# then open http://127.0.0.1:5173/reference/HELLO.mp4
```

Point the extension at it. The clip is a real ISL signer from the training corpus, so it
should recognise reliably — which makes it a good way to tell "my extension is broken"
apart from "my signing is unclear".

---

## Two things that will bite you

**1. Framing decides everything.** We measured this the hard way. When hands leave the
edge of the frame, MediaPipe reports nothing — 92% of all our failed hand detections were
exactly that. Your crop rectangle is doing the same job as camera framing: if the user
draws it tight around the signer's face, the extension will appear broken and it will not
be your code. Consider nudging the region outward by 10% automatically, and say
"include their whole upper body" in the UI.

**2. Accuracy is 76%, and lower for someone the model has not seen.** When it gets a sign
wrong, that is expected, not a bug in your code. A `…` in the overlay means the model
detected a sign but was not confident enough to name it. That is deliberate — the user
must be able to tell "not understood" from "not signing."

---

## Rules

- **Never send video or images over the socket.** Coordinates only. That decision is what
  makes the whole thing real time — 0.33 Mbps instead of 8.
- **Never fetch code or models from the internet at runtime.** MV3 forbids it and it will
  fail in a way that is hard to diagnose.
- **Do not modify `app/src/normalise.ts`.** Talk to Eashan if you think you need to.
- **Keep the "SignSight — automated, may contain errors" label visible** in the overlay.
  It is not decoration. We are putting words in a deaf person's mouth and a viewer has to
  know a machine wrote them. The report discusses this.
- Run `pytest` before you push. CI runs it anyway.

## Done when

A live Google Meet call with a signer on another machine produces overlay captions
matching what the speaker app would say for the same signs. That is the M6 definition of
done in PRD §7.

## If you get stuck

- `docs/SignSight_PRD.md` §6 is the full extension design, §5.1 is the event contract.
- `app/src/landmarks.ts` is the working version of the hard part.
- `docs/frontend-guide.md` has the wider context and the event shapes.
