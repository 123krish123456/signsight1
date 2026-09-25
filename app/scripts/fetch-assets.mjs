/**
 * Vendors the MediaPipe WASM bundle + holistic model into app/public/ and
 * extension/vendor/. PRD §6.2: MV3 forbids remote code, so the extension MUST load
 * these locally — and doing it for the app too means the demo works offline.
 *
 * Run: npm run fetch-assets
 */
import { cp, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const APP = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const WASM_SRC = resolve(APP, "node_modules/@mediapipe/tasks-vision/wasm");
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

// The library itself, not just its wasm. Vite bundles this into the app from
// node_modules, but the extension has no build step: offscreen.js is a plain ES module
// loaded by Chrome, so it needs the file sitting next to it to import by relative path.
// Without this the extension cannot run MediaPipe at all, and the failure looks like a
// module resolution error rather than a missing asset.
const LIB_SRC = resolve(APP, "node_modules/@mediapipe/tasks-vision/vision_bundle.mjs");
const LIB_DEST = resolve(APP, "../extension/vendor/vision_bundle.mjs");

const targets = [
  { wasm: resolve(APP, "public/wasm"), model: resolve(APP, "public/models") },
  { wasm: resolve(APP, "../extension/vendor/wasm"), model: resolve(APP, "../extension/vendor/models") },
];

const exists = (p) => stat(p).then(() => true, () => false);

if (!(await exists(WASM_SRC))) {
  console.error(`missing ${WASM_SRC} — run npm install first`);
  process.exit(1);
}

console.log("downloading holistic_landmarker.task (~10 MB)…");
const res = await fetch(MODEL_URL);
if (!res.ok) {
  console.error(`model download failed: ${res.status} ${res.statusText}`);
  process.exit(1);
}
const model = Buffer.from(await res.arrayBuffer());

for (const t of targets) {
  await mkdir(t.model, { recursive: true });
  await cp(WASM_SRC, t.wasm, { recursive: true });
  await writeFile(resolve(t.model, "holistic_landmarker.task"), model);
  console.log(`vendored → ${t.wasm}, ${t.model}`);
}
await cp(LIB_SRC, LIB_DEST);
console.log(`vendored → ${LIB_DEST}`);

console.log(`done (${(model.length / 1e6).toFixed(1)} MB model)`);
