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
console.log(`done (${(model.length / 1e6).toFixed(1)} MB model)`);
