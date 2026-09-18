/** MediaPipe Tasks — Holistic Landmarker wrapper (PRD §4.1). Runs at 15 FPS by design:
 *  30 FPS doubles compute for no accuracy gain on signs lasting 0.5–2 s. */

import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { normaliseFrame, type Point } from "./normalise";

/** Vendored by `npm run fetch-assets` — never fetched from a CDN at runtime, so the
 *  demo survives bad wifi and the same assets drop straight into extension/vendor. */
const WASM_DIR = "/wasm";
const MODEL_PATH = "/models/holistic_landmarker.task";

export interface FrameResult {
  vector: Float64Array;
  handsPresent: [boolean, boolean];
  raw: HolisticLandmarkerResult;
}

export class LandmarkStream {
  private landmarker: HolisticLandmarker | null = null;
  private raf = 0;
  private lastSent = 0;
  private lastTimestamp = -1;
  private readonly minIntervalMs: number;

  constructor(fps = 15) {
    this.minIntervalMs = 1000 / fps;
  }

  async init(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);
    this.landmarker = await HolisticLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
    });
  }

  /** Drives detection off rAF, gated to the target FPS. Returns a stop function. */
  start(video: HTMLVideoElement, onFrame: (f: FrameResult) => void): () => void {
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      const now = performance.now();
      if (now - this.lastSent < this.minIntervalMs) return;
      if (!this.landmarker || video.readyState < 2) return;
      this.lastSent = now;

      // detectForVideo rejects non-monotonic timestamps.
      const ts = Math.max(now, this.lastTimestamp + 1);
      this.lastTimestamp = ts;
      const res = this.landmarker.detectForVideo(video, ts);

      // Results are per-detected-person; holistic is single-person, so take [0].
      const first = (blocks: NormalizedLandmark[][]): Point[] | null => blocks?.[0] ?? null;
      const left = first(res.leftHandLandmarks);
      const right = first(res.rightHandLandmarks);
      onFrame({
        vector: normaliseFrame(first(res.poseLandmarks), left, right, first(res.faceLandmarks)),
        handsPresent: [!!left?.length, !!right?.length],
        raw: res,
      });
    };
    this.raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(this.raf);
  }

  close(): void {
    cancelAnimationFrame(this.raf);
    this.landmarker?.close();
    this.landmarker = null;
  }
}
