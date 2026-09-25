/** Draws MediaPipe landmarks over a camera preview.
 *
 *  Its job is to answer one question while you record: is the tracker seeing my hands?
 *  The first 505 clips this project collected lost two thirds of their hand tracking to
 *  hands sitting below the bottom of the frame, and nobody noticed, because from the
 *  chair everything looked fine. A skeleton on the preview makes that visible the moment
 *  it happens instead of a week later in a report.
 *
 *  Pure drawing: takes a result and a context, holds no state, knows nothing about React.
 */

import type { HolisticLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

/** MediaPipe's 21-point hand topology: wrist, then five fingers, plus the palm arch. */
const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],            // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],            // index
  [5, 9], [9, 10], [10, 11], [11, 12],       // middle
  [9, 13], [13, 14], [14, 15], [15, 16],     // ring
  [13, 17], [17, 18], [18, 19], [19, 20],    // little
  [0, 17],                                   // palm
];

/** Upper body only — the same 25 pose points the feature spec keeps. */
const POSE_BONES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
];

const LEFT = "#38bdf8";
const RIGHT = "#fbbf24";
const BODY = "#64748b";

function bones(
  ctx: CanvasRenderingContext2D,
  points: NormalizedLandmark[],
  pairs: [number, number][],
  colour: string,
  w: number,
  h: number,
  width: number,
) {
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const [a, b] of pairs) {
    const p = points[a];
    const q = points[b];
    if (!p || !q) continue;
    ctx.moveTo(p.x * w, p.y * h);
    ctx.lineTo(q.x * w, q.y * h);
  }
  ctx.stroke();
}

function dots(
  ctx: CanvasRenderingContext2D,
  points: NormalizedLandmark[],
  colour: string,
  w: number,
  h: number,
  radius: number,
) {
  ctx.fillStyle = colour;
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** One frame of skeleton. Returns which hands were drawn, so the caller can report it. */
export function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  result: HolisticLandmarkerResult,
  w: number,
  h: number,
): [boolean, boolean] {
  ctx.clearRect(0, 0, w, h);

  // Scale with the preview so the skeleton stays legible on a small window and does not
  // swamp the picture on a large one.
  const scale = Math.max(1, Math.min(w, h) / 480);

  const pose = result.poseLandmarks?.[0];
  if (pose) {
    bones(ctx, pose, POSE_BONES, BODY, w, h, 3 * scale);
    // Wrists specifically: when a hand goes untracked, the wrist is what tells you
    // whether it left the frame or was simply not resolved.
    dots(ctx, [pose[15], pose[16]].filter(Boolean), BODY, w, h, 4 * scale);
  }

  const left = result.leftHandLandmarks?.[0];
  const right = result.rightHandLandmarks?.[0];

  for (const [points, colour] of [[left, LEFT], [right, RIGHT]] as const) {
    if (!points?.length) continue;
    bones(ctx, points, HAND_BONES, colour, w, h, 2.5 * scale);
    dots(ctx, points, colour, w, h, 3 * scale);
  }

  return [!!left?.length, !!right?.length];
}
