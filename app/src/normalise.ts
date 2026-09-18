/**
 * Raw MediaPipe landmarks → 261-d feature vector (PRD §4.2).
 *
 * BINDING SPEC — must stay bit-identical to `ml/features/extract.py`.
 * `tests/test_parity.py` runs both on the same fixtures and asserts < 1e-6 drift.
 * Change one, change the other, run the test.
 */

export const POSE_N = 25;
export const HAND_N = 21;
export const FACE_N = 20;

export const POSE_DIM = POSE_N * 3; // 75
export const HAND_DIM = HAND_N * 3; // 63
export const FACE_DIM = FACE_N * 3; // 60
export const FEATURE_DIM = POSE_DIM + HAND_DIM * 2 + FACE_DIM; // 261

const L_SHOULDER = 11;
const R_SHOULDER = 12;
const MIN_SHOULDER_WIDTH = 1e-6;

/** Appendix B — MediaPipe FaceMesh 468-point topology. */
export const FACE_LITE_IDX = [
  // outer lip contour (12)
  61, 291, 39, 181, 0, 17, 269, 405, 270, 314, 13, 14,
  // eyebrows (8)
  70, 63, 105, 66, 300, 293, 334, 296,
];

export interface Point {
  x: number;
  y: number;
  z: number;
}

export type Block = Point[] | null | undefined;

function sized(block: Block, n: number): Point[] | null {
  return block != null && block.length === n ? block : null;
}

/**
 * Face mesh → the 20 face-lite points. Accepts the 468-point mesh, the 478-point
 * mesh (MediaPipe Tasks appends 10 iris points; 0-467 are unchanged), or an
 * already-selected 20-point block. Mirrors `_select_face` in extract.py.
 */
function selectFace(face: Block): Point[] | null {
  if (face == null) return null;
  if (face.length >= 468) return FACE_LITE_IDX.map((i) => face[i]);
  if (face.length === FACE_N) return face;
  return null;
}

/**
 * One frame → 261 floats. Absent hand/face blocks are zeros, never interpolated
 * (absence is informative — one-handed signs exist). Missing or degenerate
 * shoulders make the whole frame invalid: an all-zero vector.
 */
export function normaliseFrame(
  pose: Block,
  leftHand: Block,
  rightHand: Block,
  face: Block,
): Float64Array {
  const out = new Float64Array(FEATURE_DIM);

  const p = sized(pose, 33) ?? sized(pose, POSE_N);
  if (!p) return out;

  const ls = p[L_SHOULDER];
  const rs = p[R_SHOULDER];
  const mid = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2, z: (ls.z + rs.z) / 2 };
  // x,y only — MediaPipe pose z is a noisy depth estimate and scaling every
  // feature by it spreads that noise. Matches extract.py deliberately.
  const scale = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  if (scale < MIN_SHOULDER_WIDTH) return out;

  const write = (block: Point[], at: number, count: number) => {
    for (let i = 0; i < count; i++) {
      const q = block[i];
      out[at + i * 3] = (q.x - mid.x) / scale;
      out[at + i * 3 + 1] = (q.y - mid.y) / scale;
      out[at + i * 3 + 2] = (q.z - mid.z) / scale;
    }
  };

  write(p, 0, POSE_N);

  const lh = sized(leftHand, HAND_N);
  if (lh) write(lh, POSE_DIM, HAND_N);

  const rh = sized(rightHand, HAND_N);
  if (rh) write(rh, POSE_DIM + HAND_DIM, HAND_N);

  const faceLite = selectFace(face);
  if (faceLite) write(faceLite, POSE_DIM + 2 * HAND_DIM, FACE_N);

  return out;
}

/** Frame-to-frame deltas appended → 522 dims. Off by default (PRD §4.2). */
export function withVelocity(curr: Float64Array, prev: Float64Array | null): Float64Array {
  const out = new Float64Array(FEATURE_DIM * 2);
  out.set(curr, 0);
  if (prev) for (let i = 0; i < FEATURE_DIM; i++) out[FEATURE_DIM + i] = curr[i] - prev[i];
  return out;
}
