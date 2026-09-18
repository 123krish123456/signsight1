/**
 * Parity harness (PRD §9). Reads the fixture frames, prints one JSON array of
 * 261-float vectors. Driven by tests/test_parity.py — not part of the app build.
 * Run: node tests/parity_runner.ts <fixture.json>   (node >= 22.18, type stripping)
 */
import { readFileSync } from "node:fs";
import { normaliseFrame, type Block } from "../app/src/normalise.ts";

/** Fixtures store landmarks as [x,y,z] triples; MediaPipe hands back {x,y,z}. */
type Triple = [number, number, number];
const points = (block: Triple[] | null): Block =>
  block?.map(([x, y, z]) => ({ x, y, z })) ?? null;

interface Frame {
  pose: Triple[] | null;
  left_hand: Triple[] | null;
  right_hand: Triple[] | null;
  face: Triple[] | null;
}

const frames: Frame[] = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = frames.map((f) =>
  Array.from(
    normaliseFrame(points(f.pose), points(f.left_hand), points(f.right_hand), points(f.face)),
  ),
);
process.stdout.write(JSON.stringify(out));
