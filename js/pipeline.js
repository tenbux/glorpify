/**
 * Orchestration for the client-side Glorp pipeline: capped-size math and
 * the segment, recolor, default-marker composition. No DOM access here;
 * only main.js touches the DOM.
 */
import { segmentCat } from './segment.js';
import { glorpGreen } from './recolor.js';
import { detectFacePoints } from './face.js';

const MAX_SIDE = 1280;

export function computeCappedSize(width, height, maxSide = MAX_SIDE) {
  const longSide = Math.max(width, height);
  if (longSide <= maxSide) return { width, height };
  const scale = maxSide / longSide;
  // Truncate, not round, to match Python's int() behavior in the original resize.
  return { width: Math.floor(width * scale), height: Math.floor(height * scale) };
}

/**
 * Run the full segment, recolor, default-marker pipeline on an
 * already-capped RGBA buffer. Mirrors segmentCat's no-detection shape
 * ({mask: null, box: null}, no confidence key) with a single catFound flag
 * so callers never destructure a field that might not exist.
 */
export async function runPipeline({ session, ortModule, rgba, width, height }) {
  const { mask, box, confidence } = await segmentCat(session, rgba, width, height, ortModule);
  if (mask === null) return { catFound: false };

  const recoloredRgba = glorpGreen(rgba, width, height, mask);
  const { eyes, headTop } = detectFacePoints(width, height, mask, box);

  return { catFound: true, mask, recoloredRgba, eyes, headTop, confidence };
}
