import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  computeAntennaOutwardVector,
  computeAntennaGeometry,
  computeEyeAxes,
  computeEyeHighlightOffset,
} from './features.js';

test('antenna outward vector points toward headTop, sideways vector runs along the eye line', () => {
  const result = computeAntennaOutwardVector([100, 100], [200, 100], [150, 50]);
  assert.deepEqual(result, { ux: 0, uy: -1, px: 1, py: 0 });
});

test('antenna geometry scales with eyeDist and enforces minimum sizes', () => {
  assert.deepEqual(computeAntennaGeometry(100), { stalkH: 110, baseW: 10, spread: 22, bulbR: 10 });
  // Small eyeDist should hit the baseW/bulbR floors (6 and 5).
  assert.deepEqual(computeAntennaGeometry(55), { stalkH: 60, baseW: 6, spread: 12, bulbR: 5 });
});

test('eye axes scale with eyeDist and eyeScale, and enforce minimum sizes', () => {
  assert.deepEqual(computeEyeAxes(100, 1.0), { axisX: 38, axisY: 26 });
  // Tiny eyeScale should hit the axisX/axisY floors (12 and 8).
  assert.deepEqual(computeEyeAxes(100, 0.1), { axisX: 12, axisY: 8 });
});

test('eye highlight offset mirrors sign between left and right eyes', () => {
  const left = computeEyeHighlightOffset(40, 30, 0, true);
  assert.deepEqual(left, { hx: 8, hy: -9, hr: 6 });

  const right = computeEyeHighlightOffset(40, 30, 0, false);
  assert.deepEqual(right, { hx: -8, hy: -9, hr: 6 });
});

import { fillCircle, fillPolygon, fillEllipse } from './features.js';

function blankBuffer(width, height) {
  return new Uint8ClampedArray(width * height * 4);
}

function pixelAt(rgba, width, x, y) {
  const o = (y * width + x) * 4;
  return [rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]];
}

test('fillCircle paints inside the radius and leaves the rest untouched', () => {
  const width = 5, height = 5;
  const rgba = blankBuffer(width, height);

  fillCircle(rgba, width, height, 2, 2, 1, [10, 20, 30]);

  assert.deepEqual(pixelAt(rgba, width, 2, 2), [10, 20, 30, 255]); // center
  assert.deepEqual(pixelAt(rgba, width, 0, 0), [0, 0, 0, 0]); // corner, untouched
});

test('fillPolygon fills a 2x2 square exactly', () => {
  const width = 6, height = 6;
  const rgba = blankBuffer(width, height);

  fillPolygon(rgba, width, height, [[1, 1], [3, 1], [3, 3], [1, 3]], [5, 6, 7]);

  assert.deepEqual(pixelAt(rgba, width, 2, 2), [5, 6, 7, 255]); // inside
  assert.deepEqual(pixelAt(rgba, width, 4, 4), [0, 0, 0, 0]); // outside
});

test('fillEllipse with equal axes matches a circle', () => {
  const width = 7, height = 7;
  const rgba = blankBuffer(width, height);

  fillEllipse(rgba, width, height, 3, 3, 2, 2, 0, [9, 9, 9]);

  assert.deepEqual(pixelAt(rgba, width, 3, 3), [9, 9, 9, 255]); // center
  assert.deepEqual(pixelAt(rgba, width, 0, 0), [0, 0, 0, 0]); // corner, outside radius
});

import { drawGlorpFeatures } from './features.js';

test('drawGlorpFeatures returns a new buffer with visible eyes and leaves distant pixels untouched', () => {
  const width = 60, height = 40;
  const rgba = blankBuffer(width, height);
  rgba.fill(255); // start fully white so "untouched" is easy to assert

  const eyes = [[20, 20], [40, 20]];
  const headTop = [30, 10];

  const out = drawGlorpFeatures(rgba, width, height, eyes, headTop, 1.0);

  assert.notEqual(out, rgba, 'must return a new array');
  // Eye centers sit on the glossy shading gradient (not flat EYE_DARK
  // anymore), but both eyes are mirror images so their centers match.
  assert.deepEqual(pixelAt(out, width, 20, 20), [24, 44, 24, 255]);
  assert.deepEqual(pixelAt(out, width, 40, 20), [24, 44, 24, 255]);
  // A corner far from both eyes and the antennae must be untouched.
  assert.deepEqual(pixelAt(out, width, 1, 38), [255, 255, 255, 255]);
});

test('antenna stalk is symmetric around its centerline for a tilted (non-level) head', () => {
  // Regression test: the previous version's perpendicular-offset math for
  // the stalk body was a reflection, not a 90-degree rotation, so the
  // "left" and "right" edges of the stalk polygon were offset by a
  // direction that wasn't perpendicular to the stalk at all. That's only
  // invisible when the stalk happens to be purely horizontal or vertical
  // (exactly what the level-eyes test above exercises, which is why it
  // never caught this), and produces a visibly skewed/pinched stalk at
  // any other angle. This checks the stalk is actually painted on BOTH
  // sides of its true (independently computed) centerline, not just one.
  const width = 160, height = 160;
  const rgba = blankBuffer(width, height);
  // A uniform, non-white "fur" color: the antenna now samples its dark
  // tone from the pixels under its base, so on a flat white background the
  // sampled tone (and the drawn body) would be white too, indistinguishable
  // from untouched background. A non-white fill keeps the two distinguishable.
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = 40; rgba[i + 1] = 150; rgba[i + 2] = 40; rgba[i + 3] = 255;
  }

  const eyes = [[60, 100], [100, 80]]; // tilted, not level
  const headTop = [80, 60];

  const out = drawGlorpFeatures(rgba, width, height, eyes, headTop, 1.0);

  function isBackground([r, g, b, a]) {
    return a !== 255 || (r === 40 && g === 150 && b === 40);
  }

  // Base/tip of the left-hand (sign=-1) antenna, computed independently of
  // the implementation from the same public helpers it's built from.
  const { ux, uy, px, py } = computeAntennaOutwardVector(eyes[0], eyes[1], headTop);
  const eyeDist = Math.hypot(eyes[1][0] - eyes[0][0], eyes[1][1] - eyes[0][1]);
  const { stalkH, baseW, spread } = computeAntennaGeometry(eyeDist);
  const bx = headTop[0] - spread * px, by = headTop[1] - spread * py;
  const tx = bx - spread * px + stalkH * ux, ty = by - spread * py + stalkH * uy;

  // True perpendicular to the base->tip line (independent re-derivation).
  const segDx = tx - bx, segDy = ty - by;
  const segLen = Math.hypot(segDx, segDy);
  const perpX = -segDy / segLen, perpY = segDx / segLen;

  // Sample at t=0.3 along the centerline, clear of the base (where both
  // antennae originate close together near headTop and can overlap).
  const t = 0.3;
  const cx0 = bx + t * (tx - bx), cy0 = by + t * (ty - by);
  const localHalfW = baseW + t * (2 - baseW); // linear taper: baseHalfW -> tipHalfW(=2)
  // The body's dark tone is now sampled from the (uniform, in this test)
  // background, so on a flat fill it exactly matches the background by
  // design -- only the brighter highlight capsule is guaranteed to differ.
  // It has a constant half-width along the whole stalk (see drawTaperedStalk).
  const lineHalfW = Math.max(1, Math.floor(baseW / 2)) / 2;

  // lineHalfW is sub-pixel-scale for this stalk size, so a single rounded
  // sample point is fragile to which way it happens to round; scan a small
  // neighborhood instead and require at least one non-background pixel in
  // it. A reflection bug (the regression this guards against) would put the
  // whole highlight on the wrong side, which this still catches.
  function anyNonBackground(cx, cy, r) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const x = Math.round(cx) + dx, y = Math.round(cy) + dy;
        if (x < 0 || x >= width || y < 0) continue;
        if (!isBackground(pixelAt(out, width, x, y))) return true;
      }
    }
    return false;
  }

  const side1x = cx0 + perpX * lineHalfW * 0.6, side1y = cy0 + perpY * lineHalfW * 0.6;
  const side2x = cx0 - perpX * lineHalfW * 0.6, side2y = cy0 - perpY * lineHalfW * 0.6;
  const beyond1 = pixelAt(out, width, Math.round(cx0 + perpX * localHalfW * 2.5), Math.round(cy0 + perpY * localHalfW * 2.5));
  const beyond2 = pixelAt(out, width, Math.round(cx0 - perpX * localHalfW * 2.5), Math.round(cy0 - perpY * localHalfW * 2.5));

  assert.ok(anyNonBackground(side1x, side1y, 1), `expected stalk highlight near one true edge (${side1x}, ${side1y})`);
  assert.ok(anyNonBackground(side2x, side2y, 1), `expected stalk highlight near the other true edge (${side2x}, ${side2y})`);
  assert.ok(isBackground(beyond1), `expected background well outside one true edge, got ${beyond1}`);
  assert.ok(isBackground(beyond2), `expected background well outside the other true edge, got ${beyond2}`);
});
