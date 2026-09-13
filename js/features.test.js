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
  // Eye centers should be the dark eye color (EYE_DARK = [15, 25, 15]).
  assert.deepEqual(pixelAt(out, width, 20, 20), [15, 25, 15, 255]);
  assert.deepEqual(pixelAt(out, width, 40, 20), [15, 25, 15, 255]);
  // A corner far from both eyes and the antennae must be untouched.
  assert.deepEqual(pixelAt(out, width, 1, 38), [255, 255, 255, 255]);
});
