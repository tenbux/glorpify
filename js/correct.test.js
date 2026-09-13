import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyBrushStroke } from './correct.js';

test('add mode paints a filled circle at a single point and returns a new array', () => {
  const width = 20, height = 20;
  const mask = new Uint8Array(width * height); // all zero

  const out = applyBrushStroke(mask, width, height, [[10, 10]], 3, 'add');

  assert.notEqual(out, mask);
  assert.equal(out[10 * width + 10], 255); // circle center
  assert.equal(out[10 * width + 12], 255); // within radius 3
  assert.equal(out[10 * width + 16], 0);   // well outside radius
  assert.equal(mask[10 * width + 10], 0, 'original mask must be untouched');
});

test('erase mode clears a filled circle from an all-255 mask', () => {
  const width = 20, height = 20;
  const mask = new Uint8Array(width * height).fill(255);

  const out = applyBrushStroke(mask, width, height, [[10, 10]], 3, 'erase');

  assert.equal(out[10 * width + 10], 0);
  assert.equal(out[10 * width + 16], 255);
});

test('connects consecutive points with a thick stroke so fast drags do not leave gaps', () => {
  const width = 40, height = 20;
  const mask = new Uint8Array(width * height);

  // Two points 20px apart; a point on the segment between them, farther from
  // both circle centers than the radius, must still be painted by the
  // connecting stroke.
  const out = applyBrushStroke(mask, width, height, [[5, 10], [25, 10]], 3, 'add');

  assert.equal(out[10 * width + 15], 255); // midpoint of the segment
});
