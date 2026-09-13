import { test } from 'node:test';
import assert from 'node:assert/strict';
import { glorpGreen } from './recolor.js';

test('background pixels are left untouched when mask is all zero', () => {
  const width = 2, height = 2;
  const rgba = new Uint8ClampedArray([
    10, 20, 30, 255,   40, 50, 60, 255,
    70, 80, 90, 255,   100, 110, 120, 255,
  ]);
  const mask = new Uint8Array(width * height); // all zero

  const out = glorpGreen(rgba, width, height, mask);

  assert.deepEqual(Array.from(out), Array.from(rgba));
  assert.notEqual(out, rgba, 'must return a new array, not the same reference');
});

test('a fully-white pixel fully inside the mask becomes the expected green', () => {
  // Uniform 3x3 image and mask so Gaussian feathering has no edge to soften
  // (a uniform field blurs back to itself), giving alpha = 1 everywhere.
  const width = 3, height = 3;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4] = 255; rgba[i * 4 + 1] = 255; rgba[i * 4 + 2] = 255; rgba[i * 4 + 3] = 255;
  }
  const mask = new Uint8Array(width * height).fill(255);

  const out = glorpGreen(rgba, width, height, mask);

  // White has S=0, so S boosts to the additive floor only (110); V=255 stays
  // outside the midtone lift range. hsvToRgb(120, 110, 255) = (145, 255, 145).
  const centerOffset = (1 * width + 1) * 4; // center pixel, away from any edge
  assert.equal(out[centerOffset], 145);
  assert.equal(out[centerOffset + 1], 255);
  assert.equal(out[centerOffset + 2], 145);
  assert.equal(out[centerOffset + 3], 255);
});

test('does not mutate its inputs', () => {
  const width = 2, height = 2;
  const rgba = new Uint8ClampedArray(16).fill(200);
  const mask = new Uint8Array(4).fill(255);
  const rgbaCopy = Array.from(rgba);
  const maskCopy = Array.from(mask);

  glorpGreen(rgba, width, height, mask);

  assert.deepEqual(Array.from(rgba), rgbaCopy);
  assert.deepEqual(Array.from(mask), maskCopy);
});
