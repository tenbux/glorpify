import { test } from 'node:test';
import assert from 'node:assert/strict';
import { detectFacePoints } from './face.js';

test('falls back to image-center defaults when there is no bbox', () => {
  const result = detectFacePoints(800, 600, null, null);

  const cx = 400, cy = 200, spread = 100; // width/2, height/3, width/8
  assert.deepEqual(result, {
    eyes: [[cx - spread, cy], [cx + spread, cy]],
    headTop: [cx, cy - spread],
  });
});

test('uses bbox-center defaults when a bbox is given but no mask', () => {
  const bbox = [100, 100, 300, 400]; // catW=200, catH=300
  const result = detectFacePoints(800, 600, null, bbox);

  const cx = 200; // (100+300)/2
  const cy = 175; // 100 + 300*0.25
  const aspect = 200 / 300; // 0.667 > 0.6
  const spread = Math.floor(200 * 0.20); // 40

  assert.deepEqual(result, {
    eyes: [[cx - spread, cy], [cx + spread, cy]],
    headTop: [cx, cy - spread],
  });
});

test('centers on the mask centroid within the head region when a mask is given', () => {
  const width = 800, height = 600;
  const bbox = [100, 100, 300, 400]; // catH=300, head region y in [95, 235], x in [90, 310]
  const mask = new Uint8Array(width * height);
  // A small solid block whose center is (150, 150), well inside the head region.
  for (let y = 140; y < 160; y++) {
    for (let x = 140; x < 160; x++) mask[y * width + x] = 255;
  }

  const result = detectFacePoints(width, height, mask, bbox);

  // The block spans pixels 140..159 (20 columns), so the centroid is
  // floor(149.5) = 149 in both x and y, not the naive midpoint 150.
  assert.equal(result.headTop[0], 149);
  const expectedCy = 149;
  assert.equal(result.eyes[0][1], expectedCy);
  assert.equal(result.eyes[1][1], expectedCy);
});

test('falls back to bbox-center defaults when the head region of the mask is empty', () => {
  const width = 800, height = 600;
  const bbox = [100, 100, 300, 400];
  const mask = new Uint8Array(width * height); // all zero, m00 stays 0

  const result = detectFacePoints(width, height, mask, bbox);
  const withoutMask = detectFacePoints(width, height, null, bbox);

  assert.deepEqual(result, withoutMask);
});

test('clamps default points away from the image edge for a cat near a corner', () => {
  const width = 800, height = 600; // margin = round(min(800,600) * 0.08) = 48
  const bbox = [0, 0, 100, 100]; // small cat right at the top-left corner
  const result = detectFacePoints(width, height, null, bbox);

  // Raw (unclamped) math would give eyes=[[30,25],[70,25]], headTop=[50,5],
  // all within 48px of the top edge; every y (and eyes[0]'s x) must be
  // pulled in to the 48px margin instead of sitting flush against the edge.
  assert.deepEqual(result, {
    eyes: [[48, 48], [70, 48]],
    headTop: [50, 48],
  });
});
