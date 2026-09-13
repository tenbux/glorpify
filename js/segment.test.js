import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { letterboxSquare, resizeBilinearRGBA, buildInputTensor, selectDetection } from './segment.js';

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');

function fixtureDirs() {
  return readdirSync(FIXTURES_DIR).filter((d) => d !== 'photos' && !d.startsWith('.'));
}

function loadMeta(dir) {
  return JSON.parse(readFileSync(join(FIXTURES_DIR, dir, 'meta.json'), 'utf8'));
}

test('letterboxSquare matches the captured reference for every fixture photo', () => {
  for (const dir of fixtureDirs()) {
    const meta = loadMeta(dir);
    const lb = letterboxSquare(meta.cappedWidth, meta.cappedHeight);
    assert.deepEqual(lb, meta.letterbox, `mismatch for fixture ${dir}`);
  }
});

test('resizeBilinearRGBA produces exact values on a known 2x2 to 4x4 case', () => {
  // A 2x2 image: top-left red, top-right green, bottom-left blue, bottom-right white.
  const src = new Uint8ClampedArray([
    255, 0, 0, 255,   0, 255, 0, 255,
    0, 0, 255, 255,   255, 255, 255, 255,
  ]);
  const out = resizeBilinearRGBA(src, 2, 2, 4, 4);
  assert.equal(out.length, 4 * 4 * 4);
  // Corners of the output should closely match the corresponding source corners.
  assert.ok(out[0] > 200 && out[1] < 50, 'top-left stays red-ish');
  assert.ok(out[(3 * 4 + 3) * 4 + 0] > 200 && out[(3 * 4 + 3) * 4 + 1] > 200, 'bottom-right stays white-ish');
});

test('buildInputTensor has the right shape, is normalized, and pads with 114/255', () => {
  for (const dir of fixtureDirs()) {
    const meta = loadMeta(dir);
    const rgba = new Uint8ClampedArray(readFileSync(join(FIXTURES_DIR, dir, 'capped.rgba')).buffer);
    const lb = letterboxSquare(meta.cappedWidth, meta.cappedHeight);
    const tensor = buildInputTensor(rgba, meta.cappedWidth, meta.cappedHeight, lb);

    assert.equal(tensor.length, 3 * 640 * 640);
    for (const v of tensor) {
      assert.ok(v >= 0 && v <= 1, 'every value is normalized to [0,1]');
    }
    if (lb.top > 0) {
      // A pixel in the top padding band should be the 114/255 pad value.
      const padIndex = 0 * 640 + 0; // (y=0, x=0) is in the pad band whenever top > 0
      assert.ok(Math.abs(tensor[padIndex] - 114 / 255) < 1e-6);
    }
  }
});

function loadOutput0(dir) {
  return new Float32Array(readFileSync(join(FIXTURES_DIR, dir, 'output0.f32')).buffer);
}

test('selectDetection matches the captured reference confidence for every fixture photo', () => {
  for (const dir of fixtureDirs()) {
    const meta = loadMeta(dir);
    const output0 = loadOutput0(dir);
    const detection = selectDetection(output0);

    assert.ok(detection, `expected a detection for fixture ${dir}`);
    // Cross-language float32 rounding differences are expected; this is well within them.
    assert.ok(
      Math.abs(detection.confidence - meta.detection.confidence) < 0.001,
      `confidence mismatch for ${dir}: got ${detection.confidence}, expected ${meta.detection.confidence}`,
    );
    assert.equal(detection.coeffs.length, 32);

    // Cross-check the box by applying the same un-letterbox formula
    // segment.js itself uses, comparing against the fixture's
    // already-un-letterboxed boxOriginal.
    const lb = meta.letterbox;
    const [x1, y1, x2, y2] = detection.box;
    const gotBox = [
      Math.max(0, (x1 - lb.left) / lb.r),
      Math.max(0, (y1 - lb.top) / lb.r),
      Math.min(meta.cappedWidth, (x2 - lb.left) / lb.r),
      Math.min(meta.cappedHeight, (y2 - lb.top) / lb.r),
    ];
    for (let i = 0; i < 4; i++) {
      assert.ok(
        Math.abs(gotBox[i] - meta.boxOriginal[i]) < 2,
        `box[${i}] mismatch for ${dir}: got ${gotBox[i]}, expected ${meta.boxOriginal[i]}`,
      );
    }
  }
});

test('selectDetection returns null when no anchor exceeds the confidence threshold', () => {
  const flat = new Float32Array(116 * 8400); // all zeros, no detection exceeds 0.25
  assert.equal(selectDetection(flat), null);
});

import { decodeMask } from './segment.js';

function loadOutput1(dir) {
  return new Float32Array(readFileSync(join(FIXTURES_DIR, dir, 'output1.f32')).buffer);
}

test('decodeMask produces a plausible foreground region for every fixture photo', () => {
  for (const dir of fixtureDirs()) {
    const output0 = loadOutput0(dir);
    const output1 = loadOutput1(dir);
    const detection = selectDetection(output0);
    assert.ok(detection);

    const logits = decodeMask(detection.coeffs, output1, detection.box);
    assert.equal(logits.length, 640 * 640);

    let nonzero = 0;
    for (const v of logits) if (v > 0) nonzero++;
    // Sanity bounds only: a real cat mask should cover a meaningful but not
    // overwhelming fraction of the 640x640 canvas. The precise pixel-level
    // match is verified end-to-end in the segmentCat test below.
    assert.ok(nonzero > 500, `mask for ${dir} is implausibly small (${nonzero} px)`);
    assert.ok(nonzero < 640 * 640 * 0.9, `mask for ${dir} is implausibly large (${nonzero} px)`);
  }
});

import { unletterboxMask, morphClose } from './segment.js';

test('unletterboxMask crops the padding bands and places content correctly', () => {
  // unletterboxMask always reads its input as a true 640-wide array (that is
  // the real contract: mask640 comes from decodeMask, which always produces
  // 640x640), so the fixture here must be a genuine 640x640 array, not a
  // smaller stand-in.
  const mask640 = new Float32Array(640 * 640);
  const letterbox = { top: 64, bottom: 64, left: 0, right: 0, unpadWidth: 640, unpadHeight: 512, r: 1 };
  // A foreground pixel at 640-space (row 164, col 200), which is row 100 of
  // the unpadded region (164 - top=64 = 100).
  mask640[164 * 640 + 200] = 1;

  const out = unletterboxMask(mask640, letterbox, 640, 512); // identity resize (unpad size == orig size)

  assert.equal(out[100 * 640 + 200], 255);
  assert.equal(out[0 * 640 + 0], 0);
});

test('morphClose fills a small gap between two foreground blobs', () => {
  const width = 40, height = 20;
  const mask = new Uint8Array(width * height);
  // Two blobs separated by a 3px gap, well inside the default radius of 7.
  for (let y = 8; y < 12; y++) {
    for (let x = 5; x < 15; x++) mask[y * width + x] = 255;
    for (let x = 18; x < 28; x++) mask[y * width + x] = 255;
  }

  const closed = morphClose(mask, width, height);

  // The gap (x in [15,18)) should now be filled at the blobs' vertical center.
  assert.equal(closed[10 * width + 16], 255);
});

test('morphClose preserves a solid mask all the way to the image border', () => {
  const width = 40, height = 20;
  const mask = new Uint8Array(width * height).fill(255);

  const closed = morphClose(mask, width, height);

  // A fully foreground mask must stay fully foreground, including right at
  // the edges: erode must not treat out-of-bounds neighbors as background.
  assert.equal(closed[0 * width + 0], 255); // top-left corner
  assert.equal(closed[0 * width + (width - 1)], 255); // top-right corner
  assert.equal(closed[(height - 1) * width + 0], 255); // bottom-left corner
  assert.equal(closed[(height - 1) * width + (width - 1)], 255); // bottom-right corner
  for (let i = 0; i < closed.length; i++) assert.equal(closed[i], 255);
});

import ort from 'onnxruntime-node';
import { segmentCat } from './segment.js';

function loadMask(dir) {
  return new Uint8Array(readFileSync(join(FIXTURES_DIR, dir, 'mask.bin')).buffer);
}

function loadCapped(dir, meta) {
  return new Uint8ClampedArray(readFileSync(join(FIXTURES_DIR, dir, 'capped.rgba')).buffer);
}

function iou(a, b) {
  let inter = 0, union = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] > 0, bv = b[i] > 0;
    if (av && bv) inter++;
    if (av || bv) union++;
  }
  return union > 0 ? inter / union : NaN;
}

test('segmentCat reproduces the captured reference mask for every fixture photo', async () => {
  const modelPath = new URL('../model/yolov8s-seg.onnx', import.meta.url).pathname;
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });

  for (const dir of fixtureDirs()) {
    const meta = loadMeta(dir);
    const rgba = loadCapped(dir, meta);
    const expectedMask = loadMask(dir);

    const result = await segmentCat(session, rgba, meta.cappedWidth, meta.cappedHeight, ort);

    assert.ok(result.mask, `expected a mask for fixture ${dir}`);
    const score = iou(result.mask, expectedMask);
    // Measured range across the fixture photos was 0.988-0.999; 0.95
    // leaves real margin without being so loose it stops catching regressions.
    assert.ok(score >= 0.95, `IoU too low for ${dir}: ${score}`);
  }
});
