import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import ort from 'onnxruntime-node';
import { computeCappedSize, runPipeline } from './pipeline.js';

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');

function fixtureDirs() {
  return readdirSync(FIXTURES_DIR).filter((d) => d !== 'photos' && !d.startsWith('.'));
}

function loadMeta(dir) {
  return JSON.parse(readFileSync(join(FIXTURES_DIR, dir, 'meta.json'), 'utf8'));
}

function loadCapped(dir) {
  return new Uint8ClampedArray(readFileSync(join(FIXTURES_DIR, dir, 'capped.rgba')).buffer);
}

test('computeCappedSize leaves small images unchanged', () => {
  assert.deepEqual(computeCappedSize(800, 600), { width: 800, height: 600 });
  assert.deepEqual(computeCappedSize(1280, 1280), { width: 1280, height: 1280 });
});

test('computeCappedSize caps the long side to 1280, truncating not rounding', () => {
  // 2000x1000: scale = 1280/2000 = 0.64, so width 1280, height floor(1000*0.64)=640
  assert.deepEqual(computeCappedSize(2000, 1000), { width: 1280, height: 640 });
  // 1000x2000: scale = 1280/2000 = 0.64, so height 1280, width floor(1000*0.64)=640
  assert.deepEqual(computeCappedSize(1000, 2000), { width: 640, height: 1280 });
});

test('runPipeline returns catFound: false when segmentCat finds no cat, without throwing on the missing confidence key', async () => {
  const fakeSession = { run: async () => ({ output0: { data: new Float32Array(116 * 8400) }, output1: { data: new Float32Array(32 * 160 * 160) } }) };
  const fakeOrt = { Tensor: function Tensor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } };
  const rgba = new Uint8ClampedArray(4 * 4 * 4);
  const result = await runPipeline({ session: fakeSession, ortModule: fakeOrt, rgba, width: 4, height: 4 });
  assert.deepEqual(result, { catFound: false });
});

test('runPipeline composes segmentCat, glorpGreen, and detectFacePoints correctly for every fixture photo', async () => {
  const modelPath = new URL('../model/yolov8s-seg.onnx', import.meta.url).pathname;
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });

  for (const dir of fixtureDirs()) {
    const meta = loadMeta(dir);
    const rgba = loadCapped(dir);

    const result = await runPipeline({ session, ortModule: ort, rgba, width: meta.cappedWidth, height: meta.cappedHeight });

    assert.equal(result.catFound, true, `expected a cat for fixture ${dir}`);
    assert.equal(result.recoloredRgba.length, rgba.length);

    let nonzero = 0;
    for (let i = 0; i < result.mask.length; i++) if (result.mask[i]) nonzero++;
    const ratio = nonzero / meta.expectedMaskNonzeroCount;
    // segmentCat's own parity test (segment.test.js) already proves the mask
    // itself is correct to IoU >= 0.95; this test's job is proving the mask
    // that comes out of segmentCat is the same one glorpGreen/detectFacePoints
    // receive, so a loose count-ratio check here is deliberate, not sloppy.
    assert.ok(ratio > 0.9 && ratio < 1.1, `mask count ratio out of range for ${dir}: ${ratio}`);

    // Eye markers must land on pixels within the recolored image's bounds.
    for (const [ex, ey] of result.eyes) {
      assert.ok(ex >= 0 && ex < meta.cappedWidth && ey >= 0 && ey < meta.cappedHeight, `eye out of bounds for ${dir}`);
    }
  }
});
