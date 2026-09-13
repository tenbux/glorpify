import { test } from 'node:test';
import assert from 'node:assert/strict';
import ort from 'onnxruntime-node';

const MODEL_PATH = new URL('../model/yolov8s-seg.onnx', import.meta.url).pathname;

test('the exported model loads via onnxruntime-node with the expected I/O shapes', async () => {
  const session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['cpu'] });

  assert.deepEqual(session.inputNames, ['images']);
  assert.deepEqual(session.outputNames, ['output0', 'output1']);

  const dummy = new Float32Array(1 * 3 * 640 * 640).fill(0.5);
  const inputTensor = new ort.Tensor('float32', dummy, [1, 3, 640, 640]);
  const results = await session.run({ images: inputTensor });

  assert.deepEqual(results.output0.dims, [1, 116, 8400]);
  assert.deepEqual(results.output1.dims, [1, 32, 160, 160]);
  assert.equal(results.output0.type, 'float32');
});

import { selectExecutionProviders } from './model.js';

import { fetchModelWithProgress } from './model.js';

test('selectExecutionProviders prefers webgpu, falling back to wasm', () => {
  assert.deepEqual(selectExecutionProviders(['webgpu', 'wasm']), ['webgpu', 'wasm']);
  assert.deepEqual(selectExecutionProviders(['wasm']), ['wasm']);
  assert.deepEqual(selectExecutionProviders([]), ['wasm']); // always fall back to wasm even if not "detected"
});

test('fetchModelWithProgress reports cumulative progress and reassembles the full buffer', async () => {
  const chunk1 = new Uint8Array([1, 2, 3]);
  const chunk2 = new Uint8Array([4, 5]);
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(chunk1);
      controller.enqueue(chunk2);
      controller.close();
    },
  });
  const fakeResponse = new Response(stream, { headers: { 'content-length': '5' } });
  const fakeFetch = async () => fakeResponse;

  const events = [];
  const buffer = await fetchModelWithProgress('fake://model', (loaded, total) => events.push([loaded, total]), fakeFetch);

  assert.deepEqual(events, [[3, 5], [5, 5]]);
  assert.deepEqual(Array.from(new Uint8Array(buffer)), [1, 2, 3, 4, 5]);
});

test('fetchModelWithProgress throws on a non-ok response', async () => {
  const fakeFetch = async () => new Response(null, { status: 404 });
  await assert.rejects(() => fetchModelWithProgress('fake://model', () => {}, fakeFetch));
});

test('fetchModelWithProgress reports total 0 when Content-Length is absent, without throwing', async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array([9]));
      controller.close();
    },
  });
  const fakeResponse = new Response(stream);
  const events = [];
  await fetchModelWithProgress('fake://model', (loaded, total) => events.push([loaded, total]), async () => fakeResponse);
  assert.deepEqual(events, [[1, 0]]);
});
