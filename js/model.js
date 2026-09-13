/**
 * ONNX Runtime session management for the browser: WebGPU first, WASM
 * fallback. `createSession`'s default WebGPU detection (`navigator.gpu`)
 * is real, but only its "no `navigator`" and "no `navigator.gpu`" branches
 * are exercised by this file's automated tests, since there is no WebGPU
 * in Node; the "real browser with WebGPU" branch and the actual
 * `ort.InferenceSession.create` call are verified manually instead.
 */

// onnxruntime-web's WASM binaries (13-28MB each, one per browser-capability
// variant it auto-selects at runtime) are too large to vendor in-repo
// without bloating the repo and exceeding Cloudflare Pages' 25MiB per-file
// limit (the same reason the ONNX model itself is hosted on R2, not
// committed). Pointed at jsDelivr's mirror of the exact pinned npm version
// instead. No photo data is involved; this is pure runtime infrastructure.
const WASM_CDN_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';

/** Order of execution providers to try, given what onnxruntime-web reports as available. */
export function selectExecutionProviders(available) {
  const providers = [];
  if (available.includes('webgpu')) providers.push('webgpu');
  providers.push('wasm'); // always available as the universal fallback
  return providers;
}

function detectAvailableProviders() {
  return typeof navigator !== 'undefined' && navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
}

/** Create an inference session against modelPathOrBuffer, preferring WebGPU. */
export async function createSession(modelPathOrBuffer, ortModule, availableProviders = detectAvailableProviders()) {
  const executionProviders = selectExecutionProviders(availableProviders);
  if (ortModule.env?.wasm) {
    ortModule.env.wasm.wasmPaths = WASM_CDN_BASE;
  }
  return ortModule.InferenceSession.create(modelPathOrBuffer, { executionProviders });
}

/**
 * Fetch a URL with byte-level progress, for the model-download progress bar.
 * `fetchImpl` is injectable so this is testable in Node without a real
 * network call; it defaults to the global `fetch`, used in the browser.
 */
export async function fetchModelWithProgress(url, onProgress, fetchImpl = fetch) {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`Failed to fetch model: ${response.status}`);

  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (onProgress) onProgress(loaded, total);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}
