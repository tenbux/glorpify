/**
 * Glorpify web app entry point: upload, capped-size decode, pipeline
 * (segment + recolor + default markers), editor (drag markers, brush
 * correction), render, download. Fully client-side, no network calls
 * except the one-time model download below.
 */
import { computeCappedSize, runPipeline } from './pipeline.js';
import { createSession, fetchModelWithProgress } from './model.js';
import { applyBrushStroke } from './correct.js';
import { glorpGreen } from './recolor.js';
import { drawGlorpFeatures } from './features.js';
import * as ort from './vendor/ort.all.bundle.min.mjs';

const MODEL_URL = 'https://glorpify-assets.tenbux.dev/yolov8s-seg.v1.onnx';

const $ = (id) => document.getElementById(id);

const instructionsBtn = $('instructions-btn');
const instructionsDialog = $('instructions-dialog');
const instructionsCloseBtn = $('instructions-close-btn');
const modelLoading = $('model-loading');
const modelProgressBar = $('model-progress-bar');
const modelProgressText = $('model-progress-text');
const modelLoadingLabel = $('model-loading-label');
const modelRetryBtn = $('model-retry-btn');
const fileInput = $('file-input');
const dropZone = $('drop-zone');
const spinner = $('spinner');
const errorBanner = $('error-banner');
const uploadSec = $('upload-section');
const editorSec = $('editor-section');
const resultSec = $('result-section');
const canvas = $('preview-canvas');
const hintText = $('hint-text');
const canvasWrap = $('canvas-wrapper');
const glorpBtn = $('glorp-btn');
const resetBtn = $('reset-btn');
const redoBtn = $('redo-btn');
const newBtn = $('new-btn');
const resultImg = $('result-img');
const downloadLink = $('download-link');
const eyeScaleInput = $('eye-scale');
const fixGreenBtn = $('fix-green-btn');
const brushPanel = $('brush-panel');
const brushRadiusInput = $('brush-radius');
const brushDoneBtn = $('brush-done-btn');

let session = null;
let modelReady = false;
let pendingFile = null;

let state = {
  rgba: null,       // capped-size original RGBA (Uint8ClampedArray), pre-recolor
  recoloredRgba: null,
  mask: null,       // Uint8Array, capped-size
  width: 0,
  height: 0,
  markers: { eyeL: null, eyeR: null, head: null },
  brushMode: false,
};

instructionsBtn.addEventListener('click', () => instructionsDialog.showModal());
instructionsCloseBtn.addEventListener('click', () => instructionsDialog.close());
instructionsDialog.addEventListener('click', (e) => {
  if (e.target === instructionsDialog) instructionsDialog.close(); // backdrop click
});

function updateModelProgress(loaded, total) {
  const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
  modelProgressBar.value = pct;
  modelProgressText.textContent = `${pct}%`;
}

async function startModelLoad() {
  modelLoading.hidden = false;
  modelRetryBtn.hidden = true;
  modelLoadingLabel.textContent = 'Loading Glorp brain...';
  modelProgressBar.hidden = false;
  setError(null);
  try {
    const buffer = await fetchModelWithProgress(MODEL_URL, updateModelProgress);
    modelProgressText.textContent = 'Starting up...';
    modelProgressBar.removeAttribute('value');
    session = await createSession(buffer, ort);
    modelReady = true;
    modelLoading.hidden = true;
    if (pendingFile) {
      const file = pendingFile;
      pendingFile = null;
      processFile(file);
    }
  } catch (err) {
    console.error('Model load failed:', err);
    modelLoadingLabel.textContent = '';
    modelProgressText.textContent = 'Failed to load, check your connection.';
    modelProgressBar.value = 0;
    modelRetryBtn.hidden = false;
  }
}

modelRetryBtn.addEventListener('click', startModelLoad);

// A browser new enough to run this module already passed the `nomodule`
// gate in index.html; WebAssembly is the one remaining requirement that
// gate can't check, and failing it is not a retryable network problem.
if (typeof WebAssembly === 'undefined') {
  modelLoading.hidden = false;
  modelProgressBar.hidden = true;
  modelLoadingLabel.textContent = '';
  modelProgressText.textContent = "This browser doesn't support WebAssembly. Please update your browser.";
} else {
  startModelLoad();
}

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  if (!file) return;
  setError(null);
  if (!modelReady) {
    pendingFile = file;
    modelLoading.hidden = false;
    return;
  }
  processFile(file);
}

async function processFile(file) {
  hideAll();
  showSpinner(true);

  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch (err) {
    showSpinner(false);
    setError('Cannot decode image - unsupported format?');
    showUpload();
    return;
  }

  const { width, height } = computeCappedSize(bitmap.width, bitmap.height);
  const scratch = document.createElement('canvas');
  scratch.width = width;
  scratch.height = height;
  const scratchCtx = scratch.getContext('2d');
  scratchCtx.drawImage(bitmap, 0, 0, width, height);
  const rgba = new Uint8ClampedArray(scratchCtx.getImageData(0, 0, width, height).data);

  let result;
  try {
    result = await runPipeline({ session, ortModule: ort, rgba, width, height });
  } catch (err) {
    console.error('Pipeline error:', err);
    showSpinner(false);
    setError('Something went wrong while processing this photo.');
    showUpload();
    return;
  }

  showSpinner(false);
  if (!result.catFound) {
    setError('No cat detected. Make sure the photo contains a cat!');
    showUpload();
    return;
  }

  state.rgba = rgba;
  state.recoloredRgba = result.recoloredRgba;
  state.mask = result.mask;
  state.width = width;
  state.height = height;
  state.markers.eyeL = result.eyes[0];
  state.markers.eyeR = result.eyes[1];
  state.markers.head = result.headTop;

  loadEditor();
}

function loadEditor() {
  canvas.width = state.width;
  canvas.height = state.height;
  drawRecoloredToCanvas();

  hintText.textContent =
    'Drag the markers onto the left eye, right eye, and top of head, then click Glorp it!';

  createMarkers();
  showEditor();
}

function drawRecoloredToCanvas() {
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(state.recoloredRgba, state.width, state.height);
  ctx.putImageData(imageData, 0, 0);
}

let markerAbortController = null;

function createMarkers() {
  if (markerAbortController) markerAbortController.abort();
  markerAbortController = new AbortController();
  canvasWrap.querySelectorAll('.marker').forEach((m) => m.remove());
  createMarker('eyeL', 'marker-eye-l', 'L eye');
  createMarker('eyeR', 'marker-eye-r', 'R eye');
  createMarker('head', 'marker-head', 'antenna base');
}

function createMarker(key, cssClass, label) {
  const el = document.createElement('div');
  el.className = `marker ${cssClass}`;
  el.innerHTML = `<span class="marker-dot"></span><span class="marker-label">${label}</span>`;
  canvasWrap.appendChild(el);
  positionMarker(el, state.markers[key]);
  makeDraggable(el, key, markerAbortController.signal);
}

function positionMarker(el, [nx, ny]) {
  const rect = canvas.getBoundingClientRect();
  const wrapRect = canvasWrap.getBoundingClientRect();
  const scaleX = rect.width / state.width;
  const scaleY = rect.height / state.height;
  const cssX = rect.left - wrapRect.left + nx * scaleX;
  const cssY = rect.top - wrapRect.top + ny * scaleY;
  el.style.left = cssX + 'px';
  el.style.top = cssY + 'px';
}

function repositionAllMarkers() {
  canvasWrap.querySelectorAll('.marker').forEach((el) => {
    const key = el.classList.contains('marker-eye-l') ? 'eyeL'
      : el.classList.contains('marker-eye-r') ? 'eyeR'
      : 'head';
    positionMarker(el, state.markers[key]);
  });
}

window.addEventListener('resize', () => { if (!editorSec.hidden) repositionAllMarkers(); });

function makeDraggable(el, key, signal) {
  let dragging = false;
  let startX, startY, startLeft, startTop;

  const onStart = (clientX, clientY) => {
    dragging = true;
    startX = clientX;
    startY = clientY;
    startLeft = parseFloat(el.style.left);
    startTop = parseFloat(el.style.top);
    el.style.cursor = 'grabbing';
  };

  const onMove = (clientX, clientY) => {
    if (!dragging) return;
    const dx = clientX - startX;
    const dy = clientY - startY;
    const newLeft = startLeft + dx;
    const newTop = startTop + dy;
    el.style.left = newLeft + 'px';
    el.style.top = newTop + 'px';

    const rect = canvas.getBoundingClientRect();
    const wrapRect = canvasWrap.getBoundingClientRect();
    const scaleX = state.width / rect.width;
    const scaleY = state.height / rect.height;
    const nx = Math.round((newLeft - (rect.left - wrapRect.left)) * scaleX);
    const ny = Math.round((newTop - (rect.top - wrapRect.top)) * scaleY);
    state.markers[key] = [
      Math.max(0, Math.min(nx, state.width - 1)),
      Math.max(0, Math.min(ny, state.height - 1)),
    ];
  };

  const onEnd = () => { dragging = false; el.style.cursor = 'grab'; };

  el.addEventListener('mousedown', (e) => { e.preventDefault(); onStart(e.clientX, e.clientY); }, { signal });
  window.addEventListener('mousemove', (e) => onMove(e.clientX, e.clientY), { signal });
  window.addEventListener('mouseup', onEnd, { signal });

  el.addEventListener('touchstart', (e) => { e.preventDefault(); onStart(e.touches[0].clientX, e.touches[0].clientY); }, { signal });
  window.addEventListener('touchmove', (e) => { if (dragging) { e.preventDefault(); onMove(e.touches[0].clientX, e.touches[0].clientY); } }, { passive: false, signal });
  window.addEventListener('touchend', onEnd, { signal });
}

let lastResultUrl = null;

glorpBtn.addEventListener('click', () => {
  setError(null);
  const eyeScale = parseFloat(eyeScaleInput.value);
  const finalRgba = drawGlorpFeatures(
    state.recoloredRgba,
    state.width,
    state.height,
    [state.markers.eyeL, state.markers.eyeR],
    state.markers.head,
    eyeScale,
  );

  const renderCanvas = document.createElement('canvas');
  renderCanvas.width = state.width;
  renderCanvas.height = state.height;
  renderCanvas.getContext('2d').putImageData(new ImageData(finalRgba, state.width, state.height), 0, 0);

  renderCanvas.toBlob((blob) => {
    if (!blob) {
      setError('Something went wrong while rendering this image.');
      return;
    }
    if (lastResultUrl) URL.revokeObjectURL(lastResultUrl);
    const url = URL.createObjectURL(blob);
    lastResultUrl = url;
    resultImg.src = url;
    downloadLink.href = url;
    hideAll();
    showResult();
  }, 'image/png');
});

resetBtn.addEventListener('click', reset);
newBtn.addEventListener('click', reset);
redoBtn.addEventListener('click', () => { hideAll(); showEditor(); });

fixGreenBtn.addEventListener('click', () => {
  state.brushMode = !state.brushMode;
  fixGreenBtn.classList.toggle('active', state.brushMode);
  brushPanel.hidden = !state.brushMode;
  canvasWrap.classList.toggle('brush-mode', state.brushMode);
});

brushDoneBtn.addEventListener('click', () => {
  state.brushMode = false;
  fixGreenBtn.classList.remove('active');
  brushPanel.hidden = true;
  canvasWrap.classList.remove('brush-mode');
});

function currentBrushSettings() {
  const mode = document.querySelector('input[name="brush-mode"]:checked').value;
  const radius = parseInt(brushRadiusInput.value, 10);
  return { mode, radius };
}

function clientToNatural(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = state.width / rect.width;
  const scaleY = state.height / rect.height;
  const nx = Math.round((clientX - rect.left) * scaleX);
  const ny = Math.round((clientY - rect.top) * scaleY);
  return [
    Math.max(0, Math.min(nx, state.width - 1)),
    Math.max(0, Math.min(ny, state.height - 1)),
  ];
}

let brushStroke = null;

/**
 * Cheap live feedback while dragging: a flat semi-transparent dab painted
 * directly on the canvas. glorpGreen's full-image feathering pass costs
 * ~85-95ms on a large photo, far too slow to call on every drag move, so
 * the authoritative recompute (applyBrushStroke + glorpGreen) only runs
 * once, on brush end.
 */
function paintPreviewDab([nx, ny]) {
  const { mode, radius } = currentBrushSettings();
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = mode === 'add' ? 'rgba(57,211,83,0.6)' : 'rgba(255,255,255,0.6)';
  ctx.beginPath();
  ctx.arc(nx, ny, radius, 0, Math.PI * 2);
  ctx.fill();
}

function onBrushStart(clientX, clientY) {
  if (!state.brushMode) return;
  const pt = clientToNatural(clientX, clientY);
  brushStroke = [pt];
  paintPreviewDab(pt);
}

function onBrushMove(clientX, clientY) {
  if (!state.brushMode || !brushStroke) return;
  const pt = clientToNatural(clientX, clientY);
  brushStroke.push(pt);
  paintPreviewDab(pt);
}

function onBrushEnd() {
  if (!state.brushMode || !brushStroke) return;
  const { mode, radius } = currentBrushSettings();
  const points = brushStroke;
  brushStroke = null;

  state.mask = applyBrushStroke(state.mask, state.width, state.height, points, radius, mode);
  state.recoloredRgba = glorpGreen(state.rgba, state.width, state.height, state.mask);
  drawRecoloredToCanvas();
}

canvas.addEventListener('mousedown', (e) => { if (!state.brushMode) return; e.preventDefault(); onBrushStart(e.clientX, e.clientY); });
window.addEventListener('mousemove', (e) => { if (state.brushMode) onBrushMove(e.clientX, e.clientY); });
window.addEventListener('mouseup', () => { if (state.brushMode) onBrushEnd(); });

canvas.addEventListener('touchstart', (e) => { if (!state.brushMode) return; e.preventDefault(); onBrushStart(e.touches[0].clientX, e.touches[0].clientY); });
window.addEventListener('touchmove', (e) => { if (state.brushMode && brushStroke) { e.preventDefault(); onBrushMove(e.touches[0].clientX, e.touches[0].clientY); } }, { passive: false });
window.addEventListener('touchend', () => { if (state.brushMode) onBrushEnd(); });

function reset() {
  if (lastResultUrl) { URL.revokeObjectURL(lastResultUrl); lastResultUrl = null; }
  fileInput.value = '';
  state = { rgba: null, recoloredRgba: null, mask: null, width: 0, height: 0, markers: { eyeL: null, eyeR: null, head: null }, brushMode: false };
  brushPanel.hidden = true;
  fixGreenBtn.classList.remove('active');
  canvasWrap.classList.remove('brush-mode');
  setError(null);
  hideAll();
  showUpload();
}

function hideAll() {
  uploadSec.hidden = true;
  editorSec.hidden = true;
  resultSec.hidden = true;
  spinner.hidden = true;
}
function showSpinner(v) { spinner.hidden = !v; }
function showUpload() { uploadSec.hidden = false; }
function showEditor() { editorSec.hidden = false; repositionAllMarkers(); }
function showResult() { resultSec.hidden = false; }

function setError(msg) {
  errorBanner.hidden = !msg;
  errorBanner.textContent = msg || '';
}
