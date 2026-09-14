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
const zoomToggleBtn = $('zoom-toggle-btn');
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
  box: null,        // [x1,y1,x2,y2] cat bounding box from segmentCat, capped-size
  width: 0,
  height: 0,
  markers: { eyeL: null, eyeR: null, head: null },
  brushMode: false,
  // Markers are a fixed CSS size, so on a photo where the cat (and its
  // face) is a small fraction of the frame, they dominate and hide it.
  // Default to auto-zoomed on the cat's head region (see
  // headRegionBoundingBox); this flag is the escape hatch back to the full
  // photo.
  showFullPhoto: false,
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
  state.box = result.box;
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
  canvasWrap.style.setProperty('--canvas-aspect', `${state.width} / ${state.height}`);
  drawRecoloredToCanvas();

  hintText.textContent =
    'Drag the markers onto the left eye, right eye, and top of head, then click Glorp it!';

  state.showFullPhoto = false;
  updateZoomToggleLabel();
  // Section must be visible before rects are meaningful, and the zoom
  // transform must land before markers are positioned from it.
  showEditor();
  applyCanvasZoom();
  createMarkers();
}

function drawRecoloredToCanvas() {
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(state.recoloredRgba, state.width, state.height);
  ctx.putImageData(imageData, 0, 0);
}

// Same head-region heuristic detectFacePoints (js/face.js) uses to pick
// where to search the mask for a centroid: the top ~45% of the cat's own
// bounding box, not the default marker positions. Framing on the cat's
// actual detected location (mask/box, always correct) rather than the
// markers themselves means the zoom is still right even when the default
// markers land a bit off -- the user can then drag them into place within
// a crop that's reliably showing the real head, instead of the zoom having
// gambled on wherever the markers happened to default to.
const HEAD_REGION_TOP_MARGIN = 0.05;
const HEAD_REGION_HEIGHT_FRACTION = 0.45;
const HEAD_REGION_SIDE_MARGIN = 0.05;
const ZOOM_EXTRA_PADDING = 0.25; // extra slack around the head region, as a fraction of its own size
const MAX_ZOOM = 3;

function headRegionBoundingBox() {
  const [x1, y1, x2, y2] = state.box;
  const catW = x2 - x1;
  const catH = y2 - y1;

  const headY1 = y1 - catH * HEAD_REGION_TOP_MARGIN;
  const headY2 = y1 + catH * HEAD_REGION_HEIGHT_FRACTION;
  const headX1 = x1 - catW * HEAD_REGION_SIDE_MARGIN;
  const headX2 = x2 + catW * HEAD_REGION_SIDE_MARGIN;

  const padX = (headX2 - headX1) * ZOOM_EXTRA_PADDING;
  const padY = (headY2 - headY1) * ZOOM_EXTRA_PADDING;

  return {
    minX: headX1 - padX,
    maxX: headX2 + padX,
    minY: headY1 - padY,
    maxY: headY2 + padY,
  };
}

/**
 * Frame the canvas on the cat's head region via a CSS transform on the (now
 * absolutely-positioned) canvas element, so markers -- fixed CSS size --
 * take up proportionally less of a small/far-away cat, on both mouse and
 * touch. Falls back to the untransformed full photo when toggled off.
 *
 * All marker positioning and pointer-to-image-coordinate math elsewhere
 * (positionMarker, clientToNatural, the drag handlers) reads the canvas's
 * actual getBoundingClientRect(), which already reflects this transform,
 * so none of it needs to know zoom/pan happened at all.
 */
function applyCanvasZoom() {
  const wrapRect = canvasWrap.getBoundingClientRect();
  const box = state.showFullPhoto || !state.box ? null : headRegionBoundingBox();

  if (!box || wrapRect.width === 0) {
    canvas.style.transform = 'scale(1)';
    canvas.style.left = '0px';
    canvas.style.top = '0px';
    return;
  }

  // faceMarkerBoundingBox() already includes padding, so the box itself is
  // the padded frame -- just clamp it to the image bounds.
  const paddedW = Math.min(state.width, box.maxX - box.minX);
  const paddedH = Math.min(state.height, box.maxY - box.minY);
  const zoom = Math.max(1, Math.min(MAX_ZOOM, state.width / paddedW, state.height / paddedH));

  const cx = (box.minX + box.maxX) / 2;
  const cy = (box.minY + box.maxY) / 2;
  // The wrapper's aspect-ratio is locked to state.width/state.height, so
  // this ratio holds for both axes.
  const scaleCss = wrapRect.width / state.width;

  const scaledW = state.width * scaleCss * zoom;
  const scaledH = state.height * scaleCss * zoom;
  let left = wrapRect.width / 2 - cx * scaleCss * zoom;
  let top = wrapRect.height / 2 - cy * scaleCss * zoom;
  // Clamp so panning to frame the cat never reveals empty space past the
  // canvas's own edges (e.g. a cat near a corner of the photo).
  left = Math.min(0, Math.max(wrapRect.width - scaledW, left));
  top = Math.min(0, Math.max(wrapRect.height - scaledH, top));

  canvas.style.transform = `scale(${zoom})`;
  canvas.style.left = `${left}px`;
  canvas.style.top = `${top}px`;
}

function updateZoomToggleLabel() {
  zoomToggleBtn.textContent = state.showFullPhoto ? 'Zoom to face' : 'Show full photo';
}

zoomToggleBtn.addEventListener('click', () => {
  state.showFullPhoto = !state.showFullPhoto;
  updateZoomToggleLabel();
  applyCanvasZoom();
  repositionAllMarkers();
});

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

window.addEventListener('resize', () => {
  if (editorSec.hidden) return;
  applyCanvasZoom();
  repositionAllMarkers();
});

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

function renderFeatures(eyeScale) {
  return drawGlorpFeatures(
    state.recoloredRgba,
    state.width,
    state.height,
    [state.markers.eyeL, state.markers.eyeR],
    state.markers.head,
    eyeScale,
    state.mask,
  );
}

// Live preview while dragging the eye-size slider: paint the actual
// eyes+antennae at the in-progress size directly onto the editor canvas,
// using the current marker positions. Reverts to the plain recolored image
// on release, so a marker dragged afterward never leaves a stale preview
// showing eyes at their old spot. The markers themselves are hidden while
// previewing so they don't sit on top of (and hide) the size being shown.
eyeScaleInput.addEventListener('input', () => {
  if (!state.recoloredRgba) return;
  canvasWrap.classList.add('previewing-eye-size');
  const eyeScale = parseFloat(eyeScaleInput.value);
  const previewRgba = renderFeatures(eyeScale);
  canvas.getContext('2d').putImageData(new ImageData(previewRgba, state.width, state.height), 0, 0);
});
eyeScaleInput.addEventListener('change', () => {
  canvasWrap.classList.remove('previewing-eye-size');
  if (state.recoloredRgba) drawRecoloredToCanvas();
});

glorpBtn.addEventListener('click', () => {
  setError(null);
  const eyeScale = parseFloat(eyeScaleInput.value);
  const finalRgba = renderFeatures(eyeScale);

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

// Live preview while dragging the brush-size slider: same dab the brush
// itself paints while stroking, centered on the canvas so its size is
// visible without needing a stroke in progress. Reverts on release.
brushRadiusInput.addEventListener('input', () => {
  if (!state.recoloredRgba) return;
  drawRecoloredToCanvas();
  paintPreviewDab([Math.floor(state.width / 2), Math.floor(state.height / 2)]);
});
brushRadiusInput.addEventListener('change', () => {
  if (state.recoloredRgba) drawRecoloredToCanvas();
});

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
  state = { rgba: null, recoloredRgba: null, mask: null, box: null, width: 0, height: 0, markers: { eyeL: null, eyeR: null, head: null }, brushMode: false, showFullPhoto: false };
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
