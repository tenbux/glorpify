/**
 * Parametric Glorp feature drawing. No art assets; everything is
 * rasterized directly onto a flat RGBA buffer, scaled to the inter-eye
 * distance so it looks right at any resolution.
 */

export function computeAntennaOutwardVector(leftEye, rightEye, headTop) {
  const [lx, ly] = leftEye, [rx, ry] = rightEye, [cx, cy] = headTop;
  const ex = rx - lx, ey = ry - ly;
  const eyeLen = Math.hypot(ex, ey) || 1;
  const px = ex / eyeLen, py = ey / eyeLen;
  const perp1 = [-py, px];
  const perp2 = [py, -px];
  const midX = (lx + rx) / 2, midY = (ly + ry) / 2;
  const dot1 = (cx - midX) * perp1[0] + (cy - midY) * perp1[1];
  const [ux, uy] = dot1 >= 0 ? perp1 : perp2;
  return { ux, uy, px, py };
}

export function computeAntennaGeometry(eyeDist) {
  return {
    stalkH: Math.floor(eyeDist * 1.1),
    baseW: Math.max(Math.floor(eyeDist * 0.10), 6),
    spread: Math.floor(eyeDist * 0.22),
    bulbR: Math.max(Math.floor(eyeDist * 0.10), 5),
  };
}

export function computeEyeAxes(eyeDist, eyeScale) {
  return {
    axisX: Math.max(Math.floor(eyeDist * 0.38 * eyeScale), 12),
    axisY: Math.max(Math.floor(eyeDist * 0.26 * eyeScale), 8),
  };
}

export function computeEyeHighlightOffset(axisX, axisY, faceAngleDeg, left) {
  const sign = left ? 1 : -1;
  const angleR = (faceAngleDeg * Math.PI) / 180;
  const hx = Math.floor(sign * axisX * 0.20 * Math.cos(angleR));
  const hy = Math.floor(sign * axisX * 0.20 * Math.sin(angleR)) - Math.floor(axisY * 0.30);
  const hr = Math.max(Math.floor(axisX * 0.15), 2);
  return { hx, hy, hr };
}

function setPixel(rgba, width, height, x, y, color) {
  x = Math.round(x); y = Math.round(y);
  if (x < 0 || x >= width || y < 0 || y >= height) return;
  const o = (y * width + x) * 4;
  rgba[o] = color[0]; rgba[o + 1] = color[1]; rgba[o + 2] = color[2]; rgba[o + 3] = 255;
}

// fillCircle, fillPolygon, and fillEllipse mutate `rgba` in place (unlike the
// rest of this module's exports, which return a new array) and force alpha
// to 255 on every pixel they touch.
export function fillCircle(rgba, width, height, cx, cy, radius, color) {
  const x0 = Math.max(0, Math.floor(cx - radius));
  const x1 = Math.min(width - 1, Math.ceil(cx + radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const y1 = Math.min(height - 1, Math.ceil(cy + radius));
  const r2 = radius * radius;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy <= r2) setPixel(rgba, width, height, x, y, color);
    }
  }
}

/** Filled simple polygon via scanline fill (even-odd rule). points: [[x,y],...]. Mutates `rgba` in place; forces alpha to 255 on touched pixels. */
export function fillPolygon(rgba, width, height, points, color) {
  const ys = points.map((p) => p[1]);
  const yMin = Math.max(0, Math.floor(Math.min(...ys)));
  const yMax = Math.min(height - 1, Math.ceil(Math.max(...ys)));
  for (let y = yMin; y <= yMax; y++) {
    const xs = [];
    for (let i = 0; i < points.length; i++) {
      const [x1, y1] = points[i];
      const [x2, y2] = points[(i + 1) % points.length];
      if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
        const t = (y - y1) / (y2 - y1);
        xs.push(x1 + t * (x2 - x1));
      }
    }
    xs.sort((a, b) => a - b);
    for (let i = 0; i < xs.length - 1; i += 2) {
      const xStart = Math.max(0, Math.round(xs[i]));
      const xEnd = Math.min(width - 1, Math.round(xs[i + 1]));
      for (let x = xStart; x <= xEnd; x++) setPixel(rgba, width, height, x, y, color);
    }
  }
}

/** Filled ellipse, rotated by angleDeg around its own center. Mutates `rgba` in place; forces alpha to 255 on touched pixels. */
export function fillEllipse(rgba, width, height, cx, cy, axisX, axisY, angleDeg, color) {
  const angleR = (-angleDeg * Math.PI) / 180; // inverse-rotate sample points into the ellipse's own frame
  const cos = Math.cos(angleR), sin = Math.sin(angleR);
  const maxAxis = Math.max(axisX, axisY);
  const x0 = Math.max(0, Math.floor(cx - maxAxis));
  const x1 = Math.min(width - 1, Math.ceil(cx + maxAxis));
  const y0 = Math.max(0, Math.floor(cy - maxAxis));
  const y1 = Math.min(height - 1, Math.ceil(cy + maxAxis));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      const rx = dx * cos - dy * sin;
      const ry = dx * sin + dy * cos;
      if ((rx * rx) / (axisX * axisX) + (ry * ry) / (axisY * axisY) <= 1) {
        setPixel(rgba, width, height, x, y, color);
      }
    }
  }
}

const GREEN_DARK = [0, 120, 0];
const GREEN_MID = [20, 180, 20];
const EYE_DARK = [15, 25, 15];
const EYE_MID = [47, 90, 44];
const EYE_RIM = [5, 13, 5];
const RIM_LIGHT = [160, 235, 150];
const EDGE_DARK = [5, 15, 5];
const GLOSS_WHITE = [235, 255, 230];
function drawTaperedStalk(rgba, width, height, base, tip, baseHalfW, tipHalfW) {
  const [bx, by] = base, [tx, ty] = tip;
  const segDx = tx - bx, segDy = ty - by;
  const segLen = Math.hypot(segDx, segDy) || 1;
  const perpX = -segDy, perpY = segDx; // true perpendicular to the stalk direction

  const offset = (px, py, halfW) => [
    [px + (halfW * perpX) / segLen, py + (halfW * perpY) / segLen],
    [px - (halfW * perpX) / segLen, py - (halfW * perpY) / segLen],
  ];

  const [[bx1, by1], [bx2, by2]] = offset(bx, by, baseHalfW);
  const [[tx1, ty1], [tx2, ty2]] = offset(tx, ty, tipHalfW);
  fillPolygon(rgba, width, height, [[bx1, by1], [bx2, by2], [tx2, ty2], [tx1, ty1]], GREEN_DARK);

  // Highlight: a thinner capsule along the same base->tip line. cv2.line's
  // last argument is thickness (full width), so the half-width here is
  // half of the value used for the body polygon's half-width.
  const lineHalfW = Math.max(1, Math.floor(baseHalfW / 2)) / 2;
  const [[lbx1, lby1], [lbx2, lby2]] = offset(bx, by, lineHalfW);
  const [[ltx1, lty1], [ltx2, lty2]] = offset(tx, ty, lineHalfW);
  fillPolygon(rgba, width, height, [[lbx1, lby1], [lbx2, lby2], [ltx2, lty2], [ltx1, lty1]], GREEN_MID);
}

function drawAntennae(rgba, width, height, leftEye, rightEye, headTop, eyeDist) {
  const { ux, uy, px, py } = computeAntennaOutwardVector(leftEye, rightEye, headTop);
  const { stalkH, baseW, spread, bulbR } = computeAntennaGeometry(eyeDist);
  const [cx, cy] = headTop;

  for (const sign of [-1, 1]) {
    const bx = cx + sign * spread * px, by = cy + sign * spread * py;
    const tx = bx + sign * spread * px + stalkH * ux, ty = by + sign * spread * py + stalkH * uy;
    drawTaperedStalk(rgba, width, height, [bx, by], [tx, ty], baseW, 2);
    fillCircle(rgba, width, height, tx, ty, bulbR, GREEN_DARK);
    fillCircle(rgba, width, height, tx, ty, Math.max(bulbR - 2, 2), GREEN_MID);
  }
}

function clamp01(t) {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

function lerpColor(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

// Per-pixel equivalent of a `screen` blend against a light color: adds
// light proportional to `weight` without ever overshooting the light's own
// value, so it brightens the surface instead of pasting a flat color over
// it. `weight` is expected in [0, 1].
function addLight(base, weight, light) {
  return [
    base[0] + weight * (light[0] - base[0]),
    base[1] + weight * (light[1] - base[1]),
    base[2] + weight * (light[2] - base[2]),
  ];
}

// Weight (0..1, via Gaussian falloff) of an elongated, rotated highlight at
// local point (rx, ry) -- the per-pixel stand-in for a blurred, rotated
// radial-gradient ellipse.
function highlightWeight(rx, ry, cx, cy, radX, radY, rotDeg) {
  const rotR = (rotDeg * Math.PI) / 180;
  const cos = Math.cos(rotR), sin = Math.sin(rotR);
  const dx = rx - cx, dy = ry - cy;
  const lx = dx * cos + dy * sin;
  const ly = -dx * sin + dy * cos;
  const nx = lx / radX, ny = ly / radY;
  return Math.exp(-3 * (nx * nx + ny * ny));
}

/**
 * Glossy eye shading at local point (rx, ry): the pixel offset from the eye
 * center, already inverse-rotated into the ellipse's own frame (see
 * fillEllipseShaded). A sphere-shaded body, a primary catch-light (soft
 * halo + crisp core), a secondary ambient bounce anchored to the primary,
 * a wet-look rim light along the lower edge, and a darkened boundary band
 * so the gloss doesn't wash out the eye's silhouette.
 */
function alienEyeColor(rx, ry, axisX, axisY, left) {
  const sign = left ? 1 : -1;

  const gx = sign * axisX * 0.3, gy = -axisY * 0.5;
  const d = Math.hypot(rx - gx, ry - gy) / (axisX * 1.15);
  let color = d <= 0.55
    ? lerpColor(EYE_MID, EYE_DARK, d / 0.55)
    : lerpColor(EYE_DARK, EYE_RIM, clamp01((d - 0.55) / 0.45));

  const hx = sign * axisX * 0.20, hy = -axisY * 0.30;
  const hr = Math.max(axisX * 0.15, 2);
  const rot = -sign * 25;
  color = addLight(color, highlightWeight(rx, ry, hx, hy, hr * 1.2, hr * 0.75, rot) * 0.55, GLOSS_WHITE);
  color = addLight(color, highlightWeight(rx, ry, hx, hy, hr * 0.7, hr * 0.55, rot) * 0.5, GLOSS_WHITE);

  // Secondary highlight is anchored to the primary's position rather than
  // placed independently, so the two always read as one light source.
  const sx = hx - sign * axisX * 0.5, sy = hy + axisY * 0.6;
  color = addLight(color, highlightWeight(rx, ry, sx, sy, axisX * 0.24, axisY * 0.18, -sign * 20) * 0.25, GLOSS_WHITE);

  const rimT = clamp01((ry / axisY - 0.55) / 0.45);
  color = addLight(color, rimT * 0.35, RIM_LIGHT);

  const radiusN = Math.hypot(rx / axisX, ry / axisY);
  const edgeT = clamp01((radiusN - 0.92) / 0.08) * 0.85;
  color = lerpColor(color, EDGE_DARK, edgeT);

  return color;
}

/**
 * Filled ellipse with a per-pixel color callback instead of a flat color,
 * for shaded fills. colorFn receives (rx, ry): the pixel offset from
 * (cx, cy) after inverse-rotating into the ellipse's own frame, same
 * convention fillEllipse uses internally. Mutates `rgba` in place.
 */
function fillEllipseShaded(rgba, width, height, cx, cy, axisX, axisY, angleDeg, colorFn) {
  const angleR = (-angleDeg * Math.PI) / 180;
  const cos = Math.cos(angleR), sin = Math.sin(angleR);
  const maxAxis = Math.max(axisX, axisY);
  const x0 = Math.max(0, Math.floor(cx - maxAxis));
  const x1 = Math.min(width - 1, Math.ceil(cx + maxAxis));
  const y0 = Math.max(0, Math.floor(cy - maxAxis));
  const y1 = Math.min(height - 1, Math.ceil(cy + maxAxis));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      const rx = dx * cos - dy * sin;
      const ry = dx * sin + dy * cos;
      if ((rx * rx) / (axisX * axisX) + (ry * ry) / (axisY * axisY) <= 1) {
        setPixel(rgba, width, height, x, y, colorFn(rx, ry));
      }
    }
  }
}

function drawAlienEye(rgba, width, height, center, eyeDist, left, faceAngleDeg, eyeScale) {
  const [cx, cy] = center;
  const { axisX, axisY } = computeEyeAxes(eyeDist, eyeScale);
  fillEllipseShaded(rgba, width, height, cx, cy, axisX, axisY, faceAngleDeg, (rx, ry) =>
    alienEyeColor(rx, ry, axisX, axisY, left)
  );
}

/**
 * Draw Glorp antennae and alien eyes onto an RGBA buffer. Returns a NEW
 * Uint8ClampedArray; does not mutate rgba. eyes is [[x,y],[x,y]] left/right;
 * eyeScale uniformly scales both eyes (1.0 = default size).
 */
export function drawGlorpFeatures(rgba, width, height, eyes, headTop, eyeScale = 1.0) {
  const out = new Uint8ClampedArray(rgba);
  const [left, right] = eyes;
  const eyeDist = Math.max(Math.abs(right[0] - left[0]), 1);

  drawAntennae(out, width, height, left, right, headTop, eyeDist);

  const faceAngleDeg = (Math.atan2(right[1] - left[1], right[0] - left[0]) * 180) / Math.PI;
  drawAlienEye(out, width, height, left, eyeDist, true, faceAngleDeg, eyeScale);
  drawAlienEye(out, width, height, right, eyeDist, false, faceAngleDeg, eyeScale);

  return out;
}
