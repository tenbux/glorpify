/**
 * Manual mask correction: brush add/erase strokes. Operates directly on
 * the mask array (pure pixel math), not a canvas API.
 */

function stampCircle(mask, width, height, cx, cy, radius, value) {
  const x0 = Math.max(0, Math.floor(cx - radius));
  const x1 = Math.min(width - 1, Math.ceil(cx + radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const y1 = Math.min(height - 1, Math.ceil(cy + radius));
  const r2 = radius * radius;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy <= r2) mask[y * width + x] = value;
    }
  }
}

/**
 * Thick line as a capsule (rounded-cap) shape, rather than cv2.line's
 * flat-rectangle cap: the explicit stampCircle at every point (below)
 * already rounds the joints, so the cap-shape difference falls within the
 * parity tolerance from the client-side-rewrite spec's Testing section.
 */
function stampThickLine(mask, width, height, x1, y1, x2, y2, halfWidth, value) {
  const x0 = Math.max(0, Math.floor(Math.min(x1, x2) - halfWidth));
  const xEnd = Math.min(width - 1, Math.ceil(Math.max(x1, x2) + halfWidth));
  const y0 = Math.max(0, Math.floor(Math.min(y1, y2) - halfWidth));
  const yEnd = Math.min(height - 1, Math.ceil(Math.max(y1, y2) + halfWidth));
  const dx = x2 - x1, dy = y2 - y1;
  const lenSq = dx * dx + dy * dy || 1;
  const r2 = halfWidth * halfWidth;
  for (let y = y0; y <= yEnd; y++) {
    for (let x = x0; x <= xEnd; x++) {
      let t = ((x - x1) * dx + (y - y1) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));
      const px = x1 + t * dx, py = y1 + t * dy;
      const ddx = x - px, ddy = y - py;
      if (ddx * ddx + ddy * ddy <= r2) mask[y * width + x] = value;
    }
  }
}

/**
 * Return a NEW mask with a brush stroke applied along points. mode is "add"
 * (255) or "erase" (0). points is [[x,y],...] forming a polyline; a circle
 * is stamped at each point and a thick stroke connects consecutive points.
 */
export function applyBrushStroke(mask, width, height, points, radius, mode) {
  const value = mode === 'add' ? 255 : 0;
  const out = new Uint8Array(mask);
  for (const [x, y] of points) {
    stampCircle(out, width, height, x, y, radius, value);
  }
  for (let i = 0; i < points.length - 1; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];
    stampThickLine(out, width, height, x1, y1, x2, y2, radius, value);
  }
  return out;
}
