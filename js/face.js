/**
 * Default marker positions for the Glorp editor. No auto-detection; returns
 * sensible defaults derived from the mask/bbox so markers always have
 * somewhere reasonable to start from.
 *
 * Computing the centroid directly in absolute image coordinates (rather
 * than cropping to the head region first, then adding the crop offset back)
 * gives the identical result: for a fixed total mass m00, shifting every
 * sample point by a constant offset shifts the centroid by that same
 * offset, so the two approaches are algebraically equivalent.
 */
// Keep default marker positions at least this far from every edge (as a
// fraction of the shorter image dimension), so their draggable hit area
// never starts clipped by the canvas edge, which is especially awkward to
// grab with a touch on mobile.
const EDGE_MARGIN_FRACTION = 0.08;

function clampToEdgeMargin([x, y], width, height) {
  const margin = Math.round(Math.min(width, height) * EDGE_MARGIN_FRACTION);
  return [
    Math.max(margin, Math.min(width - 1 - margin, x)),
    Math.max(margin, Math.min(height - 1 - margin, y)),
  ];
}

export function detectFacePoints(width, height, mask, bbox) {
  if (bbox == null) {
    const cx = Math.floor(width / 2);
    const cy = Math.floor(height / 3);
    const spread = Math.floor(width / 8);
    return {
      eyes: [
        clampToEdgeMargin([cx - spread, cy], width, height),
        clampToEdgeMargin([cx + spread, cy], width, height),
      ],
      headTop: clampToEdgeMargin([cx, cy - spread], width, height),
    };
  }

  const [x1, y1, x2, y2] = bbox;
  const catH = y2 - y1;
  const catW = x2 - x1;

  const headY1 = Math.max(0, y1 - Math.floor(catH * 0.05));
  const headY2 = Math.min(height, y1 + Math.floor(catH * 0.45));
  const headX1 = Math.max(0, x1 - Math.floor(catW * 0.05));
  const headX2 = Math.min(width, x2 + Math.floor(catW * 0.05));

  let cx, cy;
  if (mask != null) {
    let m00 = 0, m10 = 0, m01 = 0;
    for (let y = headY1; y < headY2; y++) {
      for (let x = headX1; x < headX2; x++) {
        const v = mask[y * width + x];
        if (v) {
          m00 += v;
          m10 += v * x;
          m01 += v * y;
        }
      }
    }
    if (m00 > 0) {
      cx = Math.floor(m10 / m00);
      cy = Math.floor(m01 / m00);
    } else {
      cx = Math.floor((x1 + x2) / 2);
      cy = Math.floor(y1 + catH * 0.25);
    }
  } else {
    cx = Math.floor((x1 + x2) / 2);
    cy = Math.floor(y1 + catH * 0.25);
  }

  const aspect = catW / Math.max(catH, 1);
  const spread = Math.floor(catW * (aspect > 0.6 ? 0.20 : 0.15));

  return {
    eyes: [
      clampToEdgeMargin([cx - spread, cy], width, height),
      clampToEdgeMargin([cx + spread, cy], width, height),
    ],
    headTop: clampToEdgeMargin([cx, cy - spread], width, height),
  };
}
