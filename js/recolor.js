/**
 * Neon-green recolor for the cat mask region. Operates on flat RGBA
 * buffers (browser ImageData shape). The algorithm fully overwrites hue
 * and only reads S/V from the source pixel, so RGB vs BGR channel order
 * never matters here.
 */

export const GREEN_HUE_DEG = 120; // OpenCV hue 60 (0-179 scale) equals 120 degrees (0-360 scale)
export const SAT_SCALE = 2.5;
export const SAT_ADD = 110;
export const V_LIFT_LOW = 60;
export const V_LIFT_HIGH = 200;
export const V_LIFT_AMOUNT = 20;
const FEATHER_RADIUS = 5;

export function clamp255(x) {
  return x < 0 ? 0 : x > 255 ? 255 : x;
}

/** Saturation and Value from an RGB triple, matching cv2.cvtColor(BGR2HSV)'s S/V formulas. */
export function rgbToSV(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const v = max;
  const s = max === 0 ? 0 : ((max - min) / max) * 255;
  return [s, v];
}

/** HSV (h in degrees 0-360, s/v in 0-255) to RGB (0-255 each). */
export function hsvToRgb(h, s, v) {
  const s1 = s / 255;
  const v1 = v / 255;
  const c = v1 * s1;
  const hh = h / 60;
  const x = c * (1 - Math.abs((hh % 2) - 1));
  let r1, g1, b1;
  if (hh < 1) [r1, g1, b1] = [c, x, 0];
  else if (hh < 2) [r1, g1, b1] = [x, c, 0];
  else if (hh < 3) [r1, g1, b1] = [0, c, x];
  else if (hh < 4) [r1, g1, b1] = [0, x, c];
  else if (hh < 5) [r1, g1, b1] = [x, 0, c];
  else [r1, g1, b1] = [c, 0, x];
  const m = v1 - c;
  return [(r1 + m) * 255, (g1 + m) * 255, (b1 + m) * 255];
}

/** Separable Gaussian blur of a single-channel 0-255 buffer, matching cv2.GaussianBlur's auto sigma for a given odd kernel size, with edge-clamped sampling. */
function gaussianBlurChannel(src, width, height, radius) {
  const ksize = radius * 2 + 1;
  const sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
  const kernel = new Float32Array(ksize);
  let sum = 0;
  for (let i = 0; i < ksize; i++) {
    const x = i - radius;
    const wgt = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel[i] = wgt;
    sum += wgt;
  }
  for (let i = 0; i < ksize; i++) kernel[i] /= sum;

  const tmp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const sx = Math.min(width - 1, Math.max(0, x + k));
        acc += src[y * width + sx] * kernel[k + radius];
      }
      tmp[y * width + x] = acc;
    }
  }
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const sy = Math.min(height - 1, Math.max(0, y + k));
        acc += tmp[sy * width + x] * kernel[k + radius];
      }
      out[y * width + x] = acc;
    }
  }
  return out;
}

/**
 * Recolor cat pixels to neon green, preserving fur texture and shading.
 * Returns a NEW Uint8ClampedArray; does not mutate rgba or mask.
 */
export function glorpGreen(rgba, width, height, mask) {
  const maskF = new Float32Array(width * height);
  for (let i = 0; i < maskF.length; i++) maskF[i] = mask[i];
  const feathered = gaussianBlurChannel(maskF, width, height, FEATHER_RADIUS);

  const out = new Uint8ClampedArray(rgba.length);
  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    const r = rgba[o], g = rgba[o + 1], b = rgba[o + 2];
    const alpha = feathered[i] / 255;

    const [s, v] = rgbToSV(r, g, b);
    const sBoosted = clamp255(s * SAT_SCALE + SAT_ADD);
    const vLifted = v >= V_LIFT_LOW && v <= V_LIFT_HIGH ? clamp255(v + V_LIFT_AMOUNT) : v;
    const [gr, gg, gb] = hsvToRgb(GREEN_HUE_DEG, sBoosted, vLifted);

    out[o] = Math.round(gr * alpha + r * (1 - alpha));
    out[o + 1] = Math.round(gg * alpha + g * (1 - alpha));
    out[o + 2] = Math.round(gb * alpha + b * (1 - alpha));
    out[o + 3] = rgba[o + 3];
  }
  return out;
}
