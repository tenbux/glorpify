"""
Generate parity-test fixtures for js/segment.js's YOLOv8s-seg decode.

For each photo in js/fixtures/photos/*.jpg, dumps everything the JS test
suite needs to verify the decode without needing its own image codec or a
from-scratch re-derivation of the algorithm:

  js/fixtures/<name>/
    meta.json      - dimensions, letterbox params, detection, expected mask stats
    capped.rgba    - raw RGBA bytes of the _MAX_SIDE=1280-capped photo (segment.js's input)
    output0.f32    - raw float32 bytes of the model's output0 tensor (1,116,8400)
    output1.f32    - raw float32 bytes of the model's output1 tensor (1,32,160,160)
    mask.bin       - raw uint8 bytes (0 or 255) of the expected final mask, at capped size

output0/output1 are captured from a fixed 640x640 SQUARE letterbox, matching
what the browser's onnxruntime-web session is forced to use (the exported
ONNX graph has a static [1,3,640,640] input).

Run: python scripts/capture_segment_fixtures.py
Requires: model/yolov8s-seg.onnx (generate with scripts/export_model.py).
"""

import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

_MAX_SIDE = 1280
_CONF_THRES = 0.25
_CAT_CLASS = 15


def load_capped(path):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side > _MAX_SIDE:
        scale = _MAX_SIDE / long_side
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def letterbox_square(bgr, new_size=640):
    h, w = bgr.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (round(w * r), round(h * r))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized = cv2.resize(bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, top, left, bottom, right, new_unpad


def decode_square(sess, bgr):
    """Reference decode matching what segment.js must reproduce. Returns
    (binary_mask_at_capped_size, box_in_capped_coords, confidence, letterbox_info, out0, out1)."""
    padded, r, top, left, bottom, right, new_unpad = letterbox_square(bgr, 640)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]
    out0, out1 = sess.run(None, {"images": tensor})
    out0_b = out0[0]  # (116, 8400)
    out1_b = out1[0]  # (32, 160, 160)

    box = out0_b[0:4, :]
    cls_cat = out0_b[4 + _CAT_CLASS, :]
    coefs = out0_b[84:116, :]

    idx = int(np.argmax(cls_cat))
    conf = float(cls_cat[idx])
    letterbox_info = {
        "r": r, "top": top, "left": left, "bottom": bottom, "right": right,
        "unpadWidth": new_unpad[0], "unpadHeight": new_unpad[1],
    }
    if conf <= _CONF_THRES:
        return None, None, conf, letterbox_info, out0_b, out1_b

    cx, cy, bw, bh = box[:, idx]
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2

    coeffs_sel = coefs[:, idx]
    mask_logits = np.tensordot(coeffs_sel, out1_b, axes=([0], [0]))  # (160,160)

    px1, py1, px2, py2 = x1 * 0.25, y1 * 0.25, x2 * 0.25, y2 * 0.25
    xi1, yi1 = max(0, int(round(px1))), max(0, int(round(py1)))
    xi2, yi2 = min(160, int(round(px2))), min(160, int(round(py2)))
    cropped = np.zeros_like(mask_logits)
    cropped[yi1:yi2, xi1:xi2] = mask_logits[yi1:yi2, xi1:xi2]

    mask_640 = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_LINEAR)
    binary_640 = (mask_640 > 0).astype(np.uint8)

    unpad_w, unpad_h = new_unpad
    cropped_mask = binary_640[top : top + unpad_h, left : left + unpad_w]

    h, w = bgr.shape[:2]
    resized_mask = cv2.resize(cropped_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (resized_mask > 0.5).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    box_orig = (
        max(0, (x1 - left) / r),
        max(0, (y1 - top) / r),
        min(w, (x2 - left) / r),
        min(h, (y2 - top) / r),
    )
    return binary, box_orig, conf, letterbox_info, out0_b, out1_b


def main():
    sess = ort.InferenceSession("model/yolov8s-seg.onnx", providers=["CPUExecutionProvider"])
    photos = sorted(glob.glob("js/fixtures/photos/*.jpg"))
    print(f"Found {len(photos)} photos")

    for path in photos:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join("js/fixtures", stem)
        os.makedirs(out_dir, exist_ok=True)

        bgr = load_capped(path)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])

        binary, box_orig, conf, letterbox_info, out0, out1 = decode_square(sess, bgr)
        if binary is None:
            print(f"  {stem}: SKIPPED, no cat detected (conf {conf:.4f})")
            continue

        rgba.tofile(os.path.join(out_dir, "capped.rgba"))
        out0.astype(np.float32).tofile(os.path.join(out_dir, "output0.f32"))
        out1.astype(np.float32).tofile(os.path.join(out_dir, "output1.f32"))
        binary.astype(np.uint8).tofile(os.path.join(out_dir, "mask.bin"))

        meta = {
            "photo": os.path.basename(path),
            "cappedWidth": w,
            "cappedHeight": h,
            "letterbox": letterbox_info,
            "detection": {"confidence": conf, "classId": _CAT_CLASS},
            "boxOriginal": [float(v) for v in box_orig],
            "expectedMaskNonzeroCount": int((binary > 0).sum()),
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {stem}: conf={conf:.4f} nonzero={meta['expectedMaskNonzeroCount']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
