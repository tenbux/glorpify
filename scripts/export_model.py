"""
Export yolov8s-seg.pt to a static-shape fp32 ONNX model for the browser app.

Output: model/yolov8s-seg.onnx (gitignored; regenerate locally, do not commit).

Run: python scripts/export_model.py
"""

import os
import shutil

from ultralytics import YOLO


def main():
    model = YOLO("yolov8s-seg.pt")
    exported_path = model.export(format="onnx", opset=12, simplify=True)

    os.makedirs("model", exist_ok=True)
    dest = os.path.join("model", "yolov8s-seg.onnx")
    shutil.move(exported_path, dest)
    print(f"Exported to {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
