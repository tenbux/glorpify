<h1 align="center">Glorpify 👽🐱</h1>

<p align="center">
  Turn any cat photo into a <b>Glorp</b>, a neon-green alien-cat meme,<br>
  entirely in your browser. No server, no upload, no account.
</p>

<p align="center">
  <a href="#how-it-works">How it works</a> ·
  <a href="#run-it-yourself">Run it yourself</a> ·
  <a href="#private-by-default">Privacy</a> ·
  <a href="#license">License</a> ·
  <a href="mailto:glorpify@tenbux.dev">Contact</a> ·
  <a href="https://buymeacoffee.com/tenbux">Buy me a coffee</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License AGPL-3.0"></a>
  <img src="https://img.shields.io/badge/runs-entirely%20in%20your%20browser-4c8dff" alt="Runs entirely in your browser">
</p>

Upload a cat photo, and it gets segmented, recolored neon green, and topped with parametric antennae and alien eyes, all as pixels in a `<canvas>` on your own device. The photo is never sent anywhere.

## How it works

```
upload → YOLOv8-seg (cat mask, ONNX Runtime) → HSV green recolor
       → drag 3 markers onto the cat's face (L eye, R eye, antenna base)
       → "Glorp it" → antennae + alien eyes drawn → download PNG
```

- **Segmentation** runs YOLOv8s-seg via `onnxruntime-web`, picking the highest-confidence cat detection.
- **Recolor** is mask-based HSV: hue forced to green, saturation boosted (including a floor so near-white fur still goes green), brightness preserved so fur texture and shading survive.
- **Features** are drawn parametrically, scaled to the distance between the eye markers. No art assets.
- A **"Fix Green" brush** lets you paint over spots the auto-segmentation missed, or erase over-eager ones, before rendering.

## Run it yourself

The app is a static site (`index.html` + `style.css` + `js/`), no build step. It needs a `.onnx` model file at `model/yolov8s-seg.onnx`, which isn't committed (see below).

```bash
git clone https://github.com/tenbux/glorpify.git
cd glorpify

# Python side: export the model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/export_model.py        # writes model/yolov8s-seg.onnx (~45MB)

# JS side: install deps, generate test fixtures, run tests
npm install
python scripts/capture_segment_fixtures.py
npm test

# Serve the static site
python3 -m http.server 8000
# open http://localhost:8000/
```

The `yolov8s-seg.pt` weights auto-download on first export. `onnxruntime-node`'s postinstall script may need approval depending on npm's script-allowlist policy (`npm install-scripts approve onnxruntime-node`).

## Private by default

Nothing about your photo leaves your device. The only network requests the page makes are for the app's own static files, the ONNX runtime and its WASM binaries, and the segmentation model, none of which carry any of your data. There's no account, no analytics, and no image is ever uploaded to a server.

## Known limitations

- **Paw tips and tail edges** can get missed by segmentation on complex poses; use the "Fix Green" brush to touch those up.
- **Non-frontal or upside-down cats** still get a reasonable default marker position, but you'll likely want to drag them into place.
- **One cat per photo**: the single highest-confidence detection is used.
- **WebGPU is best-effort**: the app falls back to WASM automatically when WebGPU isn't available, which is slower but works everywhere.

## License

[AGPL-3.0](LICENSE). This project bundles [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (also AGPL-3.0) for cat segmentation; see their license and [enterprise licensing options](https://www.ultralytics.com/license) if you're building on this for something that needs different terms.

<p align="center">
  <sub>Made by <a href="https://github.com/tenbux">@tenbux</a> · questions to <a href="mailto:glorpify@tenbux.dev">glorpify@tenbux.dev</a></sub>
</p>
