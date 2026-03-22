# BioHackathon 2026 — Filament Segmentation Pipeline

A pipeline for detecting and analyzing **filaments** (brighter spots) in microscopy images. Includes segmentation methods (Ilastik, MorphologicalSegmenter), evaluation against Label Studio annotations, and measurement of filament length and persistence.

---

## Features

- **Segmentation**: Ilastik (probability maps), MorphologicalSegmenter (threshold + morphological ops), and classical methods (Otsu, Frangi, Watershed, etc.)
- **Evaluation**: IoU, Dice, and Boundary Dice vs ground truth from Label Studio polygon annotations
- **Measurement**: Filament length (ellipse major axis) and persistence time (longest consecutive detection run)
- **WebServer**: Streamlit app with Single Image, Batch Processing, and Stack Analysis modes

---

## Project Structure

```
BioHackathon2026/
├── workflow/                    # Python package
│   ├── segmentation.py          # Otsu, MorphologicalSegmenter, Frangi, Watershed, etc.
│   ├── data_loader.py           # RawImageLoader
│   ├── segment_tracker.py       # SegmentTracker
│   ├── evaluation.py            # SegmentationEvaluator
│   └── experiments.py           # SampleCollector, visualizations
├── WebServer/                   # Streamlit app + Docker
│   ├── app.py                   # Main app (Ilastik, evaluation, measurement)
│   ├── Dockerfile
│   └── filament_model_v1.ilp    # Ilastik project (required for Docker)
├── filament_segmentation_pipeline.ipynb
├── filament_evaluation.ipynb
├── TestSetRawTIF/               # Test images + Label Studio export JSON
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

### Option 1: Install workflow package (for notebooks)

```bash
cd BioHackathon2026
pip install -e .
```

### Option 2: Install from requirements.txt

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- numpy, Pillow, matplotlib, scipy, scikit-image, pandas
- imageio[ffmpeg], imagecodecs
- tifffile (for TIFF I/O)
- torch, sam2 (optional, for SAM2-based segmentation in notebooks)
- streamlit (for WebServer)

---

## Usage

### Jupyter notebooks

1. **`filament_segmentation_pipeline.ipynb`** — Load data, run segmentation methods, visualize
2. **`filament_evaluation.ipynb`** — Compare Ilastik, MorphologicalSegmenter, and others vs Label Studio ground truth

Ensure `workflow/data/` contains your multi-frame TIFFs (or adjust paths in the notebooks).

### WebServer (Streamlit)

**Locally (no Ilastik):**
```bash
cd WebServer
pip install streamlit pillow tifffile numpy scikit-image matplotlib pandas
streamlit run app.py
```

**With Docker (includes Ilastik):**
```bash
cd WebServer
docker build -t filament-detector-webserver .
docker run -p 8501:8501 filament-detector-webserver
```

Open http://localhost:8501

#### WebServer modes

| Mode | Description |
|------|-------------|
| **Single Image** | Preprocess TIFF → Ilastik → probability heatmap + overlay |
| **Batch Processing** | Process multiple TIFFs in parallel. Optional: upload Label Studio JSON for evaluation (IoU, Dice, Boundary Dice) of Ilastik and MorphologicalSegmenter |
| **Stack Analysis** | Upload multi-frame TIFF → filament length distribution + persistence (longest consecutive run of detections) |

---

## Evaluation

Ground truth is provided as a Label Studio export JSON with polygon annotations. Frame matching:

- JSON: `URA7_URA8_002-crop6_frame_N.png` → frame index `N`
- Uploaded files: `frame_042.tif` → frame index `42`

Place your Label Studio export JSON (e.g. `export_244107_project-244107-at-2026-03-21-11-44-ab41ab26.json`) in `TestSetRawTIF/` or upload it in the WebServer evaluation panel.

---

## License

[Specify your license here, e.g. MIT, Apache 2.0]

---

## Acknowledgments

Developed for BioHackathon 2026.
