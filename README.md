# Computer Vision Portfolio | YOLO Mine-Mockup & Shape Detection

Two **separate computer-vision experiments**, consolidated into one GitHub-ready repository:

| Project | Evidence available | Implementation status |
|---|---|---|
| [01 — Inert 3D-printed mine mockup detection](projects/01_mine_mockup_detection/README.md) | YOLO weight file, three original Python scripts, five training/evaluation figures | Original scripts preserved; convenience inference CLI included. Runtime and metrics not independently revalidated. |
| [02 — Geometric shape detection](projects/02_shape_detection/README.md) | YOLO weight file; uploaded training settings with *unconfirmed* model association | Generic image/video inference CLI included; original project-specific shape code, dataset and evaluation plots were not supplied. |

## Repository layout

```text
YOLO-Mine-and-Shape-Detection-FINAL/
├── README.md
├── START_HERE.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── results_interpretation.md
│   ├── reproducibility_and_limitations.md
│   └── inventory.md
├── tools/
│   └── run_inference.py
└── projects/
    ├── 01_mine_mockup_detection/
    │   ├── README.md
    │   ├── weights/best263D.pt
    │   ├── src/original/{pic.py, import cv2 as cv.py, video.py}
    │   ├── results/{precision_confidence.png, f1_confidence.png,
    │   │            recall_confidence.png, confusion_matrix.png, training_curves.png}
    │   └── data/       # data not supplied
    └── 02_shape_detection/
        ├── README.md
        ├── weights/best_shape.pt
        ├── training/args_unverified.yaml
        ├── results/    # evaluation plots not supplied
        └── data/       # data not supplied
```

## Quick start

Create and activate a Python virtual environment, install `requirements.txt`, then run a model on **your own input image or video**:

```bash
python -m pip install -r requirements.txt

# Mine mockups (input media not included)
python tools/run_inference.py --project mine --source path/to/image_or_video

# Shapes (model class labels read from the weight file when loadable)
python tools/run_inference.py --project shape --source path/to/image_or_video
```

Results are saved under `outputs/`. Weight files are included in the ZIP but are **not pre-tested** against a particular Python/Ultralytics installation. Loading `.pt` files deserializes code-bearing model objects: only load artifacts from sources you trust.

## Mine-mockup results at a glance

The provided validation figures name three classes: **M14**, **PFM1**, and **PMN2**. The combined curves label an **F1 score of 0.85 at confidence 0.605**, **precision of 1.00 at 0.922**, and **recall of 0.89 at 0.000**. These are *threshold-specific curve annotations*, not a general accuracy score. See [results interpretation](docs/results_interpretation.md).

## Scope and caveats

- The first project concerns **inert 3D-printed mockups** for computer-vision research. It is **not** an operational explosive-device detection system.
- The shape project is independent. Do not use the mine evaluation figures to claim shape-model performance.
- The uploaded `args.yaml` is *provisionally* stored with the shape project; **its association with the shape weights is unverified**. It specifies a YOLO26s configuration and 140 requested epochs, which does not establish the architecture or completed training epoch count of either submitted model.
- Original mine scripts contain hardcoded weight names and class names that do not consistently match the supplied artifacts. They are retained unchanged; use the generic runner as a convenience for initial experimentation.
- The dataset, original training notebook, and example input files were not provided. See [reproducibility and limitations](docs/reproducibility_and_limitations.md).

## Publication

Extract the ZIP and upload the **contents of the top-level directory** to your GitHub repository. The two `.pt` files are approximately 20 MB and 40 MB; consider Git LFS if repository cloning and long-term model versioning will matter. No license has been asserted; specify a license only when you have permission to do so.
