# 02 — Geometric shape detection

## Objective

A separate computer-vision project for recognizing geometric targets in imagery; this is **not** the mine-mockup model.

## Supplied assets

- `weights/best_shape.pt`: copy of user-uploaded `best (1).pt` (filename normalized for clarity).
- `training/args_unverified.yaml`: uploaded training configuration preserved without modification. **The user has not confirmed that this file belongs to these weights.**

## Unverified training configuration

The YAML mentions model `yolo26s.pt`, image size `640`, a requested 140 epochs, batch size 16, and a Colab Google Drive output directory. These fields are **not proof** of which model checkpoint was trained, the model's class names or final evaluation scores.

## Inference

For initial image/video inference, from the repository root:

```bash
python tools/run_inference.py --project shape --source /path/to/your/image.jpg
```

This generic script uses classes stored in the checkpoint, if the file loads successfully. It does **not** implement target pose estimation, geometry-based shape validation or UAV control.

## Still required for a reproducible study

- Original shape training notebook/script and dataset YAML.
- Validation/test images and result figures (precision, recall, mAP, confusion matrix).
- Exact class names and test methodology.
- Any shape-specific inference script and pose-estimation procedure developed during the project.

**Do not attribute mine-mockup performance curves to this shape detector.**
