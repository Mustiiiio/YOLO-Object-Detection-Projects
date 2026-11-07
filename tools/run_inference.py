#!/usr/bin/env python3
"""Generic YOLO inference convenience CLI for either *supplied* checkpoint.

Unlike the preserved historical scripts, the model path is resolved from this repository.
Inference was not validated during packaging. Run only on trusted model files.
"""

from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--project', required=True, choices=['mine', 'shape'])
    ap.add_argument('--source', required=True, help='Image/video file, directory, or webcam index e.g. 0')
    ap.add_argument('--conf', type=float, default=0.25, help='Inference confidence threshold, not an optimized deployment threshold')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None, help='Example: cpu, 0')
    args = ap.parse_args()

    if not 0 <= args.conf <= 1:
        ap.error('--conf must be between 0 and 1')

    from ultralytics import YOLO

    root = Path(__file__).resolve().parent.parent
    models = {
        'mine': root / 'projects/01_mine_mockup_detection/weights/best263D.pt',
        'shape': root / 'projects/02_shape_detection/weights/best_shape.pt',
    }
    model_path = models[args.project]
    if not model_path.exists():
        raise FileNotFoundError(f'Model weights are missing: {model_path}')

    # Ultralytics may deserialize PyTorch objects; trust this checkpoint before proceeding.
    model = YOLO(str(model_path))
    print('Checkpoint class names:', model.names)
    source = int(args.source) if args.source.isdigit() else args.source
    kwargs = {
        'source': source,
        'conf': args.conf,
        'imgsz': args.imgsz,
        'save': True,
        'project': str(root / 'outputs'),
        'name': args.project,
        'exist_ok': False,
    }
    if args.device is not None:
        kwargs['device'] = args.device
    model.predict(**kwargs)
    print('Saved annotated results under:', root / 'outputs')


if __name__ == '__main__':
    main()
