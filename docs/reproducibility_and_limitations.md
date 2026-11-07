# Reproducibility, provenance and limitations

## Confirmed contents

- Both user-supplied model checkpoint byte streams.
- Three original mine-model inference scripts, copied byte-for-byte.
- Five original mine training/evaluation graphics, copied byte-for-byte.
- Uploaded training `args.yaml`, preserved without claiming definitive model association.

## Not included

- Raw annotated training datasets, source-image licenses, train/val/test split YAML.
- Original Colab training notebook and `results.csv`.
- Shape-specific source code, validation plots or example media.
- Independent inference or quantitative benchmarking output for either checkpoint.

## Important source details

- `pic.py` reads `test.jpg` and asks for `best263D2.pt`, which differs from the delivered `best263D.pt`. It manually assigns the text `1 PMN2` to **every** detection.
- `import cv2 as cv.py` and `video.py` expect `yolomW.pt`; that file was **not** supplied. They also expect videos not included in the archive.
- `video.py`'s class-name mapping is a **placeholder** not verified against the uploaded checkpoint. Detection reuse across 20 frames is not persistent multi-object tracking.
- The supplied shape configuration requests 140 epochs and identifies a model named `yolo26s.pt`, but its relationship to `best_shape.pt` has not been verified.
- Included CLI `tools/run_inference.py` is original **new packaging code**, not historical user code. It is syntax-tested but cannot be guaranteed compatible until a user-controlled inference test is run.

## Security and ethical scope

`.pt` checkpoints may invoke Python object deserialization when loaded. Load only trustworthy files in an isolated environment with versioned dependencies. Mine images represent **inert mockups**, and the repository is not suitable as a safety-critical detector or a real-world mine/IED identification system.

## GitHub hosting and licensing

No license file was automatically attached, since permission to distribute code, training datasets and model checkpoints must be determined by the repository owner. Each model is below GitHub's 100-MB per-file hard limit; Git LFS is an option, not a prerequisite for these weights. Check permissions for any datasets or third-party media before committing them.
