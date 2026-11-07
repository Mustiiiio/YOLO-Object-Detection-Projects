# Inventory and provenance

| Repository component | Origin | Verification |
|---|---|---|
| `01_mine_mockup_detection/weights/best263D.pt` | Uploaded `best263D.pt` | File copied, SHA-256 verified; no model inference test |
| `02_shape_detection/weights/best_shape.pt` | Uploaded `best (1).pt` | File copied, SHA-256 verified; no model inference test |
| Three files under `01_mine_mockup_detection/src/original/` | Three original user-supplied Python files | Copied unchanged and Python syntax-parsed |
| Five files under `01_mine_mockup_detection/results/` | Uploaded mine evaluation PNGs | Copied unchanged and ZIP CRC checked |
| `02_shape_detection/training/args_unverified.yaml` | Uploaded `args.yaml` | Copied unchanged; model association unknown |
| `tools/run_inference.py`, documentation, config | New repository packaging | Newly written; runner syntax-checked only |

See `FILE_MANIFEST_SHA256.txt` for file-level hashes. Training datasets were **not** uploaded and are not represented as if present.
