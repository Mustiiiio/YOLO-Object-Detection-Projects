from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2 as cv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "yolomW.pt"
VIDEO_PATH    = "testdad.mp4"
OUTPUT_PATH   = "output_sahi_nmm.mp4"

CONFIDENCE    = 0.15        # low threshold — mines are small/hard to detect
IOU_THRESHOLD = 0.3

# Tuned for 1024×576
SLICE_H       = 192         # 576  / 192 → ~4 rows
SLICE_W       = 256         # 1024 / 256 → ~5 cols  → ~20 slices/frame
OVERLAP_H     = 0.3
OVERLAP_W     = 0.3

CLASSES       = [0, 1, 2]
PROCESS_EVERY = 20
DEVICE        = "cuda:0"    # "cpu" if no GPU

# NMM config
NMM_MATCH_THRESHOLD = 0.3
NMM_CLASS_AGNOSTIC  = False

# Class labels — replace with your actual class names
CLASS_NAMES = {
    0: "mine",
    1: "mine_cluster",
    2: "suspicious",
}

CLASS_COLORS = {
    0: (0,   0,   255),     # red
    1: (0,   165, 255),     # orange
    2: (0,   255, 255),     # yellow
}
DEFAULT_COLOR = (255, 0, 0)
# ─────────────────────────────────────────────────────────────────────────────


# ══ Custom NMM ════════════════════════════════════════════════════════════════

def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def box_ios(a: np.ndarray, b: np.ndarray) -> float:
    """
    Intersection over Smaller area.
    Better than IoU when one box is fully inside another
    (common at slice boundaries for tiny mines).
    """
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])

    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)

    return inter / smaller if smaller > 0 else 0.0


def merge_group(group: list) -> dict:
    """
    Merge a group of overlapping predictions into one.
      - Bbox  : union of all boxes (preserves full spatial extent)
      - Score : max confidence in the group
      - Label : class of the highest-confidence prediction
    """
    x1 = min(p["x1"] for p in group)
    y1 = min(p["y1"] for p in group)
    x2 = max(p["x2"] for p in group)
    y2 = max(p["y2"] for p in group)
    best = max(group, key=lambda p: p["score"])

    return {
        "x1":    x1,
        "y1":    y1,
        "x2":    x2,
        "y2":    y2,
        "score": best["score"],
        "label": best["label"],
        "class": best["class"],
        "count": len(group),    # how many raw boxes were merged
    }


def non_maximum_merging(predictions: list,
                        iou_threshold: float  = NMM_MATCH_THRESHOLD,
                        class_agnostic: bool  = NMM_CLASS_AGNOSTIC,
                        use_ios: bool         = True) -> list:
    """
    Custom Non-Maximum Merging.

    Steps:
      1. Convert SAHI predictions → flat box dicts
      2. Sort by confidence (descending)
      3. Greedily group overlapping boxes (IoU or IoS ≥ threshold)
      4. Merge each group into a single union box

    Args:
        predictions   : raw SAHI ObjectPrediction list
        iou_threshold : overlap threshold to trigger merging
        class_agnostic: merge across classes if True
        use_ios       : use Intersection-over-Smaller instead of IoU
                        (recommended for tiny mines at slice edges)
    """
    if not predictions:
        return []

    boxes = []
    for pred in predictions:
        b = pred.bbox
        boxes.append({
            "x1":    b.minx,
            "y1":    b.miny,
            "x2":    b.maxx,
            "y2":    b.maxy,
            "score": pred.score.value,
            "label": pred.category.name,
            "class": pred.category.id,
        })

    # Sort by confidence — high-confidence boxes anchor the groups
    boxes.sort(key=lambda b: b["score"], reverse=True)

    metric_fn = box_ios if use_ios else box_iou
    used       = [False] * len(boxes)
    results    = []

    for i, anchor in enumerate(boxes):
        if used[i]:
            continue

        group   = [anchor]
        used[i] = True
        a_arr   = np.array([anchor["x1"], anchor["y1"],
                             anchor["x2"], anchor["y2"]])

        for j, cand in enumerate(boxes):
            if used[j]:
                continue
            if not class_agnostic and cand["class"] != anchor["class"]:
                continue

            c_arr = np.array([cand["x1"], cand["y1"],
                               cand["x2"], cand["y2"]])

            if metric_fn(a_arr, c_arr) >= iou_threshold:
                group.append(cand)
                used[j] = True

        results.append(merge_group(group))

    return results


# ══ Model ═════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> AutoDetectionModel:
    print(f"      Device : {DEVICE}")
    print(f"      Slices : {SLICE_W}×{SLICE_H} px  |  "
          f"overlap {int(OVERLAP_W*100)}%×{int(OVERLAP_H*100)}%")

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=CONFIDENCE,
        device=DEVICE,
        image_size=256,         # match slice width
    )


# ══ Inference ═════════════════════════════════════════════════════════════════

def run_sahi_on_frame(frame: np.ndarray,
                      model: AutoDetectionModel) -> list:
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = get_sliced_prediction(
        image=frame_rgb,
        detection_model=model,
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP_H,
        overlap_width_ratio=OVERLAP_W,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.9,    # permissive — our NMM does the real work
        postprocess_class_agnostic=NMM_CLASS_AGNOSTIC,
        perform_standard_pred=True,         # also run on full frame, then merge
        verbose=0,
    )

    raw_preds = [
        p for p in result.object_prediction_list
        if p.category.id in CLASSES
    ]

    return non_maximum_merging(raw_preds, use_ios=True)


# ══ Drawing ═══════════════════════════════════════════════════════════════════

def draw_predictions(frame: np.ndarray, predictions: list) -> np.ndarray:
    for pred in predictions:
        x1, y1 = int(pred["x1"]), int(pred["y1"])
        x2, y2 = int(pred["x2"]), int(pred["y2"])
        label   = CLASS_NAMES.get(pred["class"], pred["label"])
        score   = pred["score"]
        count   = pred.get("count", 1)
        color   = CLASS_COLORS.get(pred["class"], DEFAULT_COLOR)

        # Box
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner markers for small detections
        corner = 8
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),
                                (x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv.line(frame, (cx, cy), (cx + dx*corner, cy), color, 3)
            cv.line(frame, (cx, cy), (cx, cy + dy*corner), color, 3)

        # Label
        merge_tag = f" [{count}]" if count > 1 else ""
        text = f"{label}: {score:.2f}{merge_tag}"
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(frame,
                     (x1, y1 - th - 8),
                     (x1 + tw + 4, y1),
                     color, -1)
        cv.putText(frame, text, (x1 + 2, y1 - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_hud(frame: np.ndarray, frame_idx: int,
             fps: float, predictions: list) -> np.ndarray:
    h, w     = frame.shape[:2]
    count    = len(predictions)

    # Top bar
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv.addWeighted(overlay, 0.55, frame, 0.45, 0)

    hud = (f"Frame {frame_idx:05d}  |  "
           f"{fps:.1f} FPS  |  "
           f"Detections: {count}  |  "
           f"NMM IoS ≥ {NMM_MATCH_THRESHOLD}  |  "
           f"Conf ≥ {CONFIDENCE}")

    color = (0, 0, 255) if count > 0 else (0, 220, 0)
    cv.putText(frame, hud, (10, 27),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Per-class mini legend bottom-left
    for i, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
        col   = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        cls_count = sum(1 for p in predictions if p["class"] == cls_id)
        txt   = f"{cls_name}: {cls_count}"
        y_pos = h - 12 - i * 22
        cv.putText(frame, txt, (10, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    return frame


def draw_slice_grid(frame: np.ndarray) -> np.ndarray:
    """Visualise the SAHI slice grid on the frame (debug toggle)."""
    h, w   = frame.shape[:2]
    step_x = int(SLICE_W * (1 - OVERLAP_W))
    step_y = int(SLICE_H * (1 - OVERLAP_H))
    overlay = frame.copy()

    x = 0
    while x < w:
        cv.line(overlay, (x, 0), (x, h), (80, 80, 80), 1)
        x += step_x

    y = 0
    while y < h:
        cv.line(overlay, (0, y), (w, y), (80, 80, 80), 1)
        y += step_y

    return cv.addWeighted(overlay, 0.4, frame, 0.6, 0)


# ══ Main ══════════════════════════════════════════════════════════════════════

def main():
    SHOW_GRID   = False         # set True to visualise slice grid overlay
    SHOW_WINDOW = True          # set False for headless / server processing

    print("── SAHI Mine Detector ────────────────────────")
    print("[1/3] Loading model …")
    model = load_model(MODEL_PATH)

    print("[2/3] Opening video …")
    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv.CAP_PROP_FPS)
    total  = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"      {width}×{height} @ {fps:.1f} fps — {total} frames")

    # Sanity check slice vs frame size
    if SLICE_H >= height or SLICE_W >= width:
        print(f"\n⚠  WARNING: slice ({SLICE_W}×{SLICE_H}) ≥ "
              f"frame ({width}×{height}) — SAHI won't tile properly!\n")

    out = cv.VideoWriter(
        OUTPUT_PATH,
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("[3/3] Running inference …\n")

    frame_idx   = 0
    last_preds  = []
    total_count = defaultdict(int)
    t0          = time.time()

    with tqdm(total=total, unit="frame", dynamic_ncols=True) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % PROCESS_EVERY == 0:
                last_preds = run_sahi_on_frame(frame, model)
                for p in last_preds:
                    total_count[CLASS_NAMES.get(p["class"], p["label"])] += 1

            live_fps  = frame_idx / (time.time() - t0 + 1e-6)
            annotated = frame.copy()

            if SHOW_GRID:
                annotated = draw_slice_grid(annotated)

            annotated = draw_predictions(annotated, last_preds)
            annotated = draw_hud(annotated, frame_idx, live_fps, last_preds)

            out.write(annotated)

            if SHOW_WINDOW:
                cv.imshow("Mine Detection — SAHI + NMM", annotated)
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nStopped by user.")
                    break
                elif key == ord("g"):      # press G to toggle grid overlay
                    SHOW_GRID = not SHOW_GRID

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    cv.destroyAllWindows()

    elapsed = time.time() - t0
    print("\n── Summary ───────────────────────────────────")
    for cls, cnt in total_count.items():
        print(f"  {cls:20s}: {cnt}")
    print(f"  {'Total frames':20s}: {frame_idx}")
    print(f"  {'Elapsed':20s}: {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps avg)")
    print(f"  {'Output':20s}: {OUTPUT_PATH}")
    print("──────────────────────────────────────────────")


if __name__ == "__main__":
    main()