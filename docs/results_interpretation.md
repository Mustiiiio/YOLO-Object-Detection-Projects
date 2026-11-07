# Mine-mockup evaluation: interpreting supplied figures

The five figures in `projects/01_mine_mockup_detection/results/` were submitted by the project author. Numerical summaries below refer to the chart labels and legible confusion-matrix entries, not rerun model evaluation.

## Precision–confidence

The combined curve reports precision 1.00 at confidence 0.922. A precision of 1.00 at one high confidence threshold **does not** mean the model is perfectly accurate in general; raising confidence typically trades recall for precision.

## F1–confidence

The combined curve labels a maximum F1 of 0.85 at confidence 0.605. This describes the precision–recall balance at the threshold shown.

## Recall–confidence

The combined curve labels recall 0.89 at confidence 0.000. Such a low confidence threshold is not generally the best deployment threshold because it tends to increase false positives.

## Confusion matrix

The diagonal true-positive counts displayed are **M14=29**, **PFM1=28** and **PMN2=40**. The figure uses predicted class on the vertical axis, true class on the horizontal axis, and a background row/column for missed detections and spurious predictions. The chart indicates errors and background confusions, so this is not 100% detection.

## Training curves

The uploaded training figure shows generally decreasing train/validation losses and improving reported precision, recall and mAP through the plotted epochs. The chart is evidence of the *recorded run's trajectory*, not evidence of external robustness, an independent test set, or a deployable safety-critical detector. Exact per-epoch values should be retrieved from the original `results.csv` if available.

## Next evaluation steps

For a portfolio-quality benchmark, retain the data split file, rerun an evaluation with the original validation set, record software versions, and report classwise precision/recall, mAP@50, mAP@50:95, inference latency and operating threshold. Do not infer those values solely from visual inspection of the figures.
