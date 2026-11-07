import cv2 as cv
import numpy as np
import ultralytics as ul
import os

# Define paths
video_path = "test.mp4"
output_video_path = "output_detected.mp4"

# Check if the video file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' not found.")

# Load the YOLO model
try:
    model = ul.YOLO("yolomW.pt")  # Load a custom model
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# Open the video file
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video file '{video_path}'.")

# Get video properties
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# Initialize video writer
out = cv.VideoWriter(
    output_video_path,
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

try:
    # Predict on the video
    results = model(video_path)

    # Process each frame
    for result in results:
        annotated_frame = result.plot()
        out.write(annotated_frame)

        # Display the frame (optional)
        cv.imshow("YOLO Detection", annotated_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Release resources
    cap.release()
    out.release()
    cv.destroyAllWindows()

print(f"Processed video saved to '{output_video_path}'.")
