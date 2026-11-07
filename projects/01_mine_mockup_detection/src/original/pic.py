import cv2 as cv
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# Load the image
img = cv.imread("test.jpg")

# Resize the image to 640x640
img_resized = cv.resize(img, (640, 640))

# Load the YOLO model
model = YOLO("best263D2.pt")

# Run inference on the resized image
results = model(img_resized)

# Process and draw results
for result in results:
    boxes = result.boxes  # Get bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
        conf = box.conf[0]  # Confidence score
        cls = box.cls[0]  # Class ID

        # Draw the bounding box and label on the resized image
        cv.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Replace class ID with mine name
        mine_name = "1 PMN2"
        label = f"Mine: {mine_name}, Conf: {conf:.2f}"
        cv.putText(img_resized, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the resized image with detections
output_path = "result.jpg"
cv.imwrite(output_path, img_resized)
print(f"Result saved to {output_path}")

# Display the resized image with detections
cv.imshow("Detections", img_resized)
cv.waitKey(0)
cv.destroyAllWindows()