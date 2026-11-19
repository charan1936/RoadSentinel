
import cv2
import numpy as np
from ultralytics import YOLO

print("Loading YOLO model...")

# Download/load a small pretrained model
model = YOLO("yolov8n.pt")

# Create a test image
image = np.zeros((640, 640, 3), dtype=np.uint8)
cv2.imwrite("test_frame.jpg", image)

# Run inference
results = model.predict(
    source="test_frame.jpg",
    save=True,
    verbose=False
)

print("YOLO inference completed successfully!")
print(f"Results generated: {len(results)}")
print("RoadSentinel detection environment is ready.")
