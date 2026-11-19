
from pathlib import Path
import torch
from ultralytics import YOLO

DATA_YAML = Path(r"D:\RoadSentinel-Datasets\processed\IDD_YOLO\idd.yaml")
RUNS_DIR = Path(r"D:\RoadSentinel-Datasets\training-runs")

if not DATA_YAML.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check the GPU setup.")

print("GPU:", torch.cuda.get_device_name(0))
print("Dataset:", DATA_YAML)
print("Training on the FULL training split.")

model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=640,
    batch=2,
    device=0,
    workers=0,
    cache=False,
    amp=True,
    val=True,
    patience=5,
    plots=True,
    project=str(RUNS_DIR),
    name="idd_yolov8n_full",
    exist_ok=True,
)

print("Training run finished.")
print("Check the saved best.pt model and validation metrics.")
