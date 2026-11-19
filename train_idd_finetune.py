from pathlib import Path
from ultralytics import YOLO
import torch

DATASET = r"D:\RoadSentinel-Datasets\processed\IDD_YOLO\idd.yaml"
CHECKPOINT = r"D:\RoadSentinel-Datasets\training-runs\idd_yolov8n_full\weights\best.pt"
PROJECT = r"D:\RoadSentinel-Datasets\training-runs"
RUN_NAME = "idd_yolov8n_finetune_50ep_20261103"

if not Path(DATASET).is_file():
    raise FileNotFoundError(DATASET)

if not Path(CHECKPOINT).is_file():
    raise FileNotFoundError(CHECKPOINT)

if (Path(PROJECT) / RUN_NAME).exists():
    raise FileExistsError(f"Output folder already exists: {Path(PROJECT) / RUN_NAME}")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

model = YOLO(CHECKPOINT)

model.train(
    data=DATASET,
    epochs=50,
    imgsz=640,
    batch=2,
    device=0,
    workers=0,
    patience=15,
    lr0=0.001,
    cos_lr=True,
    warmup_epochs=3,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=False,
    save=True,
    plots=True,
    val=True,
    amp=True,
)
