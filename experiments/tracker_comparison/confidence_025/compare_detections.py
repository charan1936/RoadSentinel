
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

BASELINE = ROOT / "experiments" / "tracker_comparison" / "botsort" / "botsort_detections.csv"

LOW_CONF = ROOT / "experiments" / "tracker_comparison" / "confidence_025" / "botsort_conf025" / "botsort_conf025_detections.csv"

OUTPUT = ROOT / "experiments" / "tracker_comparison" / "confidence_025"

print("Loading CSV files...")

baseline = pd.read_csv(BASELINE)
low_conf = pd.read_csv(LOW_CONF)

print("Baseline detections:", len(baseline))
print("Confidence 0.25 detections:", len(low_conf))

# Count detections per frame.
base_frames = baseline.groupby("frame").size()
low_frames = low_conf.groupby("frame").size()

frame_comparison = pd.DataFrame({
    "baseline_035": base_frames,
    "confidence_025": low_frames,
}).fillna(0)

frame_comparison["extra_detections"] = (
    frame_comparison["confidence_025"]
    - frame_comparison["baseline_035"]
)

print("\nFrame-level comparison:")
print(frame_comparison.describe().round(2))

print("\nFrames with the largest detection increase:")
print(
    frame_comparison.sort_values(
        "extra_detections", ascending=False
    ).head(15).to_string()
)

# Confidence distribution.
print("\nConfidence distribution:")
print(
    low_conf["confidence"].describe().round(3)
)

# Extra low-confidence detections.
print("\nConfidence 0.25 detections below 0.35:")
print(
    (low_conf["confidence"] < 0.35).sum()
)

# Save frame comparison.
output_path = OUTPUT / "frame_detection_comparison.csv"
frame_comparison.to_csv(output_path)

print("\nSaved:", output_path)
