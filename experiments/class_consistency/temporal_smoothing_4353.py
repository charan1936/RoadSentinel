
import pandas as pd
from pathlib import Path

CSV_PATH = Path(
    r"outputs\video_jobs\e925455d0c95_detections.csv"
)

OUTPUT_DIR = Path(r"experiments\class_consistency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Analyze only track 4353
track = df[df["track_id"] == 4353].copy()
track = track.sort_values("frame").reset_index(drop=True)

track["original_class"] = track["class_name"]
track["smoothed_class"] = track["class_name"]

# Replace only isolated class changes surrounded by the same class.
for i in range(1, len(track) - 1):
    previous_class = track.loc[i - 1, "class_name"]
    current_class = track.loc[i, "class_name"]
    next_class = track.loc[i + 1, "class_name"]

    if previous_class == next_class and current_class != previous_class:
        track.loc[i, "smoothed_class"] = previous_class

print("\n========== TRACK 4353 ==========")

print("Original class distribution:")
print(track["original_class"].value_counts().to_string())

print("\nSmoothed class distribution:")
print(track["smoothed_class"].value_counts().to_string())

print("\nChanged detections:")
changed = track[
    track["original_class"] != track["smoothed_class"]
]

print(f"Number changed: {len(changed)}")

print("\nChanged rows:")
print(
    changed[
        ["frame", "original_class", "smoothed_class"]
    ].to_string(index=False)
)

output_path = OUTPUT_DIR / "track_4353_temporal_smoothing.csv"

track.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")
