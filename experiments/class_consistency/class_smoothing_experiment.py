import pandas as pd
from pathlib import Path

# Original detections: read-only
CSV_PATH = Path(
    r"outputs\video_jobs\e925455d0c95_detections.csv"
)

OUTPUT_DIR = Path(r"experiments\class_consistency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Preserve original labels
df["original_class"] = df["class_name"]

# Apply dominant-class smoothing independently per track
df["smoothed_class"] = (
    df.groupby("track_id")["class_name"]
    .transform(lambda x: x.mode().iloc[0])
)

# Count original class transitions
def count_switches(series):
    return series.ne(series.shift()).sum() - 1

original_switches = (
    df.groupby("track_id")["original_class"]
    .apply(count_switches)
    .clip(lower=0)
)

smoothed_switches = (
    df.groupby("track_id")["smoothed_class"]
    .apply(count_switches)
    .clip(lower=0)
)

print("\n========== CLASS SMOOTHING EXPERIMENT ==========")

print(f"Original class switches: {original_switches.sum()}")
print(f"Smoothed class switches: {smoothed_switches.sum()}")

print(
    f"Switch reduction: "
    f"{original_switches.sum() - smoothed_switches.sum()}"
)

# Save all detections with original and smoothed labels
output_path = OUTPUT_DIR / "smoothed_detections.csv"

df.to_csv(output_path, index=False)

print(f"\nSaved experiment results to: {output_path}")

# Analyze track 4353 specifically
track = df[df["track_id"] == 4353]

print("\n========== TRACK 4353 ==========")
print("Original class distribution:")
print(track["original_class"].value_counts().to_string())

print("\nSmoothed class:")
print(track["smoothed_class"].iloc[0])