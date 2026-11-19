import pandas as pd
from pathlib import Path

# Original detection file (read-only)
CSV_PATH = Path(
    r"outputs\video_jobs\e925455d0c95_detections.csv"
)

OUTPUT_DIR = Path(r"experiments\class_consistency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

required = {
    "frame",
    "track_id",
    "class_name",
    "confidence",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("\n========== DATASET SUMMARY ==========")
print(f"Total detections: {len(df)}")
print(f"Unique track IDs: {df['track_id'].nunique()}")
print(f"Unique classes: {df['class_name'].nunique()}")

# Analyze each track
track_stats = []

for track_id, group in df.groupby("track_id"):
    group = group.sort_values("frame")

    classes = group["class_name"].dropna()
    class_counts = classes.value_counts()

    class_switches = (
        group["class_name"]
        .ne(group["class_name"].shift())
        .sum()
        - 1
    )

    frames = group["frame"]

    track_stats.append({
        "track_id": track_id,
        "detections": len(group),
        "first_frame": frames.min(),
        "last_frame": frames.max(),
        "duration_frames": frames.max() - frames.min() + 1,
        "unique_classes": classes.nunique(),
        "class_switches": max(0, class_switches),
        "dominant_class": (
            class_counts.index[0]
            if len(class_counts)
            else "unknown"
        ),
        "dominant_class_ratio": (
            class_counts.iloc[0] / len(classes)
            if len(classes)
            else 0
        ),
    })

stats = pd.DataFrame(track_stats)

# Save results
output_path = OUTPUT_DIR / "baseline_track_analysis.csv"
stats.to_csv(output_path, index=False)

print("\n========== CLASS INSTABILITY ==========")

unstable = stats[
    (stats["unique_classes"] > 1)
    | (stats["class_switches"] > 0)
]

print(f"Tracks with multiple classes: {len(unstable)}")

print("\nMost class-inconsistent tracks:")
print(
    unstable.sort_values(
        "class_switches",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\n========== SHORT TRACKS ==========")

short_tracks = stats[stats["detections"] <= 3]

print(f"Tracks with 1-3 detections: {len(short_tracks)}")

print("\n========== RESULTS ==========")
print(f"Saved analysis to: {output_path}")