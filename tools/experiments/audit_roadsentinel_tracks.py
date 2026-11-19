
from pathlib import Path
import pandas as pd

# ============================================================
# ROADSENTINEL TRACKING QUALITY AUDIT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "roadsentinel_detections.csv"

TRACK_SUMMARY_CSV = OUTPUT_DIR / "track_audit_summary.csv"
CLASS_SUMMARY_CSV = OUTPUT_DIR / "track_class_summary.csv"
FLAGGED_TRACKS_CSV = OUTPUT_DIR / "flagged_tracks.csv"
REPORT_TXT = OUTPUT_DIR / "track_audit_report.txt"

MIN_FRAMES_FOR_CLASS_AUDIT = 3
MIN_TRACK_FRAMES = 10
CLASS_CONSISTENCY_THRESHOLD = 0.80

# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Detection CSV not found:\n{INPUT_CSV}\n"
            "Run track_roadsentinel.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "class_name",
        "confidence",
        "x1", "y1", "x2", "y2",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    df["track_id"] = pd.to_numeric(
        df["track_id"], errors="coerce"
    )
    df["frame"] = pd.to_numeric(
        df["frame"], errors="coerce"
    )
    df["timestamp_seconds"] = pd.to_numeric(
        df["timestamp_seconds"], errors="coerce"
    )
    df["confidence"] = pd.to_numeric(
        df["confidence"], errors="coerce"
    )

    # Rows without a track ID cannot be used for track analysis.
    missing_id_rows = int(df["track_id"].isna().sum())
    df = df.dropna(
        subset=["track_id", "frame", "class_name"]
    ).copy()

    df["track_id"] = df["track_id"].astype(int)
    df["class_name"] = df["class_name"].astype(str)

    if df.empty:
        raise ValueError("No valid tracked detections found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # TRACK-LEVEL AUDIT
    # ========================================================

    rows = []

    for track_id, group in df.groupby("track_id", sort=True):
        group = group.sort_values("frame")

        class_counts = group["class_name"].value_counts()
        dominant_class = class_counts.index[0]
        dominant_count = int(class_counts.iloc[0])

        total_records = len(group)
        unique_frames = int(group["frame"].nunique())

        class_consistency = (
            dominant_count / total_records
            if total_records else 0.0
        )

        class_names = sorted(
            group["class_name"].dropna().unique().tolist()
        )

        first_frame = int(group["frame"].min())
        last_frame = int(group["frame"].max())

        first_time = float(
            group["timestamp_seconds"].min()
        )
        last_time = float(
            group["timestamp_seconds"].max()
        )

        observed_span_seconds = max(
            0.0, last_time - first_time
        )

        mean_confidence = float(
            group["confidence"].mean()
        )

        median_confidence = float(
            group["confidence"].median()
        )

        # Count missing frames between observed detections.
        # These gaps can occur naturally when objects are hidden.
        frame_numbers = sorted(
            group["frame"].dropna().astype(int).unique()
        )

        gaps = [
            frame_numbers[i] - frame_numbers[i - 1] - 1
            for i in range(1, len(frame_numbers))
        ]

        largest_gap_frames = max(gaps, default=0)

        reasons = []

        if unique_frames < MIN_TRACK_FRAMES:
            reasons.append("SHORT_TRACK")

        if (
            unique_frames >= MIN_FRAMES_FOR_CLASS_AUDIT
            and class_consistency < CLASS_CONSISTENCY_THRESHOLD
        ):
            reasons.append("CLASS_INCONSISTENCY")

        if len(class_names) > 1:
            reasons.append("MULTIPLE_PREDICTED_CLASSES")

        if largest_gap_frames >= 30:
            reasons.append("LARGE_OBSERVATION_GAP")

        rows.append({
            "track_id": track_id,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "first_time_seconds": round(first_time, 3),
            "last_time_seconds": round(last_time, 3),
            "observed_span_seconds": round(
                observed_span_seconds, 3
            ),
            "unique_observed_frames": unique_frames,
            "detection_records": total_records,
            "dominant_class": dominant_class,
            "dominant_class_records": dominant_count,
            "class_consistency": round(
                class_consistency, 4
            ),
            "classes_seen": " | ".join(class_names),
            "number_of_classes": len(class_names),
            "mean_confidence": round(mean_confidence, 4),
            "median_confidence": round(median_confidence, 4),
            "largest_gap_frames": largest_gap_frames,
            "audit_flags": " | ".join(reasons),
        })

    summary = pd.DataFrame(rows)

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    class_summary = (
        df.groupby("class_name")
        .agg(
            detection_records=("track_id", "size"),
            unique_track_ids=("track_id", "nunique"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("detection_records", ascending=False)
    )

    class_summary["mean_confidence"] = (
        class_summary["mean_confidence"].round(4)
    )

    # ========================================================
    # FLAG TRACKS FOR REVIEW
    # ========================================================

    flagged = summary[
        summary["audit_flags"].astype(str).str.len() > 0
    ].copy()

    summary = summary.sort_values(
        ["unique_observed_frames", "detection_records"],
        ascending=[False, False],
    )

    flagged = flagged.sort_values(
        ["number_of_classes", "class_consistency"],
        ascending=[False, True],
    )

    summary.to_csv(TRACK_SUMMARY_CSV, index=False)
    class_summary.to_csv(CLASS_SUMMARY_CSV, index=False)
    flagged.to_csv(FLAGGED_TRACKS_CSV, index=False)

    # ========================================================
    # GENERATE HUMAN-READABLE REPORT
    # ========================================================

    total_records = len(df)
    unique_tracks = int(df["track_id"].nunique())
    unique_frames = int(df["frame"].nunique())

    short_tracks = int(
        (summary["unique_observed_frames"] < MIN_TRACK_FRAMES).sum()
    )

    inconsistent_tracks = int(
        (
            (summary["unique_observed_frames"]
             >= MIN_FRAMES_FOR_CLASS_AUDIT)
            & (summary["class_consistency"]
               < CLASS_CONSISTENCY_THRESHOLD)
        ).sum()
    )

    multi_class_tracks = int(
        (summary["number_of_classes"] > 1).sum()
    )

    long_tracks = int(
        (summary["unique_observed_frames"] >= 30).sum()
    )

    report_lines = [
        "=" * 64,
        "ROADSENTINEL TRACKING QUALITY AUDIT",
        "=" * 64,
        "",
        f"Input CSV: {INPUT_CSV}",
        f"Detection records analyzed: {total_records}",
        f"Unique observed frames: {unique_frames}",
        f"Unique track IDs: {unique_tracks}",
        f"Rows without usable track IDs removed: {missing_id_rows}",
        "",
        "TRACK QUALITY COUNTS",
        "-" * 64,
        f"Tracks with fewer than {MIN_TRACK_FRAMES} observed frames: "
        f"{short_tracks}",
        f"Tracks with multiple predicted classes: {multi_class_tracks}",
        f"Tracks with class consistency below "
        f"{CLASS_CONSISTENCY_THRESHOLD:.0%}: {inconsistent_tracks}",
        f"Tracks seen in at least 30 frames: {long_tracks}",
        f"Tracks flagged for review: {len(flagged)}",
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 64,
        "These are diagnostic indicators, not ground-truth tracking metrics.",
        "Short tracks can be legitimate for objects entering or leaving view.",
        "Class changes may indicate classification instability or ID switches.",
        "A long track does not prove that the ID follows one real object.",
        "The CSV contains only detections passing its confidence filter.",
        "",
        "OUTPUT FILES",
        "-" * 64,
        str(TRACK_SUMMARY_CSV),
        str(CLASS_SUMMARY_CSV),
        str(FLAGGED_TRACKS_CSV),
        "",
    ]

    REPORT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    print("Audit complete.")
    print(f"Full report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
