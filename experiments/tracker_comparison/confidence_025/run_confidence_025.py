
from pathlib import Path
import sys

# Reuse the existing comparison script without modifying it.
ROOT = Path(__file__).resolve().parents[3]

SOURCE_SCRIPT = (
    ROOT
    / "experiments"
    / "tracker_comparison"
    / "compare_trackers.py"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "tracker_comparison"
    / "confidence_025"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the original experiment script.
source = SOURCE_SCRIPT.read_text(encoding="utf-8")

# Lower detection confidence.
source = source.replace(
    "CONFIDENCE = 0.35",
    "CONFIDENCE = 0.25",
)

# Change output folder to avoid overwriting previous experiments.
source = source.replace(
    'OUTPUT_DIR = ROOT / "experiments" / "tracker_comparison"',
    'OUTPUT_DIR = ROOT / "experiments" / "tracker_comparison" / "confidence_025"',
)

# Write a separate copy of the modified script.
modified_script = OUTPUT_DIR / "compare_trackers_conf025.py"

modified_script.write_text(
    source,
    encoding="utf-8",
)

print("Created experiment script:")
print(modified_script)

print("\nConfidence setting:")
print("CONFIDENCE = 0.25")

print("\nOutput directory:")
print(OUTPUT_DIR)
