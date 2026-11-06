
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "engagement"
    / "raw"
    / "youtube_annotation_queue.csv"
)

df = pd.read_csv(DATASET_PATH)

new_columns = {
    "annotation_confidence": "",
    "annotation_notes": "",
}

for column, default_value in new_columns.items():
    if column not in df.columns:
        df[column] = default_value

df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")

print("Annotation quality columns added.")
print(f"Total comments: {len(df)}")
print("\nUpdated columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(
    df[
        [
            "comment_id",
            "intent",
            "annotation_confidence",
            "annotation_notes",
        ]
    ].head().to_string(index=False)
)
