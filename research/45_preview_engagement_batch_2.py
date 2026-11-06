
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

batch = df[
    df["comment_id"].astype(str).isin(
        [f"YT{i:06d}" for i in range(21, 41)]
    )
]

if len(batch) != 20:
    raise ValueError(
        f"Expected 20 comments, found {len(batch)}"
    )

print("=" * 75)
print("MODERAI - ENGAGEMENT ANNOTATION BATCH 2")
print("=" * 75)

for _, row in batch.iterrows():
    print(f"\nID: {row['comment_id']}")
    print(f"Original sentiment: {row['original_sentiment']}")
    print(f"Video: {row['video_title']}")
    print(f"Comment: {row['comment_text']}")
    print("-" * 75)

print("\nBatch 2 preview completed.")
