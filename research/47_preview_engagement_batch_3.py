
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

start_id = 41
end_id = 100

batch_ids = [
    f"YT{i:06d}"
    for i in range(start_id, end_id + 1)
]

batch = df[
    df["comment_id"].astype(str).isin(batch_ids)
].copy()

batch["sort_order"] = batch["comment_id"].map(
    {comment_id: index for index, comment_id in enumerate(batch_ids)}
)

batch = batch.sort_values("sort_order")

if len(batch) != 60:
    raise ValueError(
        f"Expected 60 comments, found {len(batch)}"
    )

print("=" * 85)
print("MODERAI - ENGAGEMENT ANNOTATION BATCH 3")
print("=" * 85)

for _, row in batch.iterrows():
    print(f"\nID: {row['comment_id']}")
    print(f"Original sentiment: {row['original_sentiment']}")
    print(f"Video: {row['video_title']}")
    print(f"Comment: {row['comment_text']}")
    print("-" * 85)

print("\nTotal comments:", len(batch))
print("Batch 3 preview completed.")
