
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_PATH = (
    PROJECT_ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_batch_6.csv"
)

df = pd.read_csv(BATCH_PATH, dtype="string").fillna("")

required_ids = [f"YT{i:06d}" for i in range(381, 401)]

text_column = next(
    (col for col in ("comment_text", "text") if col in df.columns),
    None
)

if text_column is None:
    raise ValueError("Could not find comment_text or text column.")

if "comment_id" not in df.columns:
    raise ValueError("Missing comment_id column.")

group = df[df["comment_id"].isin(required_ids)].copy()

missing_ids = sorted(set(required_ids) - set(group["comment_id"]))

if missing_ids:
    raise ValueError(f"Missing comment IDs: {missing_ids}")

if len(group) != 20:
    raise ValueError(f"Expected 20 comments, found {len(group)}.")

order = {comment_id: i for i, comment_id in enumerate(required_ids)}
group["sort_order"] = group["comment_id"].map(order)
group = group.sort_values("sort_order")

print("=" * 90)
print("MODERAI — ANNOTATION BATCH 6 — GROUP 5")
print("Comments: YT000381–YT000400")
print("=" * 90)

for _, row in group.iterrows():
    print(f"\nID: {row['comment_id']}")

    if "original_sentiment" in df.columns:
        print(f"Original sentiment: {row['original_sentiment']}")

    if "video_title" in df.columns:
        print(f"Video title: {row['video_title']}")

    print(f"Comment: {row[text_column]}")
    print("-" * 90)

print("\nPreview verification: PASSED")
print(f"Total comments: {len(group)}")
