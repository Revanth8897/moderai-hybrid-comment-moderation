
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "engagement"
    / "raw"
    / "youtube_annotation_queue.csv"
)

def main():
    df = pd.read_csv(
        CSV_PATH,
        dtype={"comment_id": "string"}
    )

    expected_ids = {
        f"YT{i:06d}" for i in range(201, 301)
    }

    batch = df[df["comment_id"].isin(expected_ids)].copy()

    if len(batch) != 100:
        raise ValueError(
            f"Expected 100 comments, found {len(batch)}."
        )

    if batch["intent"].fillna("").astype("string").str.strip().ne("").any():
        raise ValueError(
            "Some comments in this batch are already annotated."
        )

    batch["numeric_id"] = (
        batch["comment_id"]
        .str.replace("YT", "", regex=False)
        .astype(int)
    )

    batch = batch.sort_values("numeric_id")

    print("\nMODERAI — ENGAGEMENT ANNOTATION BATCH 5")
    print("=" * 75)
    print(f"Comments: {len(batch)}")
    print("Range: YT000201–YT000300")
    print("=" * 75)

    for _, row in batch.iterrows():
        print(f"\nID: {row['comment_id']}")
        print(f"Original sentiment: {row.get('original_sentiment', 'N/A')}")
        print(f"Comment:\n{row['comment_text']}")
        print("-" * 75)


if __name__ == "__main__":
    main()
