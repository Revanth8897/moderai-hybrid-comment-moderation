
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_batch_6.csv"
)

def main():
    df = pd.read_csv(INPUT_PATH)

    batch = df.head(20)

    print("\nMODERAI — ANNOTATION BATCH 6")
    print("REVIEW GROUP 1: YT000301–YT000320")
    print("=" * 75)

    for _, row in batch.iterrows():
        print(f"\nID: {row['comment_id']}")
        print(f"Original sentiment: {row['original_sentiment']}")
        print(f"Comment: {row['comment_text']}")
        print(f"Video: {row['video_title']}")
        print("-" * 75)

    print("\nTotal comments displayed:", len(batch))


if __name__ == "__main__":
    main()
