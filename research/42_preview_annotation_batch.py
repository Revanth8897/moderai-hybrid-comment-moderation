
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "datasets/engagement/raw/youtube_annotation_queue.csv"
)

BATCH_SIZE = 20

def main():

    df = pd.read_csv(INPUT_FILE)

    batch = df.head(BATCH_SIZE)

    print("=" * 70)
    print("MODERAI - ANNOTATION BATCH 1")
    print("=" * 70)

    for _, row in batch.iterrows():

        print("\n" + "-" * 70)
        print("ID:", row["comment_id"])
        print("Original sentiment:", row["original_sentiment"])
        print("Comment:", row["comment_text"])
        print("Video:", row["video_title"])

    print("\n" + "=" * 70)
    print("TOTAL COMMENTS:", len(batch))
    print("=" * 70)

if __name__ == "__main__":
    main()
