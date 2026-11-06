
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Configuration
DATASET_NAME = "AmaanP314/youtube-comment-sentiment"
TARGET_ROWS = 1000
MAX_ROWS_TO_SCAN = 5000

OUTPUT_DIR = Path("datasets/engagement/raw")
OUTPUT_FILE = OUTPUT_DIR / "youtube_comment_sample.csv"

LICENSE = "CC BY-SA 4.0"

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Rename or review it before running again."
        )

    print("=" * 65)
    print("MODERAI - YOUTUBE COMMENT SAMPLE COLLECTION")
    print("=" * 65)

    print("\nConnecting to Hugging Face dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    collected = []
    scanned = 0
    excluded_augmented = 0

    print("\nStreaming dataset records...")

    for row in dataset:

        scanned += 1

        if scanned > MAX_ROWS_TO_SCAN:
            break

        comment = str(row.get("CommentText", "")).strip()
        author = str(row.get("AuthorName", "")).strip()

        if not comment or comment.lower() == "nan":
            continue

        # Exclude records explicitly marked as augmented
        if author.casefold() == "augmenteduser":
            excluded_augmented += 1
            continue

        collected.append({
            "comment_text": comment,
            "video_title": str(row.get("VideoTitle", "")),
            "original_sentiment": str(row.get("Sentiment", "")),
            "likes": row.get("Likes", 0),
            "replies": row.get("Replies", 0),
            "published_at": str(row.get("PublishedAt", "")),
            "country_code": str(row.get("CountryCode", "")),
            "source": "public_dataset",
            "source_dataset": DATASET_NAME,
            "source_license": LICENSE
        })

        if len(collected) >= TARGET_ROWS:
            break

        if scanned % 250 == 0:
            print(
                f"Scanned: {scanned} | "
                f"Collected: {len(collected)}"
            )

    if not collected:
        raise ValueError("No usable comments were collected.")

    df = pd.DataFrame(collected)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("COLLECTION SUMMARY")
    print("=" * 65)

    print("Rows scanned:", scanned)
    print("Comments collected:", len(df))
    print("Augmented records excluded:", excluded_augmented)
    print("Output:", OUTPUT_FILE)

    print("\nOriginal sentiment distribution:")
    print(df["original_sentiment"].value_counts(dropna=False))

    print("\nSample comments:")
    print(df[["comment_text", "original_sentiment"]].head(5).to_string(index=False))

    print("\nDOWNLOAD COMPLETED")

if __name__ == "__main__":
    main()
