
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "datasets/engagement/raw/youtube_comment_sample.csv"
)

OUTPUT_FILE = Path(
    "datasets/engagement/raw/youtube_annotation_queue.csv"
)

SAMPLE_PER_SENTIMENT = 100
RANDOM_STATE = 42

def main():

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Review it before running again."
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 65)
    print("MODERAI - PREPARING ANNOTATION QUEUE")
    print("=" * 65)

    # Normalize text for duplicate detection
    normalized = (
        df["comment_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.split()
        .str.join(" ")
    )

    df = df.assign(normalized_text=normalized)

    before = len(df)

    # Remove empty comments and exact normalized duplicates
    df = df[df["normalized_text"] != ""]
    df = df.drop_duplicates(
        subset="normalized_text",
        keep="first"
    )

    print("Original records:", before)
    print("After duplicate removal:", len(df))
    print("Records removed:", before - len(df))

    # Select 100 comments from each original sentiment
    selected = []

    for sentiment in ["Positive", "Neutral", "Negative"]:

        group = df[
            df["original_sentiment"].str.casefold()
            == sentiment.casefold()
        ]

        if len(group) < SAMPLE_PER_SENTIMENT:
            raise ValueError(
                f"Not enough records for {sentiment}"
            )

        sample = group.sample(
            n=SAMPLE_PER_SENTIMENT,
            random_state=RANDOM_STATE
        )

        selected.append(sample)

    result = pd.concat(
        selected,
        ignore_index=True
    )

    # Shuffle the annotation order
    result = result.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Build annotation identifiers
    result.insert(
        0,
        "comment_id",
        [f"YT{i:06d}" for i in range(1, len(result) + 1)]
    )

    # Prepare annotation fields
    result["intent"] = ""
    result["annotator"] = "pending_review"

    # Preserve source and attribution information
    result["source"] = "public_dataset"

    columns = [
        "comment_id",
        "comment_text",
        "intent",
        "source",
        "annotator",
        "original_sentiment",
        "video_title",
        "source_dataset",
        "source_license"
    ]

    result = result[columns]

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nANNOTATION QUEUE SUMMARY")
    print("Total comments:", len(result))

    print("\nOriginal sentiment distribution:")
    print(result["original_sentiment"].value_counts())

    print("\nOutput:", OUTPUT_FILE)

    print("\n[PASS] Annotation queue created.")

if __name__ == "__main__":
    main()
