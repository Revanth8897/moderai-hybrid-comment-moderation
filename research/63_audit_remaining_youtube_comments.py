
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_comment_sample.csv"
)

ANNOTATED_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv"
)


def find_column(df, candidates, description):
    lookup = {
        str(column).strip().lower().replace("_", "").replace(" ", ""):
        column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace(" ", "")
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    return value


def main():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(SAMPLE_PATH)

    if not ANNOTATED_PATH.exists():
        raise FileNotFoundError(ANNOTATED_PATH)

    sample = pd.read_csv(SAMPLE_PATH)
    annotated = pd.read_csv(ANNOTATED_PATH)

    print("\nMODERAI — REMAINING YOUTUBE COMMENT AUDIT")
    print("=" * 65)

    print("\nOriginal sample columns:")
    print(sample.columns.tolist())

    print("\nAnnotated queue columns:")
    print(annotated.columns.tolist())

    sample_text_col = find_column(
        sample,
        ["comment_text", "comment", "comments", "text"],
        "sample comment text",
    )

    annotated_text_col = find_column(
        annotated,
        ["comment_text", "comment", "comments", "text"],
        "annotated comment text",
    )

    sentiment_col = find_column(
        sample,
        ["sentiment", "original_sentiment"],
        "original sentiment",
    )

    title_col = find_column(
        sample,
        ["video_title", "VideoTitle", "title"],
        "video title",
    )

    sample["_normalized_text"] = (
        sample[sample_text_col].apply(normalize_text)
    )

    annotated["_normalized_text"] = (
        annotated[annotated_text_col].apply(normalize_text)
    )

    # Ignore blank comments.
    sample = sample[
        sample["_normalized_text"].ne("")
    ].copy()

    annotated = annotated[
        annotated["_normalized_text"].ne("")
    ].copy()

    # Identify comments already annotated.
    annotated_texts = set(annotated["_normalized_text"])

    sample["_already_annotated"] = (
        sample["_normalized_text"].isin(annotated_texts)
    )

    remaining = sample[
        ~sample["_already_annotated"]
    ].copy()

    # Remove repeated copies of the same remaining comment.
    remaining_unique = remaining.drop_duplicates(
        subset=["_normalized_text"]
    ).copy()

    print("\nDATASET COUNTS")
    print("-" * 65)

    print("Original sample rows:", len(pd.read_csv(SAMPLE_PATH)))
    print("Usable non-empty sample rows:", len(sample))
    print("Annotated queue rows:", len(annotated))
    print(
        "Unique comments already annotated:",
        len(annotated_texts),
    )
    print(
        "Sample rows matching annotated comments:",
        int(sample["_already_annotated"].sum()),
    )
    print("Remaining unannotated rows:", len(remaining))
    print(
        "Remaining unique comments:",
        len(remaining_unique),
    )
    print(
        "Duplicate copies among remaining rows:",
        len(remaining) - len(remaining_unique),
    )

    print("\nREMAINING COMMENTS BY ORIGINAL SENTIMENT")
    print("-" * 65)

    print(
        remaining_unique[sentiment_col]
        .fillna("MISSING")
        .value_counts()
        .to_string()
    )

    print("\nVIDEO METADATA")
    print("-" * 65)

    titles = (
        remaining_unique[title_col]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    print("Comments with video title:", titles.ne("").sum())
    print("Unique video titles:", titles[titles.ne("")].nunique())

    print("\nSAMPLE OF 10 REMAINING COMMENTS")
    print("-" * 65)

    for _, row in remaining_unique.head(10).iterrows():
        print("\nComment:", str(row[sample_text_col])[:300])
        print("Original sentiment:", row[sentiment_col])
        print("Video title:", row[title_col])

    print("\nAudit completed.")
    print("No files were modified.")


if __name__ == "__main__":
    main()
