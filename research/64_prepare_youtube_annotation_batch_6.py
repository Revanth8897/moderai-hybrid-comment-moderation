
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

OUTPUT_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_batch_6.csv"
)

RANDOM_STATE = 42

QUOTAS = {
    "Positive": 34,
    "Neutral": 33,
    "Negative": 33,
}


def normalize_text(value):
    value = str(value).strip().lower()
    return re.sub(r"\s+", " ", value)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Review it before creating another batch."
        )

    sample = pd.read_csv(SAMPLE_PATH)
    annotated = pd.read_csv(ANNOTATED_PATH)

    # Identify comments already annotated.
    annotated_texts = set(
        annotated["comment_text"]
        .dropna()
        .map(normalize_text)
    )

    sample["_normalized_text"] = (
        sample["comment_text"]
        .fillna("")
        .map(normalize_text)
    )

    remaining = sample[
        sample["_normalized_text"].ne("")
        & ~sample["_normalized_text"].isin(annotated_texts)
    ].copy()

    # Remove duplicate comment text.
    remaining = remaining.drop_duplicates(
        subset=["_normalized_text"]
    )

    # Select exact quotas from original sentiment categories.
    selected_parts = []

    for sentiment, count in QUOTAS.items():
        pool = remaining[
            remaining["original_sentiment"] == sentiment
        ]

        if len(pool) < count:
            raise ValueError(
                f"Not enough {sentiment} comments. "
                f"Available: {len(pool)}, required: {count}"
            )

        selected_parts.append(
            pool.sample(
                n=count,
                random_state=RANDOM_STATE,
            )
        )

    batch = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # Shuffle the final batch so categories are not grouped.
    batch = batch.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # Prepare annotation fields.
    batch.insert(
        0,
        "comment_id",
        [f"YT{i:06d}" for i in range(301, 401)],
    )

    batch["intent"] = ""
    batch["annotator"] = "pending_review"
    batch["annotation_confidence"] = ""
    batch["annotation_notes"] = ""
    batch["intent_confidence"] = ""
    batch["annotation_note"] = ""

    # Keep useful original context and metadata.
    columns = [
        "comment_id",
        "comment_text",
        "intent",
        "source",
        "annotator",
        "original_sentiment",
        "video_title",
        "source_dataset",
        "source_license",
        "annotation_confidence",
        "annotation_notes",
        "intent_confidence",
        "annotation_note",
        "likes",
        "replies",
        "published_at",
        "country_code",
    ]

    for column in columns:
        if column not in batch.columns:
            batch[column] = ""

    batch = batch[columns]

    # Verify the batch.
    if len(batch) != 100:
        raise ValueError("Expected exactly 100 comments")

    if batch["comment_id"].duplicated().any():
        raise ValueError("Duplicate IDs in new batch")

    batch_texts = set(
        batch["comment_text"].map(normalize_text)
    )

    if batch_texts & annotated_texts:
        raise ValueError("Previously annotated comments detected")

    if batch["intent"].ne("").any():
        raise ValueError("Intent labels must be blank for review")

    if not batch["annotator"].eq("pending_review").all():
        raise ValueError("Unexpected annotator status")

    batch.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Read back and verify.
    saved = pd.read_csv(OUTPUT_PATH)

    if len(saved) != 100:
        raise ValueError("Saved batch row count mismatch")

    print("\nMODERAI — ANNOTATION BATCH 6")
    print("=" * 60)

    print("Total comments:", len(saved))

    print("\nOriginal sentiment distribution:")
    print(saved["original_sentiment"].value_counts().to_string())

    print("\nComment ID range:")
    print(saved["comment_id"].iloc[0], "to", saved["comment_id"].iloc[-1])

    print("\nAnnotation status:")
    print(saved["annotator"].value_counts().to_string())

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nVerification: PASSED")
    print("No previously annotated comments included.")
    print("No duplicate comments included.")
    print("All 100 intent labels are blank for human review.")


if __name__ == "__main__":
    main()
