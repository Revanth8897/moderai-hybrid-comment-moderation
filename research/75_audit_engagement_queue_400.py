
from pathlib import Path
import pandas as pd
import re
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_FILE = (
    PROJECT_ROOT
    / "datasets/engagement/raw/youtube_annotation_queue.csv"
)

VALID_INTENTS = {
    "POSITIVE", "NEGATIVE", "CRITICAL",
    "QUESTION", "NEUTRAL", "UNCLEAR"
}

VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value))
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main():
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(f"Queue file not found: {QUEUE_FILE}")

    df = pd.read_csv(
        QUEUE_FILE,
        dtype="string",
        keep_default_na=False
    )

    # Support both known comment column names.
    if "comment_text" in df.columns:
        comment_column = "comment_text"
    elif "text" in df.columns:
        comment_column = "text"
    else:
        raise ValueError(
            "No comment text column found. "
            f"Available columns: {df.columns.tolist()}"
        )

    required_columns = [
        "comment_id",
        comment_column,
        "intent",
        "annotator",
        "annotation_confidence",
        "video_title",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    for column in required_columns:
        df[column] = (
            df[column].astype("string").fillna("").str.strip()
        )

    section("MODERAI — FULL ENGAGEMENT QUEUE AUDIT")

    print(f"Comment text column: {comment_column}")
    print(f"Total rows:           {len(df)}")
    print(f"Unique comment IDs:   {df['comment_id'].nunique()}")
    print(f"Unique video titles:  {df['video_title'].nunique()}")
    print(f"Unique comment texts: {df[comment_column].nunique()}")

    # 1. Record integrity
    section("1. RECORD INTEGRITY")

    duplicate_ids = df[
        df["comment_id"].duplicated(keep=False)
    ]

    blank_ids = df[df["comment_id"].eq("")]
    blank_text = df[df[comment_column].eq("")]
    blank_intents = df[df["intent"].eq("")]

    print(f"Expected rows:         400")
    print(f"Actual rows:           {len(df)}")
    print(f"Duplicate comment IDs: {len(duplicate_ids)}")
    print(f"Blank comment IDs:      {len(blank_ids)}")
    print(f"Blank comment texts:    {len(blank_text)}")
    print(f"Blank intent labels:    {len(blank_intents)}")

    if len(duplicate_ids):
        print("\nDuplicate IDs:")
        print(
            duplicate_ids["comment_id"]
            .drop_duplicates()
            .to_list()
        )

    # 2. Intent distribution
    section("2. INTENT DISTRIBUTION")

    intent_counts = df["intent"].value_counts()

    print(intent_counts.to_string())

    print("\nIntent percentages:")
    print(
        (intent_counts / len(df) * 100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    unknown_intents = sorted(
        set(df["intent"]) - VALID_INTENTS
    )

    missing_intents = sorted(
        VALID_INTENTS - set(df["intent"])
    )

    print(f"\nUnknown intent labels: {unknown_intents}")
    print(f"Labels with zero examples: {missing_intents}")

    if len(intent_counts) and intent_counts.min() > 0:
        print(
            "\nClass imbalance ratio:",
            round(intent_counts.max() / intent_counts.min(), 2)
        )

    # 3. Confidence distribution
    section("3. ANNOTATION CONFIDENCE")

    confidence_counts = df[
        "annotation_confidence"
    ].value_counts()

    print(confidence_counts.to_string())

    unknown_confidence = sorted(
        set(df["annotation_confidence"]) - VALID_CONFIDENCE
    )

    print(f"\nUnknown confidence values: {unknown_confidence}")

    print("\nConfidence by intent:")
    print(
        pd.crosstab(
            df["intent"],
            df["annotation_confidence"]
        ).to_string()
    )

    # 4. Annotator status
    section("4. ANNOTATOR STATUS")

    print(df["annotator"].value_counts().to_string())

    pending = df[
        df["annotator"].str.lower().str.contains(
            "pending", na=False
        )
    ]

    print(f"\nPending annotation rows: {len(pending)}")

    # 5. Duplicate comment text
    section("5. DUPLICATE COMMENT TEXT")

    df["_normalized_text"] = df[comment_column].map(normalize_text)

    duplicate_texts = df[
        df["_normalized_text"].duplicated(keep=False)
    ].sort_values("_normalized_text")

    print(
        "Rows involved in repeated normalized text:",
        len(duplicate_texts)
    )

    print(
        "Distinct repeated normalized texts:",
        duplicate_texts["_normalized_text"].nunique()
    )

    if len(duplicate_texts):
        print("\nRepeated comment examples:")
        print(
            duplicate_texts[
                ["comment_id", "intent", "video_title", comment_column]
            ].head(20).to_string(index=False)
        )

    # 6. Video title grouping
    section("6. VIDEO TITLE GROUPING")

    nonblank_titles = df[
        df["video_title"].ne("")
    ].copy()

    title_counts = (
        nonblank_titles.groupby("video_title")
        .size()
        .sort_values(ascending=False)
    )

    repeated_titles = title_counts[title_counts > 1]

    print(f"Rows with video titles:       {len(nonblank_titles)}")
    print(f"Blank video titles:           {df['video_title'].eq('').sum()}")
    print(
        "Unique nonblank video titles:",
        nonblank_titles["video_title"].nunique()
    )
    print(f"Repeated video titles:        {len(repeated_titles)}")
    print(
        "Comments in repeated groups:",
        int(repeated_titles.sum())
    )

    if len(repeated_titles):
        print("\nLargest repeated video groups:")
        print(repeated_titles.head(20).to_string())

        repeated_rows = nonblank_titles[
            nonblank_titles["video_title"].isin(
                repeated_titles.index
            )
        ]

        title_intent_counts = pd.crosstab(
            repeated_rows["video_title"],
            repeated_rows["intent"]
        )

        mixed_titles = title_intent_counts[
            (title_intent_counts > 0).sum(axis=1) > 1
        ]

        print(
            "\nRepeated titles with multiple intent labels:",
            len(mixed_titles)
        )

        if len(mixed_titles):
            print(mixed_titles.head(20).to_string())

    # 7. Source metadata
    section("7. SOURCE METADATA")

    for column in [
        "source",
        "dataset_type",
        "source_dataset",
        "source_license"
    ]:
        if column in df.columns:
            print(f"\n{column}:")
            print(
                df[column].value_counts(dropna=False)
                .head(15)
                .to_string()
            )

    # 8. Comment length
    section("8. COMMENT LENGTH")

    lengths = df[comment_column].str.len()

    print(f"Minimum:             {lengths.min()}")
    print(f"Median:              {lengths.median():.1f}")
    print(f"Mean:                {lengths.mean():.2f}")
    print(f"Maximum:             {lengths.max()}")
    print(f"Under 10 characters: {(lengths < 10).sum()}")
    print(f"Over 1000 characters: {(lengths > 1000).sum()}")

    # 9. Final checks
    section("9. AUDIT SUMMARY")

    checks = {
        "Exactly 400 rows": len(df) == 400,
        "No duplicate comment IDs": len(duplicate_ids) == 0,
        "No blank comment IDs": len(blank_ids) == 0,
        "No blank comment text": len(blank_text) == 0,
        "No blank intent labels": len(blank_intents) == 0,
        "All intent labels valid": len(unknown_intents) == 0,
        "All confidence values valid": len(unknown_confidence) == 0,
        "No pending annotations": len(pending) == 0,
        "All rows have video titles": df["video_title"].ne("").all(),
    }

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'CHECK'}] {name}")

    print(
        f"\nChecks passed: {sum(checks.values())}/{len(checks)}"
    )

    print("\nAudit completed. No files were modified.")


if __name__ == "__main__":
    main()
