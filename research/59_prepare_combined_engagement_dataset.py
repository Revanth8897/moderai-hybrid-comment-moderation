
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REAL_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv"
)

SEED_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "engagement_comments.csv"
)

OUTPUT_DIR = ROOT / "datasets" / "engagement" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "engagement_all_330.csv"
REPORT_PATH = (
    ROOT / "reports" / "engagement"
    / "combined_intent_distribution.csv"
)

VALID_INTENTS = {
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
    "UNCLEAR",
}


def prepare_real_data():
    df = pd.read_csv(REAL_PATH)

    df = df.rename(columns={"comment_text": "text"})

    df["dataset_type"] = "real_world"

    return df


def prepare_synthetic_data():
    df = pd.read_csv(SEED_PATH)

    df["dataset_type"] = "synthetic"

    return df


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Review it before replacing it."
        )

    real_df = prepare_real_data()
    synthetic_df = prepare_synthetic_data()

    # Standardized schema. Original metadata is retained.
    columns = [
        "comment_id",
        "text",
        "intent",
        "dataset_type",
        "source",
        "annotator",
        "annotation_confidence",
        "annotation_notes",
        "original_sentiment",
        "video_title",
        "source_dataset",
        "source_license",
    ]

    combined = pd.concat(
        [real_df, synthetic_df],
        ignore_index=True,
        sort=False,
    )

    # Add any absent optional metadata columns.
    for column in columns:
        if column not in combined.columns:
            combined[column] = ""

    combined = combined[columns]

    # Normalize text and labels.
    for column in columns:
        combined[column] = (
            combined[column].astype("string").fillna("")
        )

    combined["comment_id"] = combined["comment_id"].str.strip()
    combined["text"] = combined["text"].str.strip()
    combined["intent"] = (
        combined["intent"].str.strip().str.upper()
    )

    # Validate dataset.
    if len(combined) != 330:
        raise ValueError(
            f"Expected 330 rows, found {len(combined)}"
        )

    if combined["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs found")

    if combined["text"].eq("").any():
        raise ValueError("Empty comments found")

    invalid = set(combined["intent"]) - VALID_INTENTS

    if invalid:
        raise ValueError(f"Invalid intent labels: {invalid}")

    if set(combined["dataset_type"]) != {
        "real_world",
        "synthetic",
    }:
        raise ValueError("Unexpected dataset types")

    # Verify source-specific counts.
    source_counts = combined["dataset_type"].value_counts()

    if source_counts.get("real_world", 0) != 300:
        raise ValueError("Real-world row count mismatch")

    if source_counts.get("synthetic", 0) != 30:
        raise ValueError("Synthetic row count mismatch")

    # Save combined dataset.
    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Generate intent distribution report.
    distribution = pd.crosstab(
        combined["intent"],
        combined["dataset_type"],
    )

    distribution["total"] = distribution.sum(axis=1)
    distribution["percentage"] = (
        distribution["total"] / len(combined) * 100
    ).round(2)

    distribution = distribution.reset_index()

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Read back and verify saved file.
    saved = pd.read_csv(OUTPUT_PATH)

    if len(saved) != 330:
        raise ValueError("Saved dataset row count mismatch")

    if saved["comment_id"].duplicated().any():
        raise ValueError("Saved dataset contains duplicate IDs")

    print("\nMODERAI — COMBINED ENGAGEMENT DATASET")
    print("=" * 60)

    print("\nDataset size:", len(saved))

    print("\nSource distribution:")
    print(saved["dataset_type"].value_counts().to_string())

    print("\nIntent distribution by source:")
    print(distribution.to_string(index=False))

    print("\nTotal intent distribution:")
    print(
        saved["intent"]
        .value_counts()
        .to_string()
    )

    print("\nOutput files:")
    print(OUTPUT_PATH)
    print(REPORT_PATH)

    print("\nVerification: PASSED")


if __name__ == "__main__":
    main()
