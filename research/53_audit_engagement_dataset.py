
from pathlib import Path
import re
import pandas as pd
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "engagement"
    / "raw"
    / "youtube_annotation_queue.csv"
)

SEED_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "engagement"
    / "raw"
    / "engagement_comments.csv"
)

VALID_INTENTS = {
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
    "UNCLEAR",
}

VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}

EXPECTED_REAL_ROWS = 300
EXPECTED_SEED_ROWS = 30


def normalize_text(value):
    """Normalize text for duplicate detection."""
    if pd.isna(value):
        return ""

    text = str(value).casefold().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def audit_dataset(df, name, text_column, expected_rows):
    errors = []
    warnings = []

    print_section(f"{name} AUDIT")

    print(f"Total rows: {len(df)}")

    if len(df) != expected_rows:
        errors.append(
            f"Expected {expected_rows} rows, found {len(df)}"
        )

    required_columns = {
        "comment_id",
        text_column,
        "intent",
        "source",
        "annotator",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing columns: {sorted(missing_columns)}"
        )
        return errors, warnings

    # Missing values
    print("\nMissing values:")
    missing = df[list(required_columns)].isna().sum()
    print(missing.to_string())

    if missing.sum() > 0:
        errors.append("Missing values found in required columns")

    # Empty text
    empty_text = (
        df[text_column]
        .astype("string")
        .fillna("")
        .str.strip()
        .eq("")
    )

    print(f"\nEmpty comments: {empty_text.sum()}")

    if empty_text.any():
        errors.append("Empty comments found")

    # Duplicate IDs
    duplicate_ids = df["comment_id"].duplicated().sum()

    print(f"Duplicate comment IDs: {duplicate_ids}")

    if duplicate_ids > 0:
        errors.append("Duplicate comment IDs found")

    # Duplicate normalized text
    normalized = df[text_column].map(normalize_text)
    duplicate_texts = normalized.duplicated().sum()

    print(f"Duplicate normalized comments: {duplicate_texts}")

    if duplicate_texts > 0:
        warnings.append(
            f"{duplicate_texts} duplicate normalized comments in {name}"
        )

    # Intent validation
    intents = (
        df["intent"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )

    invalid_intents = sorted(set(intents) - VALID_INTENTS)
    missing_intents = intents.eq("").sum()

    print("\nIntent distribution:")
    print(intents.value_counts().to_string())

    print(f"\nMissing intent labels: {missing_intents}")
    print(f"Invalid intent labels: {invalid_intents}")

    if missing_intents > 0:
        errors.append("Missing intent labels found")

    if invalid_intents:
        errors.append(
            f"Invalid intent labels: {invalid_intents}"
        )

    missing_classes = VALID_INTENTS - set(intents)

    if missing_classes:
        warnings.append(
            f"Classes not represented in {name}: "
            f"{sorted(missing_classes)}"
        )

    # Annotator information
    print("\nAnnotator distribution:")
    print(df["annotator"].value_counts(dropna=False).to_string())

    return errors, warnings


def main():
    if not REAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Real dataset not found: {REAL_DATA_PATH}"
        )

    if not SEED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Seed dataset not found: {SEED_DATA_PATH}"
        )

    real_df = pd.read_csv(
        REAL_DATA_PATH,
        dtype={"comment_id": "string"},
    )

    seed_df = pd.read_csv(
        SEED_DATA_PATH,
        dtype={"comment_id": "string"},
    )

    all_errors = []
    all_warnings = []

    # --------------------------------------------------
    # 1. Audit real-world comments
    # --------------------------------------------------

    real_errors, real_warnings = audit_dataset(
        real_df,
        "REAL-WORLD DATASET",
        "comment_text",
        EXPECTED_REAL_ROWS,
    )

    all_errors.extend(real_errors)
    all_warnings.extend(real_warnings)

    # Additional real-world metadata checks
    print_section("REAL-WORLD METADATA CHECK")

    metadata_columns = [
        "original_sentiment",
        "video_title",
        "source_dataset",
        "source_license",
        "annotation_confidence",
        "annotation_notes",
    ]

    for column in metadata_columns:
        if column not in real_df.columns:
            all_errors.append(
                f"Missing metadata column: {column}"
            )
            print(f"{column}: MISSING COLUMN")
        else:
            missing_count = real_df[column].isna().sum()
            print(f"{column}: {missing_count} missing values")

    if "annotation_confidence" in real_df.columns:
        confidence = (
            real_df["annotation_confidence"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.upper()
        )

        invalid_confidence = sorted(
            set(confidence) - VALID_CONFIDENCE
        )

        print("\nConfidence distribution:")
        print(confidence.value_counts().to_string())

        if invalid_confidence:
            all_errors.append(
                f"Invalid confidence values: {invalid_confidence}"
            )

    if "annotator" in real_df.columns:
        unexpected_annotators = (
            real_df["annotator"]
            .astype("string")
            .fillna("")
            .ne("human_approved_assisted")
        )

        if unexpected_annotators.any():
            all_warnings.append(
                "Some real-world rows have an unexpected annotator value"
            )

    # --------------------------------------------------
    # 2. Audit synthetic seed comments
    # --------------------------------------------------

    seed_errors, seed_warnings = audit_dataset(
        seed_df,
        "SYNTHETIC SEED DATASET",
        "text",
        EXPECTED_SEED_ROWS,
    )

    all_errors.extend(seed_errors)
    all_warnings.extend(seed_warnings)

    # --------------------------------------------------
    # 3. Cross-dataset duplicate check
    # --------------------------------------------------

    print_section("CROSS-DATASET DUPLICATE CHECK")

    real_texts = set(
        real_df["comment_text"].map(normalize_text)
    )

    seed_texts = set(
        seed_df["text"].map(normalize_text)
    )

    real_texts.discard("")
    seed_texts.discard("")

    cross_duplicates = real_texts.intersection(seed_texts)

    print(
        "Comments appearing in both datasets:",
        len(cross_duplicates),
    )

    if cross_duplicates:
        all_warnings.append(
            "Some comments appear in both real-world and seed datasets"
        )

        print("\nDuplicate examples:")

        for text in list(cross_duplicates)[:5]:
            print("-", text[:150])

    # --------------------------------------------------
    # 4. Final summary
    # --------------------------------------------------

    print_section("FINAL DATASET AUDIT SUMMARY")

    print(f"Real-world comments: {len(real_df)}")
    print(f"Synthetic seed comments: {len(seed_df)}")
    print(f"Combined comments: {len(real_df) + len(seed_df)}")

    print(f"\nTotal errors: {len(all_errors)}")
    print(f"Total warnings: {len(all_warnings)}")

    if all_errors:
        print("\nERRORS:")

        for error in all_errors:
            print("-", error)

    if all_warnings:
        print("\nWARNINGS:")

        for warning in all_warnings:
            print("-", warning)

    if not all_errors:
        print("\nSTRUCTURAL AUDIT: PASSED")

        if all_warnings:
            print(
                "Review the warnings before preparing "
                "the training dataset."
            )
        else:
            print("All audit checks passed.")

    else:
        print("\nSTRUCTURAL AUDIT: FAILED")
        print("Resolve the errors before model training.")


if __name__ == "__main__":
    main()
