
import pandas as pd
from pathlib import Path

DATASET = Path(
    "datasets/engagement/raw/engagement_comments.csv"
)

VALID_LABELS = {
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
    "UNCLEAR"
}

REQUIRED_COLUMNS = [
    "comment_id",
    "text",
    "intent",
    "source",
    "annotator"
]


def main():

    df = pd.read_csv(DATASET)

    print("=" * 60)
    print("MODERAI ENGAGEMENT DATASET VALIDATION")
    print("=" * 60)

    errors = []

    # 1. Required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    else:
        print("\n[PASS] Required columns exist.")

    # 2. Empty values
    if df[REQUIRED_COLUMNS].isnull().any().any():
        errors.append("Missing values found.")

    else:
        print("[PASS] No missing values.")

    # 3. Empty comment text
    if df["text"].astype(str).str.strip().eq("").any():
        errors.append("Empty comment text found.")

    else:
        print("[PASS] No empty comments.")

    # 4. Duplicate IDs
    if df["comment_id"].duplicated().any():
        errors.append("Duplicate comment IDs found.")

    else:
        print("[PASS] Comment IDs are unique.")

    # 5. Duplicate text
    normalized = df["text"].astype(str).str.strip().str.casefold()

    duplicate_count = normalized.duplicated().sum()

    print(f"[INFO] Duplicate normalized comments: {duplicate_count}")

    # 6. Valid labels
    invalid_labels = set(df["intent"].unique()) - VALID_LABELS

    if invalid_labels:
        errors.append(f"Invalid labels: {invalid_labels}")

    else:
        print("[PASS] All intent labels are valid.")

    # 7. Class distribution
    print("\nCLASS DISTRIBUTION")

    distribution = df["intent"].value_counts()

    print(distribution.sort_index())

    missing_labels = VALID_LABELS - set(df["intent"].unique())

    if missing_labels:
        errors.append(f"Missing classes: {missing_labels}")

    # 8. Dataset summary
    print("\nDATASET SUMMARY")

    print("Total comments:", len(df))
    print("Unique classes:", df["intent"].nunique())

    print("\nVALIDATION RESULT")

    if errors:
        for error in errors:
            print("[FAIL]", error)

        raise SystemExit("DATASET VALIDATION FAILED")

    print("DATASET VALIDATION PASSED")


if __name__ == "__main__":
    main()
