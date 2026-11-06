
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "moderai_50k"
)

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv"
}

EXPECTED_ROWS = {
    "train": 40_000,
    "validation": 5_000,
    "test": 5_000
}

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

REQUIRED_COLUMNS = [
    "text",
    "occurrence_count",
    *LABELS
]


def main():

    print("=" * 70)
    print("MODERAI 50K DATASET VALIDATION")
    print("=" * 70)

    datasets = {}
    total_rows = 0
    all_texts = set()

    for split_name, file_path in FILES.items():

        print(f"\nChecking {split_name.upper()}...")

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path)

        # Check schema.
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in {split_name}: {missing_columns}"
            )

        # Check expected row count.
        expected = EXPECTED_ROWS[split_name]

        if len(df) != expected:
            raise ValueError(
                f"{split_name}: expected {expected}, got {len(df)}"
            )

        # Check missing values.
        if df["text"].isna().any():
            raise ValueError(f"Missing comment text in {split_name}")

        if df[LABELS].isna().any().any():
            raise ValueError(f"Missing annotation scores in {split_name}")

        if df["occurrence_count"].isna().any():
            raise ValueError(
                f"Missing occurrence counts in {split_name}"
            )

        # Check annotation score ranges.
        for label in LABELS:

            if not df[label].between(0, 1).all():
                raise ValueError(
                    f"Invalid scores in {split_name}: {label}"
                )

        # Check duplicate comments within each split.
        duplicate_count = int(
            df["text"].duplicated().sum()
        )

        if duplicate_count:
            raise ValueError(
                f"{duplicate_count} duplicate comments in {split_name}"
            )

        # Check overlap between splits.
        current_texts = set(df["text"])

        overlap = all_texts.intersection(current_texts)

        if overlap:
            raise ValueError(
                f"Found {len(overlap)} comments repeated across splits"
            )

        all_texts.update(current_texts)

        datasets[split_name] = df

        total_rows += len(df)

        print("Rows:", f"{len(df):,}")
        print("Unique comments:", f"{df['text'].nunique():,}")
        print("Missing values:", int(df.isna().sum().sum()))
        print("Duplicate comments:", duplicate_count)

    # Final split-size check.
    if total_rows != 50_000:
        raise ValueError(
            f"Expected 50,000 total records, got {total_rows}"
        )

    print("\n" + "=" * 70)
    print("LABEL POSITIVE COUNTS")
    print("=" * 70)

    for split_name, df in datasets.items():

        print(f"\n{split_name.upper()}")

        for label in LABELS:

            positives = int((df[label] >= 0.5).sum())

            print(
                f"{label:20} {positives:7,}"
            )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    print("Training records:   ", f"{len(datasets['train']):,}")
    print("Validation records: ", f"{len(datasets['validation']):,}")
    print("Test records:       ", f"{len(datasets['test']):,}")
    print("Total records:      ", f"{total_rows:,}")
    print("Unique comments:    ", f"{len(all_texts):,}")
    print("Annotation columns: ", len(LABELS))

    print("\nDATASET VALIDATION: PASSED")


if __name__ == "__main__":
    main()
