import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "civil_comments_clean"
)

SPLITS = {
    "train": 1_771_879,
    "validation": 98_438,
    "test": 98_438
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

EXPECTED_TOTAL_ROWS = 1_968_755
EXPECTED_OCCURRENCES = 1_981_413
EXPECTED_REPEATED_ROWS = 5_982

CHUNK_SIZE = 100_000


def validate_split(name, expected_rows):

    file_path = DATA_DIR / f"{name}.csv"

    print("\n" + "=" * 65)
    print(f"VALIDATING: {name.upper()}")
    print("=" * 65)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    total_rows = 0
    total_occurrences = 0
    repeated_rows = 0

    missing_text = 0
    missing_labels = 0
    invalid_scores = 0

    positive_counts = {
        label: 0 for label in LABELS
    }

    score_sums = {
        label: 0.0 for label in LABELS
    }

    required_columns = [
        "text",
        "occurrence_count",
        *LABELS
    ]

    reader = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        if not set(required_columns).issubset(chunk.columns):
            raise ValueError(
                f"Required columns missing in {name}"
            )

        total_rows += len(chunk)

        missing_text += chunk["text"].isna().sum()

        missing_labels += (
            chunk[LABELS].isna().sum().sum()
        )

        total_occurrences += (
            chunk["occurrence_count"].sum()
        )

        repeated_rows += (
            chunk["occurrence_count"] > 1
        ).sum()

        for label in LABELS:

            values = chunk[label]

            invalid_scores += (
                ((values < 0) | (values > 1)).sum()
            )

            positive_counts[label] += (
                (values >= 0.5).sum()
            )

            score_sums[label] += values.sum()

        print(
            f"Validated {total_rows:,} records...",
            end="\r"
        )

    print("\n")

    print("Expected records:", f"{expected_rows:,}")
    print("Actual records:", f"{total_rows:,}")
    print("Original occurrences:", f"{total_occurrences:,}")
    print("Repeated-comment rows:", f"{repeated_rows:,}")
    print("Missing text:", missing_text)
    print("Missing labels:", missing_labels)
    print("Invalid scores:", invalid_scores)

    return {
        "name": name,
        "rows": total_rows,
        "expected_rows": expected_rows,
        "occurrences": total_occurrences,
        "repeated_rows": repeated_rows,
        "missing_text": missing_text,
        "missing_labels": missing_labels,
        "invalid_scores": invalid_scores,
        "positive_counts": positive_counts,
        "score_sums": score_sums
    }


def main():

    print("=" * 65)
    print("MODERAI CLEAN SPLIT VALIDATION")
    print("=" * 65)

    results = []

    for name, expected_rows in SPLITS.items():

        result = validate_split(
            name,
            expected_rows
        )

        results.append(result)

    total_rows = sum(r["rows"] for r in results)

    total_occurrences = sum(
        r["occurrences"] for r in results
    )

    total_repeated = sum(
        r["repeated_rows"] for r in results
    )

    total_missing_text = sum(
        r["missing_text"] for r in results
    )

    total_missing_labels = sum(
        r["missing_labels"] for r in results
    )

    total_invalid_scores = sum(
        r["invalid_scores"] for r in results
    )

    print("\n" + "=" * 65)
    print("COMBINED DATASET VALIDATION")
    print("=" * 65)

    print("Total records:", f"{total_rows:,}")
    print("Original occurrences:", f"{total_occurrences:,}")
    print("Repeated-comment rows:", f"{total_repeated:,}")
    print("Missing text:", total_missing_text)
    print("Missing labels:", total_missing_labels)
    print("Invalid scores:", total_invalid_scores)

    print("\n" + "=" * 65)
    print("POSITIVE LABEL DISTRIBUTION")
    print("=" * 65)

    for label in LABELS:

        positives = sum(
            r["positive_counts"][label]
            for r in results
        )

        percentage = positives / total_rows * 100

        print(
            f"{label:20} "
            f"{positives:10,} "
            f"({percentage:.4f}%)"
        )

    passed = (
        total_rows == EXPECTED_TOTAL_ROWS
        and total_occurrences == EXPECTED_OCCURRENCES
        and total_repeated == EXPECTED_REPEATED_ROWS
        and total_missing_text == 0
        and total_missing_labels == 0
        and total_invalid_scores == 0
        and all(
            r["rows"] == r["expected_rows"]
            for r in results
        )
    )

    print("\n" + "=" * 65)

    if passed:
        print("CLEAN SPLIT VALIDATION: PASSED")
    else:
        print("CLEAN SPLIT VALIDATION: REVIEW REQUIRED")

    print("=" * 65)


if __name__ == "__main__":
    main()