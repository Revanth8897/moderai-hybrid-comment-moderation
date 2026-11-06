
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT
    / "datasets"
    / "processed"
    / "civil_comments_clean"
    / "all_unique_comments.csv"
)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

CHUNK_SIZE = 100_000


def main():

    print("=" * 65)
    print("MODERAI AGGREGATED DATASET VALIDATION")
    print("=" * 65)

    total_rows = 0
    total_occurrences = 0

    missing_text = 0
    missing_labels = 0
    invalid_scores = 0

    positive_counts = {label: 0 for label in LABELS}
    score_sums = {label: 0.0 for label in LABELS}

    repeated_texts = 0

    reader = pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        total_rows += len(chunk)

        missing_text += chunk["text"].isna().sum()

        missing_labels += chunk[LABELS].isna().sum().sum()

        total_occurrences += chunk["occurrence_count"].sum()

        repeated_texts += (
            chunk["occurrence_count"] > 1
        ).sum()

        for label in LABELS:

            values = chunk[label]

            invalid_scores += (
                ((values < 0) | (values > 1))
                .sum()
            )

            positive_counts[label] += (
                values >= 0.5
            ).sum()

            score_sums[label] += values.sum()

        print(
            f"Validated {total_rows:,} records...",
            end="\r"
        )

    print("\n\n" + "=" * 65)
    print("DATASET INTEGRITY")
    print("=" * 65)

    print("Unique records:", f"{total_rows:,}")
    print("Original occurrences:", f"{total_occurrences:,}")
    print("Rows containing repeated comments:", f"{repeated_texts:,}")
    print("Missing comment text:", missing_text)
    print("Missing label values:", missing_labels)
    print("Invalid annotation scores:", invalid_scores)

    print("\n" + "=" * 65)
    print("UPDATED LABEL DISTRIBUTION")
    print("=" * 65)

    results = []

    for label in LABELS:

        positives = positive_counts[label]

        percentage = (
            positives / total_rows * 100
            if total_rows else 0
        )

        mean_score = (
            score_sums[label] / total_rows
            if total_rows else 0
        )

        results.append({
            "label": label,
            "positive_count": positives,
            "positive_percentage": round(percentage, 4),
            "mean_annotation_score": round(mean_score, 6)
        })

    summary = pd.DataFrame(results)

    print(summary.to_string(index=False))

    print("\n" + "=" * 65)

    if (
        total_rows == 1_968_755
        and total_occurrences == 1_981_413
        and missing_text == 0
        and missing_labels == 0
        and invalid_scores == 0
    ):

        print("DATASET INTEGRITY CHECK: PASSED")

    else:

        print("DATASET INTEGRITY CHECK: REVIEW REQUIRED")

    print("=" * 65)


if __name__ == "__main__":
    main()
