
import pandas as pd
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets" / "raw" / "civil_comments"
)

REPORT_DIR = (
    Path(__file__).resolve().parent / "reports"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

THRESHOLD = 0.5

for split in ["train", "validation", "test"]:

    print("\n" + "=" * 65)
    print(f"ANALYZING {split.upper()} DATASET")
    print("=" * 65)

    total = 0
    positive_counts = {label: 0 for label in LABELS}
    score_sums = {label: 0.0 for label in LABELS}
    sample_comments = []

    for chunk in pd.read_csv(
        DATA_DIR / f"{split}.csv",
        chunksize=100_000
    ):

        total += len(chunk)

        for label in LABELS:
            positive_counts[label] += (
                chunk[label] >= THRESHOLD
            ).sum()

            score_sums[label] += chunk[label].sum()

        if len(sample_comments) < 5:
            remaining = 5 - len(sample_comments)

            sample_comments.extend(
                chunk[["text", "toxicity"]]
                .head(remaining)
                .to_dict("records")
            )

    print(f"\nTotal comments: {total:,}")

    print("\nLabel distribution:")
    for label in LABELS:
        count = positive_counts[label]
        percentage = count / total * 100
        average_score = score_sums[label] / total

        print(
            f"{label:20} "
            f"Positive: {count:10,} "
            f"({percentage:6.2f}%) "
            f"Average score: {average_score:.4f}"
        )

    print("\nSample comments:")
    for item in sample_comments:
        print("\nComment:", item["text"])
        print("Toxicity score:", item["toxicity"])

print("\nAnalysis completed!")
