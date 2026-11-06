
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT / "datasets" / "processed" / "civil_comments"
)

TFIDF_DIR = PROJECT_ROOT / "datasets" / "processed" / "tfidf"

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

SPLITS = ["train", "validation", "test"]


def count_full_dataset(split):
    file_path = PROCESSED_DIR / f"{split}.csv"

    counts = {label: 0 for label in LABELS}
    total_rows = 0

    for chunk in pd.read_csv(file_path, chunksize=100000):
        total_rows += len(chunk)

        for label in LABELS:
            counts[label] += int((chunk[label] >= 0.5).sum())

    return total_rows, counts


def count_tfidf_sample(split):
    file_path = TFIDF_DIR / f"y_{split}.csv"
    df = pd.read_csv(file_path)

    counts = {
        label: int((df[label] >= 0.5).sum())
        for label in LABELS
    }

    return len(df), counts


print("\nFULL PROCESSED DATASET DISTRIBUTION")
print("=" * 80)

for split in SPLITS:
    total, counts = count_full_dataset(split)

    print(f"\n{split.upper()} — {total:,} comments")

    for label in LABELS:
        percentage = counts[label] / total * 100
        print(
            f"{label:20} "
            f"{counts[label]:10,} "
            f"({percentage:.4f}%)"
        )


print("\n\nTF-IDF SAMPLE DISTRIBUTION")
print("=" * 80)

for split in SPLITS:
    total, counts = count_tfidf_sample(split)

    print(f"\n{split.upper()} — {total:,} comments")

    for label in LABELS:
        percentage = counts[label] / total * 100
        print(
            f"{label:20} "
            f"{counts[label]:10,} "
            f"({percentage:.4f}%)"
        )

print("\nDistribution audit completed!")
