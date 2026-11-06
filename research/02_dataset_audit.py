
import pandas as pd
from pathlib import Path

# Dataset directory
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets" / "raw" / "civil_comments"
)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

for split in ["train", "validation", "test"]:

    print("\n" + "=" * 60)
    print(f"DATASET AUDIT: {split.upper()}")
    print("=" * 60)

    df = pd.read_csv(DATA_DIR / f"{split}.csv")

    print(f"\nTotal comments: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate comments:")
    print(df["text"].duplicated().sum())

    print("\nLabel statistics:")
    for label in LABELS:
        positive = (df[label] >= 0.5).sum()
        percentage = positive / len(df) * 100

        print(
            f"{label:20} "
            f"{positive:10,} "
            f"({percentage:.2f}%)"
        )

    print("\nSample comments:")
    print(df[["text", "toxicity"]].head(5).to_string(index=False))

print("\nDataset audit completed!")
