
import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "civil_comments"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "civil_comments"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

COLUMNS = ["text"] + LABELS


def normalize_text(text):
    """Normalize whitespace without destroying linguistic signals."""

    if not isinstance(text, str):
        return None

    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    return text


for split in ["train", "validation", "test"]:

    print(f"\nProcessing {split} dataset...")

    input_file = RAW_DIR / f"{split}.csv"
    output_file = PROCESSED_DIR / f"{split}.csv"

    total = 0
    removed = 0
    seen = set()
    written = 0

    for chunk in pd.read_csv(input_file, chunksize=100_000):

        total += len(chunk)

        chunk = chunk[COLUMNS].copy()

        # Remove rows with missing labels or comments
        chunk = chunk.dropna(subset=COLUMNS)

        # Normalize text
        chunk["text"] = chunk["text"].apply(normalize_text)

        # Remove empty comments
        chunk = chunk.dropna(subset=["text"])

        # Validate annotation scores
        valid_labels = chunk[LABELS].apply(
            lambda col: col.between(0, 1)
        ).all(axis=1)

        chunk = chunk[valid_labels]

        # Remove exact duplicate comments within this split
        chunk = chunk.drop_duplicates(subset=["text"])

        # Write processed records
        chunk.to_csv(
            output_file,
            mode="a" if written else "w",
            header=(written == 0),
            index=False
        )

        written += len(chunk)

    removed = total - written

    print(f"Original records: {total:,}")
    print(f"Processed records: {written:,}")
    print(f"Records removed: {removed:,}")
    print(f"Saved to: {output_file}")

print("\nPreprocessing completed!")
