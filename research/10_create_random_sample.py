
import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "civil_comments"
    / "train.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "random_sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "train_sample.csv"

TOTAL_ROWS = 1_787_805
SAMPLE_SIZE = 200_000
CHUNK_SIZE = 100_000
RANDOM_SEED = 42

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

# Select random row positions across the entire dataset.
rng = np.random.default_rng(RANDOM_SEED)

sample_indices = np.sort(
    rng.choice(
        TOTAL_ROWS,
        size=SAMPLE_SIZE,
        replace=False,
    )
)

print("Selected random row positions.")
print("Expected sample size:", SAMPLE_SIZE)

# Read the CSV in chunks to avoid loading the entire dataset.
selected_count = 0
chunk_start = 0
first_chunk = True

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=COLUMNS,
    chunksize=CHUNK_SIZE,
):

    chunk_end = chunk_start + len(chunk)

    # Find selected positions belonging to this chunk.
    left = np.searchsorted(sample_indices, chunk_start)
    right = np.searchsorted(sample_indices, chunk_end)

    local_indices = (
        sample_indices[left:right] - chunk_start
    )

    if len(local_indices) > 0:
        selected_rows = chunk.iloc[local_indices]

        selected_rows.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        selected_count += len(selected_rows)
        first_chunk = False

    chunk_start = chunk_end

    print(
        f"Scanned {chunk_end:,} / {TOTAL_ROWS:,} rows | "
        f"Selected {selected_count:,}"
    )

if chunk_start != TOTAL_ROWS:
    raise ValueError(
        f"Expected {TOTAL_ROWS} rows, found {chunk_start}"
    )

if selected_count != SAMPLE_SIZE:
    raise ValueError(
        f"Expected {SAMPLE_SIZE} samples, found {selected_count}"
    )

print("\nRandom sampling completed!")
print("Total selected:", selected_count)
print("Saved to:", OUTPUT_FILE)

# Verify label distribution in the sample.
sample_df = pd.read_csv(OUTPUT_FILE)

print("\nRANDOM SAMPLE LABEL DISTRIBUTION")
print("=" * 60)

for label in LABELS:
    positive = int((sample_df[label] >= 0.5).sum())
    percentage = positive / len(sample_df) * 100

    print(
        f"{label:20} {positive:8,} "
        f"({percentage:.4f}%)"
    )
