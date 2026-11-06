import numpy as np
import pandas as pd

from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT
    / "datasets"
    / "processed"
    / "civil_comments_clean"
    / "all_unique_comments.csv"
)

OUTPUT_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "civil_comments_clean"
)

CHUNK_SIZE = 100_000
RANDOM_STATE = 42

# Severe toxicity is excluded because it has only 13 positive records.
LABELS = [
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]


def load_binary_labels():

    print("\nLoading labels for stratification...")

    label_chunks = []

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=LABELS,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        binary_labels = (
            chunk[LABELS].to_numpy() >= 0.5
        ).astype(np.int8)

        label_chunks.append(binary_labels)

        print(
            f"Loaded {sum(len(x) for x in label_chunks):,} records...",
            end="\r"
        )

    return np.concatenate(label_chunks, axis=0)


def create_splits(y):

    total_rows = len(y)

    print("\n\nCreating 90/5/5 multilabel splits...")

    # First: 90% training and 10% temporary data.
    first_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.10,
        random_state=RANDOM_STATE
    )

    train_idx, temp_idx = next(
        first_split.split(
            np.zeros((total_rows, 1)),
            y
        )
    )

    # Second: divide temporary data equally into validation and test.
    second_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE + 1
    )

    val_relative_idx, test_relative_idx = next(
        second_split.split(
            np.zeros((len(temp_idx), 1)),
            y[temp_idx]
        )
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    # Verify that every record belongs to exactly one split.
    seen = np.zeros(total_rows, dtype=bool)

    for name, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx)
    ]:

        if seen[indices].any():
            raise ValueError(
                f"Overlapping records found in {name} split."
            )

        seen[indices] = True

    if not seen.all():
        raise ValueError(
            "Some records were not assigned to any split."
        )

    print("\nSplit sizes:")

    print("Training:", f"{len(train_idx):,}")
    print("Validation:", f"{len(val_idx):,}")
    print("Test:", f"{len(test_idx):,}")

    print("Total:", f"{len(train_idx) + len(val_idx) + len(test_idx):,}")

    return train_idx, val_idx, test_idx


def export_splits(train_idx, val_idx, test_idx, total_rows):

    print("\nExporting split CSV files...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 0 = train, 1 = validation, 2 = test
    assignment = np.full(
        total_rows,
        -1,
        dtype=np.int8
    )

    assignment[train_idx] = 0
    assignment[val_idx] = 1
    assignment[test_idx] = 2

    split_files = {
        0: OUTPUT_DIR / "train.csv",
        1: OUTPUT_DIR / "validation.csv",
        2: OUTPUT_DIR / "test.csv"
    }

    # Remove previous output files before writing.
    for file_path in split_files.values():

        if file_path.exists():
            file_path.unlink()

    first_write = {
        0: True,
        1: True,
        2: True
    }

    offset = 0

    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        end = offset + len(chunk)

        chunk_assignments = assignment[offset:end]

        for split_id, file_path in split_files.items():

            mask = chunk_assignments == split_id

            if mask.any():

                chunk.loc[mask].to_csv(
                    file_path,
                    index=False,
                    mode="w" if first_write[split_id] else "a",
                    header=first_write[split_id]
                )

                first_write[split_id] = False

        offset = end

        print(
            f"Exported {offset:,} / {total_rows:,} records...",
            end="\r"
        )

    if offset != total_rows:
        raise ValueError("Exported row count does not match input.")

    print("\n\nFiles created:")

    for name, file_path in [
        ("Training", split_files[0]),
        ("Validation", split_files[1]),
        ("Test", split_files[2])
    ]:

        print(name, ":", file_path)


def print_label_distribution(y, train_idx, val_idx, test_idx):

    print("\n" + "=" * 75)
    print("LABEL DISTRIBUTION ACROSS SPLITS")
    print("=" * 75)

    splits = {
        "Train": train_idx,
        "Validation": val_idx,
        "Test": test_idx
    }

    for name, indices in splits.items():

        print(f"\n{name} ({len(indices):,} records)")

        for i, label in enumerate(LABELS):

            positives = int(y[indices, i].sum())

            percentage = (
                positives / len(indices) * 100
            )

            print(
                f"{label:20} "
                f"{positives:10,} "
                f"({percentage:.4f}%)"
            )


def main():

    print("=" * 75)
    print("MODERAI CLEAN DATASET SPLITTING")
    print("=" * 75)

    y = load_binary_labels()

    print("\nTotal records:", f"{len(y):,}")

    train_idx, val_idx, test_idx = create_splits(y)

    print_label_distribution(
        y,
        train_idx,
        val_idx,
        test_idx
    )

    export_splits(
        train_idx,
        val_idx,
        test_idx,
        len(y)
    )

    print("\n" + "=" * 75)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()