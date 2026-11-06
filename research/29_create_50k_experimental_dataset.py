
import numpy as np
import pandas as pd

from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = (
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
    / "moderai_50k"
)

SAMPLE_SIZE = 50_000
TRAIN_SIZE = 40_000
VALIDATION_SIZE = 5_000
TEST_SIZE = 5_000

RANDOM_STATE = 42
CHUNK_SIZE = 100_000

# Severe toxicity is excluded from stratification because
# the complete dataset contains only 13 positive comments.
STRATIFICATION_LABELS = [
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

ALL_LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]


def load_labels():

    print("\nLoading six stratification labels...")

    chunks = []

    reader = pd.read_csv(
        SOURCE_FILE,
        usecols=STRATIFICATION_LABELS,
        chunksize=CHUNK_SIZE
    )

    total = 0

    for chunk in reader:

        y = (
            chunk[STRATIFICATION_LABELS]
            .to_numpy() >= 0.5
        ).astype(np.int8)

        chunks.append(y)

        total += len(chunk)

        print(
            f"Loaded {total:,} records...",
            end="\r"
        )

    print(f"\nTotal records loaded: {total:,}")

    return np.concatenate(chunks, axis=0)


def select_50k_sample(y):

    print("\nSelecting representative 50K sample...")

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    _, selected_idx = next(
        splitter.split(
            np.zeros((len(y), 1), dtype=np.uint8),
            y
        )
    )

    print("Selected records:", f"{len(selected_idx):,}")

    return selected_idx


def create_final_splits(y, selected_idx):

    print("\nCreating 80/10/10 splits...")

    selected_labels = y[selected_idx]

    # First split: 80% training and 20% temporary.
    first_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=10_000,
        random_state=RANDOM_STATE + 1
    )

    train_relative, temp_relative = next(
        first_split.split(
            np.zeros((SAMPLE_SIZE, 1), dtype=np.uint8),
            selected_labels
        )
    )

    # Second split: divide temporary into 5K validation and 5K test.
    second_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=5_000,
        random_state=RANDOM_STATE + 2
    )

    validation_relative, test_relative = next(
        second_split.split(
            np.zeros((10_000, 1), dtype=np.uint8),
            selected_labels[temp_relative]
        )
    )

    train_idx = selected_idx[train_relative]
    validation_idx = selected_idx[
        temp_relative[validation_relative]
    ]
    test_idx = selected_idx[
        temp_relative[test_relative]
    ]

    print("\nFinal split sizes:")
    print("Training:", f"{len(train_idx):,}")
    print("Validation:", f"{len(validation_idx):,}")
    print("Test:", f"{len(test_idx):,}")

    return train_idx, validation_idx, test_idx


def print_distribution(y, splits):

    print("\n" + "=" * 70)
    print("EXPERIMENTAL DATASET LABEL DISTRIBUTION")
    print("=" * 70)

    for name, indices in splits.items():

        print(f"\n{name.upper()} ({len(indices):,})")

        for i, label in enumerate(STRATIFICATION_LABELS):

            positives = int(y[indices, i].sum())

            percentage = (
                positives / len(indices) * 100
            )

            print(
                f"{label:20} "
                f"{positives:7,} "
                f"({percentage:.4f}%)"
            )


def export_dataset(splits):

    print("\nExporting experimental dataset...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_files = {
        "train": OUTPUT_DIR / "train.csv",
        "validation": OUTPUT_DIR / "validation.csv",
        "test": OUTPUT_DIR / "test.csv"
    }

    for path in output_files.values():

        if path.exists():

            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Move or rename existing files before rerunning."
            )

    # -1 means the record is not selected.
    # 0 = train, 1 = validation, 2 = test.
    assignment = np.full(
        SOURCE_ROW_COUNT,
        -1,
        dtype=np.int8
    )

    assignment[splits["train"]] = 0
    assignment[splits["validation"]] = 1
    assignment[splits["test"]] = 2

    split_ids = {
        0: "train",
        1: "validation",
        2: "test"
    }

    first_write = {
        "train": True,
        "validation": True,
        "test": True
    }

    reader = pd.read_csv(
        SOURCE_FILE,
        chunksize=CHUNK_SIZE
    )

    offset = 0

    for chunk in reader:

        end = offset + len(chunk)

        chunk_assignment = assignment[offset:end]

        for split_id, split_name in split_ids.items():

            mask = chunk_assignment == split_id

            if mask.any():

                chunk.loc[mask].to_csv(
                    output_files[split_name],
                    index=False,
                    mode="w" if first_write[split_name] else "a",
                    header=first_write[split_name]
                )

                first_write[split_name] = False

        offset = end

        print(
            f"Scanned {offset:,} source records...",
            end="\r"
        )

    if offset != SOURCE_ROW_COUNT:
        raise ValueError("Source row count changed during export.")

    print("\n\nExport completed.")

    for name, path in output_files.items():

        print(
            f"{name.capitalize()}: {path}"
        )


def main():

    global SOURCE_ROW_COUNT

    print("=" * 70)
    print("MODERAI 50K EXPERIMENTAL DATASET")
    print("=" * 70)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    # Read source row count without loading all comments.
    SOURCE_ROW_COUNT = sum(
        1 for _ in open(SOURCE_FILE, encoding="utf-8")
    ) - 1

    print("Source records:", f"{SOURCE_ROW_COUNT:,}")
    print("Target sample:", f"{SAMPLE_SIZE:,}")

    if SOURCE_ROW_COUNT < SAMPLE_SIZE:
        raise ValueError("Source dataset is smaller than sample size.")

    if any(
        (OUTPUT_DIR / f"{name}.csv").exists()
        for name in ["train", "validation", "test"]
    ):
        raise FileExistsError(
            "Experimental split files already exist. "
            "Please move them before running again."
        )

    y = load_labels()

    if len(y) != SOURCE_ROW_COUNT:
        raise ValueError("Label count does not match source row count.")

    selected_idx = select_50k_sample(y)

    train_idx, validation_idx, test_idx = create_final_splits(
        y,
        selected_idx
    )

    splits = {
        "train": train_idx,
        "validation": validation_idx,
        "test": test_idx
    }

    print_distribution(y, splits)

    export_dataset(splits)

    print("\n" + "=" * 70)
    print("MODERAI 50K DATASET CREATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
