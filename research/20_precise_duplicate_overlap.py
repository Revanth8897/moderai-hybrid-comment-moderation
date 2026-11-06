
import hashlib
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "processed" / "civil_comments"

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv"
}

CHUNK_SIZE = 100_000


def hash_comment(text):
    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16
    ).digest()


def load_hashes(file_path):

    hashes = set()

    reader = pd.read_csv(
        file_path,
        usecols=["text"],
        dtype={"text": "string"},
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        for text in chunk["text"]:

            if pd.isna(text) or text == "":
                continue

            hashes.add(hash_comment(str(text)))

    return hashes


def main():

    print("Loading training comment hashes...")
    train = load_hashes(FILES["train"])

    print("Loading validation comment hashes...")
    validation = load_hashes(FILES["validation"])

    print("Loading test comment hashes...")
    test = load_hashes(FILES["test"])

    # Validation overlap
    validation_train_overlap = validation & train

    # Test overlap categories
    test_train_overlap = test & train
    test_validation_overlap = test & validation

    test_both_overlap = test & train & validation

    test_train_only = test_train_overlap - validation
    test_validation_only = test_validation_overlap - train

    test_any_overlap = test & (train | validation)

    # Results
    print("\n" + "=" * 65)
    print("PRECISE UNIQUE DUPLICATE OVERLAP")
    print("=" * 65)

    print("\nUnique comments:")
    print("Train:", f"{len(train):,}")
    print("Validation:", f"{len(validation):,}")
    print("Test:", f"{len(test):,}")

    print("\nValidation:")
    print(
        "Unique comments also in train:",
        f"{len(validation_train_overlap):,}"
    )

    print("\nTest:")
    print(
        "Unique comments also in train:",
        f"{len(test_train_overlap):,}"
    )

    print(
        "Unique comments also in validation:",
        f"{len(test_validation_overlap):,}"
    )

    print(
        "Unique comments also in BOTH:",
        f"{len(test_both_overlap):,}"
    )

    print(
        "Unique comments in train ONLY:",
        f"{len(test_train_only):,}"
    )

    print(
        "Unique comments in validation ONLY:",
        f"{len(test_validation_only):,}"
    )

    print(
        "Total unique test comments overlapping either split:",
        f"{len(test_any_overlap):,}"
    )

    print(
        "Test comments with no overlap:",
        f"{len(test - (train | validation)):,}"
    )

    print("\nAudit completed. No dataset files modified.")


if __name__ == "__main__":
    main()
