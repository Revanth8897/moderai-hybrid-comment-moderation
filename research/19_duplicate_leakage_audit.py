
import hashlib
import pandas as pd

from pathlib import Path

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "processed" / "civil_comments"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "19_duplicate_audit_summary.csv"

CHUNK_SIZE = 100_000


# --------------------------------------------------
# HASH FUNCTION
# --------------------------------------------------

def get_text_hash(text):

    """
    Generate a stable 128-bit hash for comment text.

    Identical text produces the same hash.
    """

    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16
    ).digest()


# --------------------------------------------------
# DATASET SCANNER
# --------------------------------------------------

def scan_dataset(file_path, previous_sets=None):

    if previous_sets is None:
        previous_sets = {}

    print(f"\nScanning: {file_path.name}")

    seen_current = set()

    internal_duplicates = 0
    total_rows = 0
    empty_rows = 0

    cross_split_matches = {
        name: 0 for name in previous_sets
    }

    reader = pd.read_csv(
        file_path,
        usecols=["text"],
        dtype={"text": "string"},
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        for text in chunk["text"]:

            total_rows += 1

            if pd.isna(text) or text == "":
                empty_rows += 1
                continue

            text_hash = get_text_hash(str(text))

            # Check duplicates within current dataset.
            if text_hash in seen_current:

                internal_duplicates += 1

            else:

                seen_current.add(text_hash)

            # Check overlap with previously scanned splits.
            for split_name, previous_hashes in previous_sets.items():

                if text_hash in previous_hashes:
                    cross_split_matches[split_name] += 1

        print(
            f"Processed {total_rows:,} rows...",
            end="\r"
        )

    print(f"\nCompleted: {file_path.name}")

    print("Total rows:", f"{total_rows:,}")
    print("Unique non-empty comments:", f"{len(seen_current):,}")
    print("Internal duplicate rows:", f"{internal_duplicates:,}")
    print("Empty rows:", f"{empty_rows:,}")

    for split_name, count in cross_split_matches.items():

        print(
            f"Duplicates matching {split_name}: {count:,}"
        )

    return {
        "hashes": seen_current,
        "total_rows": total_rows,
        "unique_comments": len(seen_current),
        "internal_duplicates": internal_duplicates,
        "empty_rows": empty_rows,
        "cross_split_matches": cross_split_matches
    }


# --------------------------------------------------
# MAIN AUDIT
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MODERAI DATASET DUPLICATE LEAKAGE AUDIT")
    print("=" * 65)

    # 1. Scan training dataset.
    train = scan_dataset(TRAIN_FILE)

    # 2. Scan validation against training.
    validation = scan_dataset(
        VALIDATION_FILE,
        previous_sets={
            "train": train["hashes"]
        }
    )

    # 3. Scan test against training and validation.
    test = scan_dataset(
        TEST_FILE,
        previous_sets={
            "train": train["hashes"],
            "validation": validation["hashes"]
        }
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results = [

        {
            "split": "train",
            "total_rows": train["total_rows"],
            "unique_comments": train["unique_comments"],
            "internal_duplicate_rows": train["internal_duplicates"],
            "empty_rows": train["empty_rows"],
            "overlap_with_train": 0,
            "overlap_with_validation": 0,
            "overlap_with_test": 0
        },

        {
            "split": "validation",
            "total_rows": validation["total_rows"],
            "unique_comments": validation["unique_comments"],
            "internal_duplicate_rows": validation["internal_duplicates"],
            "empty_rows": validation["empty_rows"],
            "overlap_with_train": validation["cross_split_matches"]["train"],
            "overlap_with_validation": 0,
            "overlap_with_test": 0
        },

        {
            "split": "test",
            "total_rows": test["total_rows"],
            "unique_comments": test["unique_comments"],
            "internal_duplicate_rows": test["internal_duplicates"],
            "empty_rows": test["empty_rows"],
            "overlap_with_train": test["cross_split_matches"]["train"],
            "overlap_with_validation": test["cross_split_matches"]["validation"],
            "overlap_with_test": 0
        }
    ]

    summary = pd.DataFrame(results)

    print("\n")
    print("=" * 65)
    print("FINAL DUPLICATE AUDIT SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nAudit completed.")
    print("Report saved to:", OUTPUT_FILE)

    print("\nIMPORTANT:")
    print("No dataset files were modified or deleted.")


if __name__ == "__main__":
    main()
