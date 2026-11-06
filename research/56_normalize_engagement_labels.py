
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv",

    ROOT / "datasets" / "engagement" / "raw"
    / "engagement_comments.csv",
]

VALID_LABELS = {
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
    "UNCLEAR",
}


def normalize_file(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "intent" not in df.columns:
        raise ValueError(f"Missing intent column: {path.name}")

    original = df["intent"].astype("string").fillna("")
    normalized = original.str.strip().str.upper()

    invalid = sorted(set(normalized) - VALID_LABELS)

    if invalid:
        raise ValueError(
            f"Invalid labels in {path.name}: {invalid}"
        )

    changed = (original != normalized).sum()

    # Backup before modifying.
    backup = path.with_name(
        path.stem + "_before_label_normalization.csv"
    )

    if not backup.exists():
        df.to_csv(backup, index=False, encoding="utf-8-sig")

    df["intent"] = normalized

    df.to_csv(path, index=False, encoding="utf-8-sig")

    # Verify saved values.
    verified = pd.read_csv(path)

    saved_labels = (
        verified["intent"].astype("string").fillna("")
    )

    if not saved_labels.equals(normalized.reset_index(drop=True)):
        raise ValueError(
            f"Verification failed for {path.name}"
        )

    print(f"\nFile: {path.name}")
    print("Labels normalized:", changed)
    print("Intent distribution:")
    print(verified["intent"].value_counts().to_string())
    print("Verification: PASSED")


def main():
    for path in FILES:
        normalize_file(path)

    print("\nALL ENGAGEMENT LABELS NORMALIZED SUCCESSFULLY")


if __name__ == "__main__":
    main()
