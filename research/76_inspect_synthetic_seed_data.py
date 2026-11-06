
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

COMBINED_FILE = (
    ROOT
    / "datasets"
    / "engagement"
    / "processed"
    / "engagement_all_330.csv"
)

SPLIT_DIR = (
    ROOT
    / "datasets"
    / "engagement"
    / "splits"
)


def main():
    print("=" * 85)
    print("MODERAI — SYNTHETIC SEED DATA INSPECTION")
    print("=" * 85)

    if not COMBINED_FILE.exists():
        raise FileNotFoundError(
            f"Combined dataset not found: {COMBINED_FILE}"
        )

    df = pd.read_csv(
        COMBINED_FILE,
        dtype="string",
        keep_default_na=False
    )

    required = {
        "dataset_type",
        "intent",
        "comment_id",
        "text",
        "source",
        "annotator",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print("\n1. COMBINED DATASET")
    print("-" * 85)

    print("File:", COMBINED_FILE)
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())

    print("\nDataset type distribution:")
    print(df["dataset_type"].value_counts().to_string())

    synthetic = df[
        df["dataset_type"].eq("synthetic")
    ].copy()

    real = df[
        df["dataset_type"].eq("real_world")
    ].copy()

    print("\n2. SYNTHETIC DATASET")
    print("-" * 85)

    print("Synthetic rows:", len(synthetic))

    print("\nSynthetic intent distribution:")
    print(synthetic["intent"].value_counts().to_string())

    print("\nSynthetic comment examples:")
    print(
        synthetic[
            ["comment_id", "text", "intent", "source", "annotator"]
        ].head(10).to_string(index=False)
    )

    print("\n3. REAL-WORLD DATASET")
    print("-" * 85)

    print("Real-world rows:", len(real))
    print("Real-world intent distribution:")
    print(real["intent"].value_counts().to_string())

    print("\n4. EXISTING SPLIT FILES")
    print("-" * 85)

    for filename in [
        "train.csv",
        "validation.csv",
        "test.csv",
    ]:
        path = SPLIT_DIR / filename
        print(f"{filename}: {'EXISTS' if path.exists() else 'NOT FOUND'}")

    print("\n5. VERIFICATION")
    print("-" * 85)

    print("Synthetic rows expected: 30")
    print("Synthetic rows found:", len(synthetic))

    print("Real-world rows expected in old dataset: 300")
    print("Real-world rows found:", len(real))

    if len(synthetic) == 30:
        print("[PASS] Synthetic seed data count")
    else:
        print("[CHECK] Synthetic seed data count")

    if synthetic["comment_id"].is_unique:
        print("[PASS] Synthetic IDs are unique")
    else:
        print("[CHECK] Duplicate synthetic IDs")

    print("\nInspection completed. No files were modified.")


if __name__ == "__main__":
    main()
