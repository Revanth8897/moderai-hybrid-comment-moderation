
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

REAL_FILE = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv"
)

OLD_COMBINED_FILE = (
    ROOT / "datasets" / "engagement" / "processed"
    / "engagement_all_330.csv"
)

OUTPUT_FILE = (
    ROOT / "datasets" / "engagement" / "processed"
    / "engagement_all_430.csv"
)


def normalize_text(value):
    return " ".join(str(value).casefold().split())


def main():
    print("=" * 85)
    print("MODERAI — BUILD UPDATED ENGAGEMENT DATASET")
    print("=" * 85)

    # Check source files
    for path in [REAL_FILE, OLD_COMBINED_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists. Not overwriting: {OUTPUT_FILE}"
        )

    # Load latest real-world annotations
    real = pd.read_csv(
        REAL_FILE,
        dtype="string",
        keep_default_na=False
    )

    # Load old combined dataset to recover synthetic examples
    old = pd.read_csv(
        OLD_COMBINED_FILE,
        dtype="string",
        keep_default_na=False
    )

    required_real = {"comment_id", "comment_text", "intent", "video_title"}
    required_old = {"comment_id", "text", "intent", "dataset_type"}

    if not required_real.issubset(real.columns):
        raise ValueError(
            f"Real-world file missing columns: "
            f"{sorted(required_real - set(real.columns))}"
        )

    if not required_old.issubset(old.columns):
        raise ValueError(
            f"Old combined file missing columns: "
            f"{sorted(required_old - set(old.columns))}"
        )

    # Separate synthetic examples from old combined dataset
    synthetic = old[
        old["dataset_type"].eq("synthetic")
    ].copy()

    # Standardize the comment column
    synthetic["comment_text"] = synthetic["text"]

    # Mark latest real-world annotations
    real = real.copy()
    real["dataset_type"] = "real_world"

    # Remove old text column from synthetic data
    synthetic = synthetic.drop(columns=["text"])

    # Combine while preserving all available metadata columns
    combined = pd.concat(
        [real, synthetic],
        ignore_index=True,
        sort=False
    ).fillna("")

    # Standardize text values
    combined["comment_text"] = (
        combined["comment_text"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    # Validate expected dataset size
    print("\n1. DATASET COUNTS")
    print("-" * 85)

    print("Latest real-world rows:", len(real))
    print("Preserved synthetic rows:", len(synthetic))
    print("Combined rows:", len(combined))

    if len(real) != 400:
        raise ValueError(f"Expected 400 real-world rows, found {len(real)}")

    if len(synthetic) != 30:
        raise ValueError(
            f"Expected 30 synthetic rows, found {len(synthetic)}"
        )

    if len(combined) != 430:
        raise ValueError(
            f"Expected 430 combined rows, found {len(combined)}"
        )

    # Validate IDs and comment text
    if combined["comment_id"].duplicated().any():
        duplicates = combined.loc[
            combined["comment_id"].duplicated(keep=False),
            "comment_id"
        ].tolist()
        raise ValueError(f"Duplicate comment IDs found: {duplicates[:10]}")

    if combined["comment_text"].eq("").any():
        raise ValueError("Blank comment text found.")

    normalized = combined["comment_text"].map(normalize_text)

    if normalized.duplicated().any():
        raise ValueError("Duplicate normalized comment text found.")

    # Validate labels
    allowed_intents = {
        "POSITIVE",
        "NEGATIVE",
        "CRITICAL",
        "QUESTION",
        "NEUTRAL",
        "UNCLEAR",
    }

    invalid = set(combined["intent"].unique()) - allowed_intents

    if invalid:
        raise ValueError(f"Unexpected intent labels: {sorted(invalid)}")

    # Report distributions
    print("\n2. DATASET TYPE DISTRIBUTION")
    print("-" * 85)
    print(combined["dataset_type"].value_counts().to_string())

    print("\n3. INTENT DISTRIBUTION")
    print("-" * 85)
    print(
        combined.groupby(
            ["dataset_type", "intent"]
        ).size().unstack(fill_value=0).to_string()
    )

    print("\n4. VIDEO METADATA")
    print("-" * 85)

    real_combined = combined[
        combined["dataset_type"].eq("real_world")
    ]

    print("Real-world comments:", len(real_combined))
    print(
        "Real-world comments with video titles:",
        real_combined["video_title"].astype("string").str.strip().ne("").sum()
    )
    print(
        "Unique real-world video titles:",
        real_combined["video_title"].nunique()
    )

    # Save new dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n5. FINAL VERIFICATION")
    print("-" * 85)
    print("[PASS] 400 real-world comments")
    print("[PASS] 30 synthetic comments")
    print("[PASS] 430 total comments")
    print("[PASS] Unique comment IDs")
    print("[PASS] Unique normalized comment text")
    print("[PASS] No blank comments")
    print("[PASS] All intent labels valid")
    print("[PASS] Standardized column: comment_text")
    print("[PASS] Original files were not modified")

    print("\nNEW DATASET:")
    print(OUTPUT_FILE)

    print("\nDataset creation completed successfully.")


if __name__ == "__main__":
    main()
