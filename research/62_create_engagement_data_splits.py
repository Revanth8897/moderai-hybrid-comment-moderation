
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "datasets" / "engagement" / "processed"
    / "engagement_all_330.csv"
)

OUTPUT_DIR = (
    ROOT / "datasets" / "engagement" / "splits"
)

RANDOM_STATE = 42

INTENTS = [
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
    "UNCLEAR",
]


def split_by_video(data, test_size, random_state):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_indices, test_indices = next(
        splitter.split(
            data,
            groups=data["video_title"],
        )
    )

    return (
        data.iloc[train_indices].copy(),
        data.iloc[test_indices].copy(),
    )


def validate_splits(train, validation, test):
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Check unique comment IDs.
    all_ids = []

    for name, df in splits.items():
        if df["comment_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs inside {name}")

        all_ids.extend(df["comment_id"].tolist())

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Comment IDs overlap between splits")

    # Synthetic examples must be training-only.
    if (train["dataset_type"] != "real_world").sum() != 30:
        raise ValueError("Expected exactly 30 synthetic training rows")

    for name in ["validation", "test"]:
        if not (splits[name]["dataset_type"] == "real_world").all():
            raise ValueError(
                f"Synthetic data found in {name}"
            )

    # Video titles must not overlap.
    title_sets = {}

    for name, df in splits.items():
        real = df[df["dataset_type"] == "real_world"]
        title_sets[name] = set(real["video_title"])

    if title_sets["train"] & title_sets["validation"]:
        raise ValueError("Video-title leakage: train/validation")

    if title_sets["train"] & title_sets["test"]:
        raise ValueError("Video-title leakage: train/test")

    if title_sets["validation"] & title_sets["test"]:
        raise ValueError("Video-title leakage: validation/test")

    # Every original row must be assigned exactly once.
    if sum(len(df) for df in splits.values()) != 330:
        raise ValueError("Total split row count is incorrect")

    return title_sets


def print_report(name, df):
    print(f"\n{name.upper()}")
    print("-" * 55)

    print("Total comments:", len(df))
    print(
        "Real-world:",
        (df["dataset_type"] == "real_world").sum(),
    )
    print(
        "Synthetic:",
        (df["dataset_type"] == "synthetic").sum(),
    )

    print("\nIntent distribution:")

    counts = (
        df["intent"]
        .value_counts()
        .reindex(INTENTS, fill_value=0)
    )

    print(counts.to_string())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_files = [
        OUTPUT_DIR / "train.csv",
        OUTPUT_DIR / "validation.csv",
        OUTPUT_DIR / "test.csv",
    ]

    existing = [path for path in output_files if path.exists()]

    if existing:
        raise FileExistsError(
            "Split files already exist. Review them before replacing:\n"
            + "\n".join(str(path) for path in existing)
        )

    df = pd.read_csv(INPUT_PATH)

    real = df[df["dataset_type"] == "real_world"].copy()
    synthetic = df[df["dataset_type"] == "synthetic"].copy()

    real["video_title"] = (
        real["video_title"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    if real["video_title"].eq("").any():
        raise ValueError("Real-world rows contain missing video titles")

    if len(real) != 300 or len(synthetic) != 30:
        raise ValueError("Unexpected source dataset sizes")

    # First: reserve approximately 15% for testing.
    train_validation_real, test_real = split_by_video(
        real,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    # Second: reserve approximately 15% of the full real dataset
    # for validation. 0.17647 of the remaining 85% is about 15%.
    train_real, validation_real = split_by_video(
        train_validation_real,
        test_size=0.17647,
        random_state=RANDOM_STATE,
    )

    # Synthetic examples are training-only.
    train = pd.concat(
        [train_real, synthetic],
        ignore_index=True,
    )

    validation = validation_real.reset_index(drop=True)
    test = test_real.reset_index(drop=True)
    train = train.reset_index(drop=True)

    title_sets = validate_splits(
        train,
        validation,
        test,
    )

    # Save split files.
    train.to_csv(output_files[0], index=False, encoding="utf-8-sig")
    validation.to_csv(
        output_files[1], index=False, encoding="utf-8-sig"
    )
    test.to_csv(output_files[2], index=False, encoding="utf-8-sig")

    print("\nMODERAI — ENGAGEMENT DATA SPLITS")
    print("=" * 60)

    print_report("Training", train)
    print_report("Validation", validation)
    print_report("Testing", test)

    print("\nVIDEO GROUP AUDIT")
    print("-" * 55)

    for name, titles in title_sets.items():
        print(f"{name}: {len(titles)} unique video titles")

    print("\nOUTPUT FILES")
    for path in output_files:
        print(path)

    print("\nVERIFICATION: PASSED")
    print("No comment ID overlap.")
    print("No video-title leakage.")
    print("Synthetic examples are training-only.")
    print("All 330 comments assigned exactly once.")


if __name__ == "__main__":
    main()
