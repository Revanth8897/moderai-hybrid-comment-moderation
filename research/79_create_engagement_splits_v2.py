
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT
    / "datasets"
    / "engagement"
    / "processed"
    / "engagement_all_430.csv"
)

OUTPUT_DIR = (
    ROOT
    / "datasets"
    / "engagement"
    / "splits"
)

OUTPUT_FILES = [
    "train_v2.csv",
    "validation_v2.csv",
    "test_v2.csv",
    "train_unclear_excluded_v2.csv",
    "split_summary_v2.txt",
]

ALL_INTENTS = [
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
    "UNCLEAR",
]

TRAIN_INTENTS = [
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
]

N_SPLITS = 7
SEEDS = range(42, 142)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_group(value):
    """Normalize video titles for group-based splitting."""
    return " ".join(str(value).casefold().split())


def normalize_text(value):
    """Normalize comment text for duplicate detection."""
    return " ".join(str(value).casefold().split())


def fold_score(frame, target_rows, target_counts):
    """Measure how closely a fold matches the target distribution."""

    score = ((len(frame) - target_rows) / target_rows) ** 2

    actual = (
        frame["intent"]
        .value_counts()
        .reindex(ALL_INTENTS, fill_value=0)
    )

    for intent in ALL_INTENTS:
        target = target_counts[intent]

        score += (
            (actual[intent] - target) / max(target, 1)
        ) ** 2

        # Penalize missing labels in validation or test.
        if actual[intent] == 0:
            score += 25

    return score


# ============================================================
# 3. FIND BEST GROUP-AWARE SPLIT
# ============================================================

def find_best_group_split(real):

    y = real["intent"]
    groups = real["_video_group"]

    total_counts = (
        y.value_counts()
        .reindex(ALL_INTENTS, fill_value=0)
    )

    target_rows = len(real) / N_SPLITS
    target_counts = total_counts / N_SPLITS

    best = None

    for seed in SEEDS:

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        fold_ids = np.full(len(real), -1, dtype=int)

        for fold_number, (_, fold_indices) in enumerate(
            splitter.split(real, y, groups)
        ):
            fold_ids[fold_indices] = fold_number

        if (fold_ids < 0).any():
            raise RuntimeError(
                "Some real-world rows were not assigned to a fold."
            )

        for validation_fold, test_fold in combinations(
            range(N_SPLITS), 2
        ):

            validation = real[
                fold_ids == validation_fold
            ]

            test = real[
                fold_ids == test_fold
            ]

            train = real[
                ~np.isin(
                    fold_ids,
                    [validation_fold, test_fold]
                )
            ]

            train_clear_counts = (
                train[
                    train["intent"].isin(TRAIN_INTENTS)
                ]["intent"]
                .value_counts()
            )

            missing_train_classes = [
                label
                for label in TRAIN_INTENTS
                if train_clear_counts.get(label, 0) == 0
            ]

            score = (
                fold_score(
                    validation,
                    target_rows,
                    target_counts,
                )
                +
                fold_score(
                    test,
                    target_rows,
                    target_counts,
                )
                +
                1000 * len(missing_train_classes)
            )

            if best is None or score < best["score"]:

                best = {
                    "score": score,
                    "seed": seed,
                    "validation_fold": validation_fold,
                    "test_fold": test_fold,
                    "fold_ids": fold_ids.copy(),
                    "missing_train_classes": missing_train_classes,
                }

    if best is None:
        raise RuntimeError("Unable to find a valid group split.")

    return best


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 85)
    print("MODERAI — GROUP-AWARE ENGAGEMENT SPLITTING V2")
    print("=" * 85)

    # --------------------------------------------------------
    # Step 1: Check input and output paths
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = [
        name
        for name in OUTPUT_FILES
        if (OUTPUT_DIR / name).exists()
    ]

    if existing:
        raise FileExistsError(
            "Output files already exist. Nothing was overwritten: "
            f"{existing}"
        )

    # --------------------------------------------------------
    # Step 2: Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        dtype="string",
        keep_default_na=False,
    )

    required = {
        "comment_id",
        "comment_text",
        "intent",
        "dataset_type",
        "video_title",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs found.")

    if df["comment_text"].str.strip().eq("").any():
        raise ValueError("Blank comment text found.")

    if not set(df["intent"].unique()).issubset(ALL_INTENTS):
        raise ValueError("Unexpected intent label found.")

    if set(df["dataset_type"].unique()) != {
        "real_world",
        "synthetic",
    }:
        raise ValueError("Unexpected dataset_type value found.")

    # --------------------------------------------------------
    # Step 3: Separate real and synthetic comments
    # --------------------------------------------------------

    real = df[
        df["dataset_type"].eq("real_world")
    ].copy()

    synthetic = df[
        df["dataset_type"].eq("synthetic")
    ].copy()

    if len(real) != 400 or len(synthetic) != 30:
        raise ValueError(
            f"Expected 400 real + 30 synthetic; "
            f"found {len(real)} real + {len(synthetic)} synthetic."
        )

    if real["video_title"].str.strip().eq("").any():
        raise ValueError(
            "Real-world comments contain blank video titles."
        )

    # Check normalized text uniqueness.
    normalized = df["comment_text"].map(normalize_text)

    if normalized.duplicated().any():
        raise ValueError(
            "Duplicate normalized comment text found."
        )

    # --------------------------------------------------------
    # Step 4: Create video groups
    # --------------------------------------------------------

    real["_video_group"] = real["video_title"].map(
        normalize_group
    )

    print("\n1. INPUT DATA")
    print("-" * 85)

    print("Total comments:", len(df))
    print("Real-world comments:", len(real))
    print("Synthetic comments:", len(synthetic))
    print("Unique video groups:", real["_video_group"].nunique())

    print("\nReal-world intent distribution:")

    print(
        real["intent"]
        .value_counts()
        .reindex(ALL_INTENTS, fill_value=0)
        .to_string()
    )

    # --------------------------------------------------------
    # Step 5: Split all real-world comments
    # --------------------------------------------------------

    # IMPORTANT:
    # Split all six labels first, including UNCLEAR.
    # Do not remove UNCLEAR before group splitting.

    best = find_best_group_split(real)

    fold_ids = best["fold_ids"]

    validation_real = real[
        fold_ids == best["validation_fold"]
    ].copy()

    test_real = real[
        fold_ids == best["test_fold"]
    ].copy()

    train_real_all = real[
        ~np.isin(
            fold_ids,
            [
                best["validation_fold"],
                best["test_fold"],
            ],
        )
    ].copy()

    # --------------------------------------------------------
    # Step 6: Exclude UNCLEAR from supervised training
    # --------------------------------------------------------

    train_unclear_real = train_real_all[
        train_real_all["intent"].eq("UNCLEAR")
    ].copy()

    train_real = train_real_all[
        train_real_all["intent"].isin(TRAIN_INTENTS)
    ].copy()

    # Keep only five supervised classes from synthetic data.
    synthetic_clear = synthetic[
        synthetic["intent"].isin(TRAIN_INTENTS)
    ].copy()

    # Preserve synthetic UNCLEAR examples separately.
    synthetic_unclear = synthetic[
        synthetic["intent"].eq("UNCLEAR")
    ].copy()

    # Combine all excluded UNCLEAR examples.
    train_unclear = pd.concat(
        [
            train_unclear_real,
            synthetic_unclear,
        ],
        ignore_index=True,
        sort=False,
    )

    # Final supervised training dataset.
    train = pd.concat(
        [
            train_real,
            synthetic_clear,
        ],
        ignore_index=True,
        sort=False,
    )

    # --------------------------------------------------------
    # Step 7: Remove internal grouping column
    # --------------------------------------------------------

    train = train.drop(
        columns=["_video_group"],
        errors="ignore",
    )

    validation_real = validation_real.drop(
        columns=["_video_group"],
        errors="ignore",
    )

    test_real = test_real.drop(
        columns=["_video_group"],
        errors="ignore",
    )

    train_unclear = train_unclear.drop(
        columns=["_video_group"],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Step 8: Print split information
    # --------------------------------------------------------

    print("\n2. SELECTED SPLIT")
    print("-" * 85)

    print("Selected random seed:", best["seed"])
    print("Validation fold:", best["validation_fold"])
    print("Test fold:", best["test_fold"])
    print("Distribution score:", round(best["score"], 4))

    print("\n3. FINAL ROW COUNTS")
    print("-" * 85)

    print("Training rows:", len(train))
    print("  Real-world training rows:", len(train_real))
    print("  Synthetic training rows:", len(synthetic_clear))
    print("  Synthetic UNCLEAR excluded:", len(synthetic_unclear))

    print("Validation rows:", len(validation_real))
    print("Test rows:", len(test_real))

    print("UNCLEAR excluded from training:", len(train_unclear))

    # --------------------------------------------------------
    # Step 9: Intent distribution
    # --------------------------------------------------------

    print("\n4. INTENT DISTRIBUTIONS")
    print("-" * 85)

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation_real),
        ("TEST", test_real),
        ("TRAIN UNCLEAR EXCLUDED", train_unclear),
    ]:

        print(f"\n{name}:")

        print(
            frame["intent"]
            .value_counts()
            .reindex(ALL_INTENTS, fill_value=0)
            .to_string()
        )

    # --------------------------------------------------------
    # Step 10: Verify video group isolation
    # --------------------------------------------------------

    train_groups = set(
        train_real["video_title"].map(normalize_group)
    )

    validation_groups = set(
        validation_real["video_title"].map(normalize_group)
    )

    test_groups = set(
        test_real["video_title"].map(normalize_group)
    )

    if train_groups & validation_groups:
        raise RuntimeError(
            "Video leakage: train/validation overlap."
        )

    if train_groups & test_groups:
        raise RuntimeError(
            "Video leakage: train/test overlap."
        )

    if validation_groups & test_groups:
        raise RuntimeError(
            "Video leakage: validation/test overlap."
        )

    # --------------------------------------------------------
    # Step 11: Verify comment ID isolation
    # --------------------------------------------------------

    train_ids = set(train["comment_id"])
    validation_ids = set(validation_real["comment_id"])
    test_ids = set(test_real["comment_id"])
    excluded_ids = set(train_unclear["comment_id"])

    if train_ids & validation_ids:
        raise RuntimeError("Comment ID overlap: train/validation.")

    if train_ids & test_ids:
        raise RuntimeError("Comment ID overlap: train/test.")

    if validation_ids & test_ids:
        raise RuntimeError("Comment ID overlap: validation/test.")

    if train_ids & excluded_ids:
        raise RuntimeError("Excluded IDs found in training.")

    # --------------------------------------------------------
    # Step 12: Verify class and dataset rules
    # --------------------------------------------------------

    if set(train["intent"].unique()) != set(TRAIN_INTENTS):
        raise RuntimeError(
            "Training does not contain exactly five classes."
        )

    if train["intent"].eq("UNCLEAR").any():
        raise RuntimeError(
            "UNCLEAR unexpectedly found in training."
        )

    if set(validation_real["intent"].unique()) != set(ALL_INTENTS):
        raise RuntimeError(
            "Validation must contain all six intent labels."
        )

    if set(test_real["intent"].unique()) != set(ALL_INTENTS):
        raise RuntimeError(
            "Test must contain all six intent labels."
        )

    if len(synthetic_clear) != 25:
        raise RuntimeError(
            "Expected 25 synthetic training examples."
        )

    if len(synthetic_unclear) != 5:
        raise RuntimeError(
            "Expected 5 synthetic UNCLEAR examples."
        )

    # --------------------------------------------------------
    # Step 13: Verify every input row is accounted for
    # --------------------------------------------------------

    real_assigned_ids = (
        set(train_real["comment_id"])
        | set(validation_real["comment_id"])
        | set(test_real["comment_id"])
        | set(train_unclear_real["comment_id"])
    )

    synthetic_assigned_ids = (
        set(synthetic_clear["comment_id"])
        | set(synthetic_unclear["comment_id"])
    )

    if real_assigned_ids != set(real["comment_id"]):
        raise RuntimeError(
            "Some real-world comments are missing or duplicated "
            "across the split assignment."
        )

    if synthetic_assigned_ids != set(synthetic["comment_id"]):
        raise RuntimeError(
            "Some synthetic comments are missing or duplicated."
        )

    if len(train) + len(validation_real) + len(test_real) + len(train_unclear) != len(df):
        raise RuntimeError(
            "Final row accounting does not equal the input dataset size."
        )

    # --------------------------------------------------------
    # Step 14: Save output files
    # --------------------------------------------------------

    outputs = {
        "train_v2.csv": train,
        "validation_v2.csv": validation_real,
        "test_v2.csv": test_real,
        "train_unclear_excluded_v2.csv": train_unclear,
    }

    # All validations happen before this point.
    for filename, frame in outputs.items():

        frame.to_csv(
            OUTPUT_DIR / filename,
            index=False,
            encoding="utf-8",
        )

    # --------------------------------------------------------
    # Step 15: Save summary report
    # --------------------------------------------------------

    summary = [
        "MODERAI ENGAGEMENT SPLIT V2",
        "=" * 60,
        f"Input rows: {len(df)}",
        f"Real-world rows: {len(real)}",
        f"Synthetic rows: {len(synthetic)}",
        f"Selected seed: {best['seed']}",
        f"Validation fold: {best['validation_fold']}",
        f"Test fold: {best['test_fold']}",
        "",
        f"Training rows: {len(train)}",
        f"Validation rows: {len(validation_real)}",
        f"Test rows: {len(test_real)}",
        f"UNCLEAR excluded from training: {len(train_unclear)}",
        "",
        f"Training video groups: {len(train_groups)}",
        f"Validation video groups: {len(validation_groups)}",
        f"Test video groups: {len(test_groups)}",
        "",
        "Video group overlaps:",
        f"Train/validation: {len(train_groups & validation_groups)}",
        f"Train/test: {len(train_groups & test_groups)}",
        f"Validation/test: {len(validation_groups & test_groups)}",
        "",
        "Training intent distribution:",
        train["intent"].value_counts().to_string(),
        "",
        "Validation intent distribution:",
        validation_real["intent"].value_counts().to_string(),
        "",
        "Test intent distribution:",
        test_real["intent"].value_counts().to_string(),
        "",
        "Excluded UNCLEAR distribution:",
        train_unclear["intent"].value_counts().to_string(),
    ]

    (OUTPUT_DIR / "split_summary_v2.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Step 16: Final report
    # --------------------------------------------------------

    print("\n5. LEAKAGE VERIFICATION")
    print("-" * 85)

    print("[PASS] No video-title overlap between splits")
    print("[PASS] No comment ID overlap between splits")
    print("[PASS] Synthetic examples are training-only")
    print("[PASS] 25 synthetic examples included in training")
    print("[PASS] 5 synthetic UNCLEAR examples excluded")
    print("[PASS] UNCLEAR excluded from supervised training")
    print("[PASS] Training contains exactly five intent classes")
    print("[PASS] Validation contains all six intent labels")
    print("[PASS] Test contains all six intent labels")
    print("[PASS] All 430 input rows accounted for")

    print("\n6. OUTPUT FILES")
    print("-" * 85)

    for filename in OUTPUT_FILES:
        print(OUTPUT_DIR / filename)

    print("\nGroup-aware splitting completed successfully.")


if __name__ == "__main__":
    main()
