
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "engagement_comments.csv"
)

EXPECTED_LABELS = {
    **{f"ENG{i:06d}": "POSITIVE" for i in range(1, 6)},
    **{f"ENG{i:06d}": "NEGATIVE" for i in range(6, 11)},
    **{f"ENG{i:06d}": "CRITICAL" for i in range(11, 16)},
    **{f"ENG{i:06d}": "QUESTION" for i in range(16, 21)},
    **{f"ENG{i:06d}": "NEUTRAL" for i in range(21, 26)},
    **{f"ENG{i:06d}": "UNCLEAR" for i in range(26, 31)},
}


def main():
    df = pd.read_csv(CSV_PATH, dtype={"comment_id": "string"})

    required = {"comment_id", "intent", "annotator"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["comment_id"] = df["comment_id"].str.strip()
    df["intent"] = df["intent"].astype("string").str.strip().str.upper()
    df["annotator"] = df["annotator"].astype("string").fillna("")

    if len(df) != 30:
        raise ValueError(f"Expected 30 rows, found {len(df)}")

    if df["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs found")

    if set(df["comment_id"]) != set(EXPECTED_LABELS):
        raise ValueError("The synthetic comment IDs do not match the review list")

    for comment_id, expected_intent in EXPECTED_LABELS.items():
        row = df.loc[df["comment_id"] == comment_id]

        actual_intent = row.iloc[0]["intent"]

        if actual_intent != expected_intent:
            raise ValueError(
                f"{comment_id}: expected {expected_intent}, "
                f"found {actual_intent}"
            )

    allowed_statuses = {"pending_review", "human_approved_assisted"}

    unexpected = set(df["annotator"]) - allowed_statuses

    if unexpected:
        raise ValueError(f"Unexpected annotator statuses: {unexpected}")

    backup = CSV_PATH.with_name(
        "engagement_comments_before_approval.csv"
    )

    if not backup.exists():
        pd.read_csv(CSV_PATH).to_csv(
            backup,
            index=False,
            encoding="utf-8-sig",
        )

    changed = df["annotator"].ne("human_approved_assisted").sum()

    df["annotator"] = "human_approved_assisted"

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # Verify the saved dataset.
    saved = pd.read_csv(CSV_PATH, dtype={"comment_id": "string"})

    if len(saved) != 30:
        raise ValueError("Saved row count verification failed")

    if not saved["annotator"].eq("human_approved_assisted").all():
        raise ValueError("Approval status verification failed")

    print("\nMODERAI — SYNTHETIC SEED REVIEW COMPLETED")
    print("=" * 55)
    print("Reviewed examples:", len(saved))
    print("Approval statuses updated:", changed)
    print("Intent labels changed: 0")
    print("Backup:", backup.name)

    print("\nIntent distribution:")
    print(saved["intent"].value_counts().to_string())

    print("\nAnnotator status:")
    print(saved["annotator"].value_counts().to_string())

    print("\nVerification: PASSED")


if __name__ == "__main__":
    main()
