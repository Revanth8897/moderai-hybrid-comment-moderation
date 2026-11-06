
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv"
)

CONFIDENCE = {
    "YT000041": "MEDIUM", "YT000042": "HIGH",
    "YT000043": "HIGH", "YT000044": "HIGH",
    "YT000045": "MEDIUM", "YT000046": "HIGH",
    "YT000047": "HIGH", "YT000048": "HIGH",
    "YT000049": "HIGH", "YT000050": "HIGH",
    "YT000051": "HIGH", "YT000052": "HIGH",
    "YT000053": "HIGH", "YT000054": "HIGH",
    "YT000055": "MEDIUM", "YT000056": "HIGH",
    "YT000057": "HIGH", "YT000058": "HIGH",
    "YT000059": "HIGH", "YT000060": "HIGH",
    "YT000061": "HIGH", "YT000062": "HIGH",
    "YT000063": "HIGH", "YT000064": "MEDIUM",
    "YT000065": "LOW", "YT000066": "MEDIUM",
    "YT000067": "HIGH", "YT000068": "HIGH",
    "YT000069": "HIGH", "YT000070": "MEDIUM",
    "YT000071": "HIGH", "YT000072": "MEDIUM",
    "YT000073": "MEDIUM", "YT000074": "HIGH",
    "YT000075": "MEDIUM", "YT000076": "HIGH",
    "YT000077": "HIGH", "YT000078": "HIGH",
    "YT000079": "MEDIUM", "YT000080": "MEDIUM",
    "YT000081": "HIGH", "YT000082": "MEDIUM",
    "YT000083": "HIGH", "YT000084": "MEDIUM",
    "YT000085": "HIGH", "YT000086": "HIGH",
    "YT000087": "MEDIUM", "YT000088": "HIGH",
    "YT000089": "HIGH", "YT000090": "HIGH",
    "YT000091": "HIGH", "YT000092": "MEDIUM",
    "YT000093": "MEDIUM", "YT000094": "MEDIUM",
    "YT000095": "LOW", "YT000096": "HIGH",
    "YT000097": "HIGH", "YT000098": "HIGH",
    "YT000099": "HIGH", "YT000100": "HIGH",
}


def main():
    df = pd.read_csv(CSV_PATH, dtype={"comment_id": "string"})

    df["comment_id"] = df["comment_id"].str.strip()
    df["annotation_confidence"] = (
        df["annotation_confidence"].astype("string").fillna("")
    )

    # Validate every target ID before changing anything.
    for comment_id in CONFIDENCE:
        matches = df.index[df["comment_id"] == comment_id]

        if len(matches) != 1:
            raise ValueError(
                f"{comment_id}: expected one row, found {len(matches)}"
            )

        current = str(
            df.loc[matches[0], "annotation_confidence"]
        ).strip()

        expected = CONFIDENCE[comment_id]

        if current and current.upper() != expected:
            raise ValueError(
                f"{comment_id}: existing value conflicts with expected value"
            )

    # Restore only blank values.
    restored = 0

    for comment_id, expected in CONFIDENCE.items():
        index = df.index[df["comment_id"] == comment_id][0]

        if not str(df.at[index, "annotation_confidence"]).strip():
            df.at[index, "annotation_confidence"] = expected
            restored += 1

    # Verify before saving.
    for comment_id, expected in CONFIDENCE.items():
        index = df.index[df["comment_id"] == comment_id][0]

        actual = str(
            df.at[index, "annotation_confidence"]
        ).strip().upper()

        if actual != expected:
            raise ValueError(f"Verification failed: {comment_id}")

    # Preserve the original CSV as a backup.
    backup = CSV_PATH.with_name(
        "youtube_annotation_queue_before_confidence_restore.csv"
    )

    if not backup.exists():
        original = pd.read_csv(CSV_PATH)
        original.to_csv(backup, index=False, encoding="utf-8-sig")

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("\nCONFIDENCE RESTORATION SUCCESSFUL")
    print("=" * 45)
    print("Batch 3 rows:", len(CONFIDENCE))
    print("Values restored:", restored)
    print("Backup:", backup.name)

    print("\nOverall confidence distribution:")
    print(df["annotation_confidence"].value_counts().to_string())

    missing = (
        df["annotation_confidence"].str.strip().eq("").sum()
    )

    print("\nRemaining missing values:", missing)


if __name__ == "__main__":
    main()
