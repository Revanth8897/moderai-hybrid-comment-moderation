
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_PATH = PROJECT_ROOT / "datasets/engagement/raw/youtube_annotation_batch_6.csv"
MASTER_PATH = PROJECT_ROOT / "datasets/engagement/raw/youtube_annotation_queue.csv"

APPROVED = {
    "YT000361": ("NEGATIVE", "MEDIUM", "Expresses an unfavorable opinion about McDonald's."),
    "YT000362": ("CRITICAL", "HIGH", "Identifies problems with the weapons and recommends a better response from Germany."),
    "YT000363": ("NEUTRAL", "HIGH", "Uses a proverb without a clear request or emotional intent."),
    "YT000364": ("NEUTRAL", "HIGH", "Makes a humorous observation about movie scenes."),
    "YT000365": ("UNCLEAR", "LOW", "Unclear reference without enough context."),
    "YT000366": ("NEGATIVE", "HIGH", "Expresses personal disapproval and criticizes someone's parenting."),
    "YT000367": ("UNCLEAR", "LOW", "Repetitive text without identifiable intent."),
    "YT000368": ("NEGATIVE", "HIGH", "Expresses disapproval of how people treated someone."),
    "YT000369": ("UNCLEAR", "LOW", "Meme-style phrase without a clear standalone intent."),
    "YT000370": ("CRITICAL", "HIGH", "Raises concerns about a potentially serious public and political situation."),
    "YT000371": ("NEGATIVE", "MEDIUM", "Expresses skepticism and distrust toward the horse-riding demonstration."),
    "YT000372": ("NEUTRAL", "HIGH", "Shares a personal learning-progress update."),
    "YT000373": ("UNCLEAR", "LOW", "Brief reaction that lacks enough context."),
    "YT000374": ("POSITIVE", "MEDIUM", "Expresses appreciation for someone's actions."),
    "YT000375": ("NEGATIVE", "MEDIUM", "Expresses frustration about excessive volume."),
    "YT000376": ("QUESTION", "MEDIUM", "Uses a question to invite identification of an association."),
    "YT000377": ("CRITICAL", "HIGH", "Criticizes insurance companies and explains consequences for hospitals and patients."),
    "YT000378": ("CRITICAL", "HIGH", "Presents an argument about workers' contribution to production and wealth."),
    "YT000379": ("NEGATIVE", "HIGH", "Strongly expresses political disapproval."),
    "YT000380": ("CRITICAL", "HIGH", "Explains a potential long-term impact of AI on software development employment."),
}

EXPECTED_IDS = [f"YT{i:06d}" for i in range(361, 381)]

def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())

def find_text_column(df):
    for col in ("comment_text", "text"):
        if col in df.columns:
            return col
    raise ValueError("No comment_text or text column found.")

def main():
    if not BATCH_PATH.exists() or not MASTER_PATH.exists():
        raise FileNotFoundError("Batch or master queue file is missing.")

    batch = pd.read_csv(BATCH_PATH, dtype="string").fillna("")
    master = pd.read_csv(MASTER_PATH, dtype="string").fillna("")

    batch_text_col = find_text_column(batch)
    master_text_col = find_text_column(master)

    selected = batch[batch["comment_id"].isin(EXPECTED_IDS)].copy()

    if len(selected) != 20:
        raise ValueError(f"Expected 20 comments, found {len(selected)}.")

    if selected["comment_id"].duplicated().any():
        raise ValueError("Duplicate IDs found in batch.")

    if set(selected["comment_id"]) != set(EXPECTED_IDS):
        raise ValueError("Missing expected comment IDs.")

    if set(APPROVED) != set(EXPECTED_IDS):
        raise ValueError("Annotation dictionary does not match expected IDs.")

    if master["comment_id"].isin(EXPECTED_IDS).any():
        raise ValueError("Some Group 4 IDs already exist in master queue.")

    batch_texts = selected[batch_text_col].map(normalize_text)
    master_texts = set(master[master_text_col].map(normalize_text))

    if batch_texts.eq("").any() or batch_texts.duplicated().any():
        raise ValueError("Empty or duplicate comment text found in batch.")

    if batch_texts.isin(master_texts).any():
        raise ValueError("Comment text already exists in master queue.")

    # Ensure annotation columns exist.
    for col in (
        "intent", "annotator", "annotation_confidence",
        "intent_confidence", "annotation_notes", "annotation_note"
    ):
        if col not in batch.columns:
            batch[col] = ""

    # Apply annotations.
    for idx in selected.index:
        comment_id = batch.at[idx, "comment_id"]
        intent, confidence, note = APPROVED[comment_id]

        batch.at[idx, "intent"] = intent
        batch.at[idx, "annotator"] = "human_approved_assisted"
        batch.at[idx, "annotation_confidence"] = confidence
        batch.at[idx, "intent_confidence"] = confidence
        batch.at[idx, "annotation_notes"] = note
        batch.at[idx, "annotation_note"] = note

    selected = batch[batch["comment_id"].isin(EXPECTED_IDS)].copy()

    for col in master.columns:
        if col not in selected.columns:
            selected[col] = ""

    selected = selected[master.columns]
    updated_master = pd.concat([master, selected], ignore_index=True)

    if updated_master["comment_id"].duplicated().any():
        raise ValueError("Duplicate IDs detected after combining.")

    # Create backups.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_backup = BATCH_PATH.with_name(
        f"{BATCH_PATH.stem}_before_group4_{timestamp}.csv"
    )
    master_backup = MASTER_PATH.with_name(
        f"{MASTER_PATH.stem}_before_group4_{timestamp}.csv"
    )

    shutil.copy2(BATCH_PATH, batch_backup)
    shutil.copy2(MASTER_PATH, master_backup)

    # Write temporary files and replace originals.
    batch_temp = BATCH_PATH.with_name(BATCH_PATH.stem + ".tmp")
    master_temp = MASTER_PATH.with_name(MASTER_PATH.stem + ".tmp")

    try:
        batch.to_csv(batch_temp, index=False)
        updated_master.to_csv(master_temp, index=False)

        batch_temp.replace(BATCH_PATH)
        master_temp.replace(MASTER_PATH)
    finally:
        if batch_temp.exists():
            batch_temp.unlink()
        if master_temp.exists():
            master_temp.unlink()

    # Reload and verify.
    saved_batch = pd.read_csv(BATCH_PATH, dtype="string").fillna("")
    saved_master = pd.read_csv(MASTER_PATH, dtype="string").fillna("")

    saved_group = saved_batch[
        saved_batch["comment_id"].isin(EXPECTED_IDS)
    ]

    if len(saved_group) != 20:
        raise RuntimeError("Batch verification failed.")

    if not saved_group["annotator"].eq("human_approved_assisted").all():
        raise RuntimeError("Annotator verification failed.")

    for comment_id, (intent, confidence, _) in APPROVED.items():
        row = saved_group[saved_group["comment_id"] == comment_id].iloc[0]

        if row["intent"] != intent:
            raise RuntimeError(f"Intent mismatch: {comment_id}")

        if row["annotation_confidence"] != confidence:
            raise RuntimeError(f"Confidence mismatch: {comment_id}")

    if len(saved_master) != len(master) + 20:
        raise RuntimeError("Master queue row-count verification failed.")

    if not set(EXPECTED_IDS).issubset(set(saved_master["comment_id"])):
        raise RuntimeError("Missing IDs in master queue.")

    print("=" * 65)
    print("MODERAI — GROUP 4 ANNOTATIONS SAVED")
    print("=" * 65)
    print(f"Approved comments: {len(saved_group)}")
    print(f"Master queue before: {len(master)}")
    print(f"Master queue after: {len(saved_master)}")

    print("\nIntent distribution:")
    print(saved_group["intent"].value_counts().to_string())

    print("\nConfidence distribution:")
    print(saved_group["annotation_confidence"].value_counts().to_string())

    print("\nAnnotator status:")
    print(saved_group["annotator"].value_counts().to_string())

    print("\nBackups:")
    print(batch_backup)
    print(master_backup)

    print("\nVerification: PASSED")

if __name__ == "__main__":
    main()
