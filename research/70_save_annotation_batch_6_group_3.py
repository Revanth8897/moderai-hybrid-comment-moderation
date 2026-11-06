
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_PATH = PROJECT_ROOT / "datasets/engagement/raw/youtube_annotation_batch_6.csv"
MASTER_PATH = PROJECT_ROOT / "datasets/engagement/raw/youtube_annotation_queue.csv"

APPROVED = {
    "YT000341": ("QUESTION", "HIGH", "Requests a video explaining a common interview question."),
    "YT000342": ("UNCLEAR", "LOW", "Emojis alone do not establish a clear intent."),
    "YT000343": ("CRITICAL", "MEDIUM", "Criticizes unequal legal accountability."),
    "YT000344": ("CRITICAL", "HIGH", "Specifically criticizes Britain's leadership and government."),
    "YT000345": ("POSITIVE", "HIGH", "Shows enthusiasm and appreciation for learning Go."),
    "YT000346": ("POSITIVE", "HIGH", "Expresses gratitude for the creator's work."),
    "YT000347": ("NEGATIVE", "MEDIUM", "Expresses distress or disapproval about the dog's situation."),
    "YT000348": ("POSITIVE", "HIGH", "Strongly praises the talk."),
    "YT000349": ("NEUTRAL", "HIGH", "Shares additional factual information about another video."),
    "YT000350": ("POSITIVE", "HIGH", "Praises the video and its opening."),
    "YT000351": ("POSITIVE", "HIGH", "Expresses gratitude and explains how the talk inspired them."),
    "YT000352": ("POSITIVE", "HIGH", "Praises the tutorial series and shares a positive outcome."),
    "YT000353": ("POSITIVE", "HIGH", "Expresses detailed appreciation for educational content."),
    "YT000354": ("NEUTRAL", "MEDIUM", "Makes an observation about the timing of the activity."),
    "YT000355": ("POSITIVE", "HIGH", "Expresses affection and appreciation."),
    "YT000356": ("NEGATIVE", "HIGH", "Expresses distrust and disapproval of political accountability."),
    "YT000357": ("NEUTRAL", "MEDIUM", "Shares a code example without a clear request or opinion."),
    "YT000358": ("NEUTRAL", "HIGH", "Shares a factual summary of the presentation."),
    "YT000359": ("UNCLEAR", "LOW", "A score without context does not reveal the intended message."),
    "YT000360": ("CRITICAL", "HIGH", "Highlights environmental harm caused by plastic pollution."),
}

EXPECTED_IDS = [f"YT{i:06d}" for i in range(341, 361)]

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

    if master["comment_id"].isin(EXPECTED_IDS).any():
        raise ValueError("Some Group 3 IDs already exist in master queue.")

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

    # Apply approved annotations.
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

    # Backups before modifying original files.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_backup = BATCH_PATH.with_name(
        f"{BATCH_PATH.stem}_before_group3_{timestamp}.csv"
    )
    master_backup = MASTER_PATH.with_name(
        f"{MASTER_PATH.stem}_before_group3_{timestamp}.csv"
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
    print("MODERAI — GROUP 3 ANNOTATIONS SAVED")
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
