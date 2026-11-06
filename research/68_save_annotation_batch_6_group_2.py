
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_PATH = (
    PROJECT_ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_batch_6.csv"
)
MASTER_PATH = (
    PROJECT_ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv"
)

APPROVED = {
    "YT000321": ("UNCLEAR", "LOW", "Short phrase without enough context to identify intent."),
    "YT000322": ("NEGATIVE", "HIGH", "Expresses disapproval through a political accusation."),
    "YT000323": ("NEGATIVE", "MEDIUM", "Sarcastic remark mocking a public figure."),
    "YT000324": ("QUESTION", "HIGH", "Asks what someone put in their pocket."),
    "YT000325": ("NEGATIVE", "MEDIUM", "Strong expression of dissatisfaction."),
    "YT000326": ("CRITICAL", "HIGH", "Gives specific criticism of debate performance and campaign strategy."),
    "YT000327": ("NEUTRAL", "HIGH", "Adds a factual observation about another movie scene."),
    "YT000328": ("NEGATIVE", "HIGH", "Expresses disagreement and uses insulting language."),
    "YT000329": ("POSITIVE", "HIGH", "Expresses support and goodwill."),
    "YT000330": ("NEGATIVE", "HIGH", "Uses strongly unfavorable language about a political figure."),
    "YT000331": ("POSITIVE", "HIGH", "Expresses celebration and goodwill."),
    "YT000332": ("QUESTION", "HIGH", "Asks a technical question about an angle calculation."),
    "YT000333": ("NEUTRAL", "MEDIUM", "Offers speculation about the cause of an event."),
    "YT000334": ("POSITIVE", "MEDIUM", "Makes an admiring, humorous observation."),
    "YT000335": ("NEGATIVE", "HIGH", "Expresses disapproval by calling the media biased."),
    "YT000336": ("POSITIVE", "HIGH", "Directly praises the creator's content."),
    "YT000337": ("QUESTION", "HIGH", "Asks whether the drop bear is real."),
    "YT000338": ("QUESTION", "HIGH", "Asks whether the object is a real M16."),
    "YT000339": ("UNCLEAR", "HIGH", "Timestamp-only comment without identifiable standalone intent."),
    "YT000340": ("NEGATIVE", "HIGH", "Sarcastically insults the person being discussed."),
}

EXPECTED_IDS = [f"YT{i:06d}" for i in range(321, 341)]

def normalize_text(value):
    value = str(value).strip().lower()
    return re.sub(r"\s+", " ", value)

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

    if "comment_id" not in batch.columns or "comment_id" not in master.columns:
        raise ValueError("Both CSV files must contain comment_id.")

    batch_text_col = find_text_column(batch)
    master_text_col = find_text_column(master)

    if set(APPROVED) != set(EXPECTED_IDS):
        raise ValueError("Annotation dictionary does not match expected IDs.")

    selected = batch[batch["comment_id"].isin(EXPECTED_IDS)].copy()

    if len(selected) != 20:
        raise ValueError(f"Expected 20 batch comments; found {len(selected)}.")

    if selected["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs found in batch.")

    if set(selected["comment_id"]) != set(EXPECTED_IDS):
        raise ValueError("One or more expected IDs are missing.")

    if master["comment_id"].isin(EXPECTED_IDS).any():
        raise ValueError("Some Group 2 IDs already exist in the master queue.")

    batch_texts = selected[batch_text_col].map(normalize_text)
    master_texts = master[master_text_col].map(normalize_text)

    if batch_texts.eq("").any():
        raise ValueError("Empty comment text found in selected batch.")

    if batch_texts.duplicated().any():
        raise ValueError("Duplicate comment text found within Group 2.")

    if batch_texts.isin(set(master_texts)).any():
        raise ValueError("A Group 2 comment already exists in the master queue.")

    # Apply approved annotations to the batch.
    for col in (
        "intent",
        "annotator",
        "annotation_confidence",
        "intent_confidence",
        "annotation_notes",
        "annotation_note",
    ):
        if col not in batch.columns:
            batch[col] = ""

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

    # Align batch columns with the master queue schema.
    for col in master.columns:
        if col not in selected.columns:
            selected[col] = ""

    selected = selected[master.columns]

    updated_master = pd.concat(
        [master, selected],
        ignore_index=True
    )

    if updated_master["comment_id"].duplicated().any():
        raise ValueError("Duplicate IDs detected after combining.")

    # Create timestamped backups before writing.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_backup = BATCH_PATH.with_name(
        f"{BATCH_PATH.stem}_before_group2_{timestamp}.csv"
    )
    master_backup = MASTER_PATH.with_name(
        f"{MASTER_PATH.stem}_before_group2_{timestamp}.csv"
    )

    shutil.copy2(BATCH_PATH, batch_backup)
    shutil.copy2(MASTER_PATH, master_backup)

    # Write temporary files, then replace originals.
    batch_temp = BATCH_PATH.with_name(BATCH_PATH.stem + ".tmp")
    master_temp = MASTER_PATH.with_name(MASTER_PATH.stem + ".tmp")

    batch.to_csv(batch_temp, index=False)
    updated_master.to_csv(master_temp, index=False)

    batch_temp.replace(BATCH_PATH)
    master_temp.replace(MASTER_PATH)

    # Reload and verify saved files.
    saved_batch = pd.read_csv(BATCH_PATH, dtype="string").fillna("")
    saved_master = pd.read_csv(MASTER_PATH, dtype="string").fillna("")

    saved_group = saved_batch[
        saved_batch["comment_id"].isin(EXPECTED_IDS)
    ]

    if len(saved_group) != 20:
        raise RuntimeError("Batch verification failed: expected 20 rows.")

    if not saved_group["annotator"].eq("human_approved_assisted").all():
        raise RuntimeError("Batch verification failed: annotator status.")

    if not saved_group["intent"].isin(
        ["POSITIVE", "NEGATIVE", "CRITICAL", "QUESTION", "NEUTRAL", "UNCLEAR"]
    ).all():
        raise RuntimeError("Batch verification failed: invalid intent.")

    if len(saved_master) != len(master) + 20:
        raise RuntimeError("Master queue row-count verification failed.")

    if not set(EXPECTED_IDS).issubset(set(saved_master["comment_id"])):
        raise RuntimeError("Master queue verification failed: missing IDs.")

    print("=" * 65)
    print("MODERAI — GROUP 2 ANNOTATIONS SAVED")
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
