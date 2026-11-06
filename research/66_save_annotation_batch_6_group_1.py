
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BATCH_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_batch_6.csv"
)

MASTER_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv"
)

APPROVED_BY = "human_approved_assisted"

# Approved human-reviewed annotations.
ANNOTATIONS = {
    "YT000301": {
        "intent": "NEGATIVE",
        "confidence": "HIGH",
        "note": "Expresses dissatisfaction with the economy and political situation.",
    },
    "YT000302": {
        "intent": "QUESTION",
        "confidence": "HIGH",
        "note": "Asks how many times Jensen said AI.",
    },
    "YT000303": {
        "intent": "CRITICAL",
        "confidence": "HIGH",
        "note": "Identifies outdated React Router code and explains the required correction.",
    },
    "YT000304": {
        "intent": "NEUTRAL",
        "confidence": "MEDIUM",
        "note": "Provides a factual clarification about the vehicle starting issue.",
    },
    "YT000305": {
        "intent": "QUESTION",
        "confidence": "HIGH",
        "note": "Requests an explanation of the algorithm.",
    },
    "YT000306": {
        "intent": "QUESTION",
        "confidence": "HIGH",
        "note": "Asks whether the honeycomb is real.",
    },
    "YT000307": {
        "intent": "POSITIVE",
        "confidence": "HIGH",
        "note": "Expresses praise for the sheriff.",
    },
    "YT000308": {
        "intent": "NEGATIVE",
        "confidence": "MEDIUM",
        "note": "Tagalog comment accusing Ma'am Risa of turning a blind eye.",
    },
    "YT000309": {
        "intent": "POSITIVE",
        "confidence": "HIGH",
        "note": "Appreciates Oscar's humility.",
    },
    "YT000310": {
        "intent": "POSITIVE",
        "confidence": "HIGH",
        "note": "Praises the tutorial and expresses gratitude.",
    },
    "YT000311": {
        "intent": "POSITIVE",
        "confidence": "HIGH",
        "note": "Compliments the instructor and expresses excitement about the course.",
    },
    "YT000312": {
        "intent": "CRITICAL",
        "confidence": "HIGH",
        "note": "Calls for concrete action from the West instead of empty statements.",
    },
    "YT000313": {
        "intent": "NEGATIVE",
        "confidence": "HIGH",
        "note": "Expresses disapproval and shame regarding the US.",
    },
    "YT000314": {
        "intent": "POSITIVE",
        "confidence": "HIGH",
        "note": "Expresses gratitude and appreciation for an encouraging message.",
    },
    "YT000315": {
        "intent": "CRITICAL",
        "confidence": "HIGH",
        "note": "Criticizes leadership and explicitly recommends firing the CEO.",
    },
    "YT000316": {
        "intent": "NEUTRAL",
        "confidence": "MEDIUM",
        "note": "Makes a humorous observation without a clear request or strong sentiment.",
    },
    "YT000317": {
        "intent": "CRITICAL",
        "confidence": "HIGH",
        "note": "Criticizes reporting accuracy and calls for better reporting.",
    },
    "YT000318": {
        "intent": "NEGATIVE",
        "confidence": "MEDIUM",
        "note": "Expresses disapproval and disbelief about the situation.",
    },
    "YT000319": {
        "intent": "POSITIVE",
        "confidence": "HIGH",
        "note": "Praises the explanation for improving understanding.",
    },
    "YT000320": {
        "intent": "CRITICAL",
        "confidence": "MEDIUM",
        "note": "Criticizes corrupt politicians and advocates severe punishment; separately flag violent content for safety review.",
    },
}


def normalize_text(value):
    return " ".join(str(value).strip().lower().split())


def make_backup(path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(
        f"{path.stem}_before_group1_{timestamp}{path.suffix}"
    )
    shutil.copy2(path, backup_path)
    return backup_path


def main():
    if not BATCH_PATH.exists():
        raise FileNotFoundError(BATCH_PATH)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(MASTER_PATH)

    batch = pd.read_csv(BATCH_PATH, dtype=str).fillna("")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")

    expected_ids = set(ANNOTATIONS)
    actual_ids = set(batch["comment_id"])

    # Validate that all expected comments exist in the batch.
    if not expected_ids.issubset(actual_ids):
        missing = expected_ids - actual_ids
        raise ValueError(f"Missing batch IDs: {sorted(missing)}")

    if len(ANNOTATIONS) != 20:
        raise ValueError("Expected exactly 20 approved annotations")

    if batch["comment_id"].duplicated().any():
        raise ValueError("Duplicate IDs found in batch")

    if master["comment_id"].duplicated().any():
        raise ValueError("Duplicate IDs already exist in master queue")

    # Ensure none of the approved IDs were previously added.
    overlap = expected_ids & set(master["comment_id"])

    if overlap:
        raise ValueError(
            f"These IDs already exist in master queue: {sorted(overlap)}"
        )

    # Ensure comments are not duplicates of existing master comments.
    master_texts = {
        normalize_text(value)
        for value in master["comment_text"]
        if normalize_text(value)
    }

    selected = batch[
        batch["comment_id"].isin(expected_ids)
    ].copy()

    selected_texts = {
        normalize_text(value)
        for value in selected["comment_text"]
    }

    if len(selected_texts) != 20:
        raise ValueError("Duplicate comment text inside approved group")

    if selected_texts & master_texts:
        raise ValueError(
            "Some approved comments already exist in the master queue"
        )

    # Apply approved annotations to batch.
    for index, row in selected.iterrows():
        annotation = ANNOTATIONS[row["comment_id"]]

        batch.at[index, "intent"] = annotation["intent"]
        batch.at[index, "annotator"] = APPROVED_BY
        batch.at[index, "annotation_confidence"] = annotation["confidence"]
        batch.at[index, "intent_confidence"] = annotation["confidence"]
        batch.at[index, "annotation_notes"] = annotation["note"]
        batch.at[index, "annotation_note"] = annotation["note"]

    # Select only the approved rows for the master queue.
    approved_rows = batch[
        batch["comment_id"].isin(expected_ids)
    ].copy()

    # Align columns to the master queue schema.
    for column in master.columns:
        if column not in approved_rows.columns:
            approved_rows[column] = ""

    approved_rows = approved_rows[master.columns]

    updated_master = pd.concat(
        [master, approved_rows],
        ignore_index=True,
    )

    # Final validation before modifying files.
    if len(approved_rows) != 20:
        raise ValueError("Expected 20 approved rows")

    if len(updated_master) != len(master) + 20:
        raise ValueError("Master queue row count mismatch")

    if updated_master["comment_id"].duplicated().any():
        raise ValueError("Duplicate IDs detected after merging")

    if not approved_rows["annotator"].eq(APPROVED_BY).all():
        raise ValueError("Approval status verification failed")

    if approved_rows["intent"].eq("").any():
        raise ValueError("Some approved labels are empty")

    # Create backups before writing.
    batch_backup = make_backup(BATCH_PATH)
    master_backup = make_backup(MASTER_PATH)

    # Write updated files.
    batch_temp = BATCH_PATH.with_suffix(".tmp")
    master_temp = MASTER_PATH.with_suffix(".tmp")

    batch.to_csv(batch_temp, index=False, encoding="utf-8-sig")
    updated_master.to_csv(
        master_temp,
        index=False,
        encoding="utf-8-sig",
    )

    batch_temp.replace(BATCH_PATH)
    master_temp.replace(MASTER_PATH)

    # Read back and verify saved files.
    saved_batch = pd.read_csv(BATCH_PATH, dtype=str).fillna("")
    saved_master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")

    saved_group = saved_batch[
        saved_batch["comment_id"].isin(expected_ids)
    ]

    if len(saved_group) != 20:
        raise ValueError("Saved batch verification failed")

    if not saved_group["annotator"].eq(APPROVED_BY).all():
        raise ValueError("Saved batch approval status mismatch")

    if not expected_ids.issubset(set(saved_master["comment_id"])):
        raise ValueError("Approved IDs missing from master queue")

    if len(saved_master) != len(master) + 20:
        raise ValueError("Saved master queue count mismatch")

    print("\nMODERAI — GROUP 1 ANNOTATIONS SAVED")
    print("=" * 65)

    print("Approved comments:", len(saved_group))
    print("Master queue before:", len(master))
    print("Master queue after:", len(saved_master))

    print("\nIntent distribution:")
    print(saved_group["intent"].value_counts().to_string())

    print("\nConfidence distribution:")
    print(
        saved_group["annotation_confidence"]
        .value_counts()
        .to_string()
    )

    print("\nAnnotator status:")
    print(saved_group["annotator"].value_counts().to_string())

    print("\nBackups:")
    print(batch_backup)
    print(master_backup)

    print("\nVerification: PASSED")


if __name__ == "__main__":
    main()
