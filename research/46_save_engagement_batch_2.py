
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "engagement"
    / "raw"
    / "youtube_annotation_queue.csv"
)

ANNOTATOR = "human_approved_assisted"

ANNOTATIONS = {
    "YT000021": ("POSITIVE", "HIGH", "Expresses sympathy and support."),
    "YT000022": ("POSITIVE", "MEDIUM", "Playful compliment about appearance."),
    "YT000023": ("CRITICAL", "HIGH", "Detailed technical criticism about power consumption."),
    "YT000024": ("NEGATIVE", "HIGH", "Direct disapproval."),
    "YT000025": ("QUESTION", "HIGH", "Requests advice."),
    "YT000026": ("POSITIVE", "HIGH", "Expresses political support."),
    "YT000027": ("CRITICAL", "HIGH", "Presents a detailed argument against affirmative action."),
    "YT000028": ("NEGATIVE", "HIGH", "Expresses dissatisfaction."),
    "YT000029": ("UNCLEAR", "MEDIUM", "Insufficient context to establish the intended engagement."),
    "YT000030": ("CRITICAL", "HIGH", "Criticizes behavior and suggests improvement."),
    "YT000031": ("POSITIVE", "HIGH", "Expresses enthusiastic appreciation."),
    "YT000032": ("CRITICAL", "MEDIUM", "Recommends alternative driving equipment."),
    "YT000033": ("POSITIVE", "HIGH", "Expresses excitement and appreciation."),
    "YT000034": ("POSITIVE", "HIGH", "Expresses encouragement and support."),
    "YT000035": ("QUESTION", "HIGH", "Explicitly asks a genuine question."),
    "YT000036": ("NEUTRAL", "MEDIUM", "Humorous comparison without clear positive or negative intent."),
    "YT000037": ("POSITIVE", "HIGH", "Appreciates the educational content."),
    "YT000038": ("POSITIVE", "HIGH", "Expresses enjoyment and appreciation."),
    "YT000039": ("POSITIVE", "HIGH", "Compliments the person."),
    "YT000040": ("POSITIVE", "MEDIUM", "Playful appreciation of the video moment."),
}

if not DATASET_PATH.exists():
    raise FileNotFoundError(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "comment_id",
    "intent",
    "annotator",
    "annotation_confidence",
    "annotation_notes",
]

missing = set(required_columns) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df["comment_id"].duplicated().any():
    raise ValueError("Duplicate comment IDs found.")

existing_ids = set(df["comment_id"].astype(str))
missing_ids = set(ANNOTATIONS) - existing_ids

if missing_ids:
    raise ValueError(f"Missing IDs: {sorted(missing_ids)}")

# Check for conflicting annotations before writing.
for comment_id, (intent, confidence, notes) in ANNOTATIONS.items():
    row = df.loc[
        df["comment_id"].astype(str) == comment_id
    ].iloc[0]

    current = row["intent"]

    if pd.notna(current) and str(current).strip():
        if str(current).strip() != intent:
            raise ValueError(
                f"{comment_id} already has conflicting intent: {current}"
            )

# Use nullable string dtype to safely handle previously empty columns.
for column in [
    "intent",
    "annotator",
    "annotation_confidence",
    "annotation_notes",
]:
    df[column] = df[column].astype("string").fillna("")

# Apply annotations.
for comment_id, (intent, confidence, notes) in ANNOTATIONS.items():
    mask = df["comment_id"].astype(str) == comment_id

    df.loc[mask, "intent"] = intent
    df.loc[mask, "annotation_confidence"] = confidence
    df.loc[mask, "annotation_notes"] = notes
    df.loc[mask, "annotator"] = ANNOTATOR

df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")

# Verify saved values.
saved = pd.read_csv(DATASET_PATH)
batch = saved[
    saved["comment_id"].astype(str).isin(ANNOTATIONS)
]

if len(batch) != 20 or batch["intent"].isna().any():
    raise ValueError("Post-save verification failed.")

print("=" * 55)
print("BATCH 2 SAVED AND VERIFIED")
print("=" * 55)

print(f"Annotated comments: {len(batch)}")

print("\nIntent distribution:")
print(batch["intent"].value_counts().to_string())

print("\nConfidence distribution:")
print(batch["annotation_confidence"].value_counts().to_string())

print("\nBatch 2 completed successfully.")
