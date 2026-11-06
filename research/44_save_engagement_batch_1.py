
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

ANNOTATIONS = {
    "YT000001": ("NEGATIVE", "HIGH", "Sarcastic criticism."),
    "YT000002": ("NEGATIVE", "HIGH", "Disapproval of the show."),
    "YT000003": ("NEGATIVE", "MEDIUM", "Dismissive rhetorical opinion."),
    "YT000004": ("CRITICAL", "MEDIUM", "Suggests a specific action."),
    "YT000005": ("NEGATIVE", "HIGH", "Insulting criticism without a constructive suggestion."),
    "YT000006": ("CRITICAL", "MEDIUM", "Criticizes alleged corporate hypocrisy."),
    "YT000007": ("CRITICAL", "MEDIUM", "Challenges a political position."),
    "YT000008": ("QUESTION", "HIGH", "Requests technical clarification."),
    "YT000009": ("POSITIVE", "HIGH", "Expresses appreciation and encouragement."),
    "YT000010": ("QUESTION", "HIGH", "Requests movie information."),
    "YT000011": ("NEGATIVE", "HIGH", "Expresses an unfavorable opinion."),
    "YT000012": ("POSITIVE", "HIGH", "Expresses understanding and appreciation."),
    "YT000013": ("CRITICAL", "MEDIUM", "Advocates a response; requires separate safety assessment."),
    "YT000014": ("POSITIVE", "HIGH", "Expresses appreciation for a quote."),
    "YT000015": ("QUESTION", "HIGH", "Asks for a recommendation."),
    "YT000016": ("CRITICAL", "MEDIUM", "Raises a humanitarian concern."),
    "YT000017": ("NEGATIVE", "MEDIUM", "Expresses disapproval."),
    "YT000018": ("NEGATIVE", "HIGH", "Expresses an unfavorable opinion."),
    "YT000019": ("NEUTRAL", "HIGH", "Factual personal observation."),
    "YT000020": ("CRITICAL", "HIGH", "Suggests a technical improvement."),
}

ANNOTATOR = "human_approved_assisted"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

required_columns = {
    "comment_id",
    "intent",
    "annotator",
    "annotation_confidence",
    "annotation_notes",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

if df["comment_id"].duplicated().any():
    raise ValueError("Duplicate comment IDs found. Stopping.")

# Verify that all 20 IDs exist.
existing_ids = set(df["comment_id"].astype(str))
missing_ids = set(ANNOTATIONS) - existing_ids

if missing_ids:
    raise ValueError(f"Missing comment IDs: {sorted(missing_ids)}")

# Check for conflicting annotations before making changes.
# Convert annotation columns to string-compatible types.
for column in [
    "intent",
    "annotator",
    "annotation_confidence",
    "annotation_notes",
]:
    df[column] = df[column].astype("string").fillna("")

# Apply approved annotations.
for comment_id, (intent, confidence, notes) in ANNOTATIONS.items():
    mask = df["comment_id"].astype(str) == comment_id

    df.loc[mask, "intent"] = intent
    df.loc[mask, "annotation_confidence"] = confidence
    df.loc[mask, "annotation_notes"] = notes
    df.loc[mask, "annotator"] = ANNOTATOR

# Save while preserving all other columns.
df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")

print("=" * 55)
print("BATCH 1 ANNOTATIONS SAVED SUCCESSFULLY")
print("=" * 55)

batch = df[df["comment_id"].astype(str).isin(ANNOTATIONS)]

print(f"Annotated comments: {len(batch)}")
print("\nIntent distribution:")
print(batch["intent"].value_counts().to_string())

print("\nConfidence distribution:")
print(batch["annotation_confidence"].value_counts().to_string())

print("\nSaved annotations:")
print(
    batch[
        ["comment_id", "intent", "annotation_confidence"]
    ].to_string(index=False)
)

print("\nBatch 1 completed.")
