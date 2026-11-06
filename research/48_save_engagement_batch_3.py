
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "engagement"
    / "raw"
    / "youtube_annotation_queue.csv"
)

ANNOTATOR = "human_approved_assisted"

# Approved annotations for YT000041–YT000100
ANNOTATIONS = {
    "YT000041": ("POSITIVE", "MEDIUM", "Expresses desire for a similar car"),
    "YT000042": ("POSITIVE", "HIGH", "Appreciates tutorial and educational value"),
    "YT000043": ("NEGATIVE", "HIGH", "Complains about unwanted subscription prompts"),
    "YT000044": ("POSITIVE", "HIGH", "Appreciates information"),
    "YT000045": ("NEUTRAL", "MEDIUM", "Points out something visible in the background"),
    "YT000046": ("POSITIVE", "HIGH", "Enthusiasm for BMW"),
    "YT000047": ("CRITICAL", "HIGH", "Challenges interpretation of gaming performance"),
    "YT000048": ("CRITICAL", "HIGH", "Criticizes policy and proposes a different approach"),
    "YT000049": ("POSITIVE", "HIGH", "Expresses appreciation for a board game"),
    "YT000050": ("CRITICAL", "HIGH", "Challenges an argument about China's decline"),
    "YT000051": ("QUESTION", "HIGH", "Asks about other conflicts and war crimes"),
    "YT000052": ("CRITICAL", "HIGH", "Identifies outdated sections in a tutorial"),
    "YT000053": ("NEUTRAL", "HIGH", "Observational statement predicting increased public transport use during economic hardship"),
    "YT000054": ("POSITIVE", "HIGH", "Expresses anticipation"),
    "YT000055": ("POSITIVE", "MEDIUM", "Agreement and relatability"),
    "YT000056": ("NEGATIVE", "HIGH", "Strongly criticizes Germany; assess identity-related safety separately"),
    "YT000057": ("POSITIVE", "HIGH", "Expresses support and goodwill"),
    "YT000058": ("POSITIVE", "HIGH", "Expresses enthusiasm for a speech"),
    "YT000059": ("QUESTION", "HIGH", "Asks whether workers are inside the box"),
    "YT000060": ("POSITIVE", "HIGH", "Expresses aspiration"),
    "YT000061": ("NEGATIVE", "HIGH", "Expresses distrust and disapproval of politicians"),
    "YT000062": ("POSITIVE", "HIGH", "Compliments someone"),
    "YT000063": ("CRITICAL", "HIGH", "Challenges reasoning in an argument"),
    "YT000064": ("NEGATIVE", "MEDIUM", "Expresses an unfavorable opinion"),
    "YT000065": ("UNCLEAR", "LOW", "Single-word comment; insufficient context"),
    "YT000066": ("POSITIVE", "MEDIUM", "Encouragement"),
    "YT000067": ("POSITIVE", "HIGH", "Expresses affection and support"),
    "YT000068": ("CRITICAL", "HIGH", "Challenges a lack of empathy"),
    "YT000069": ("POSITIVE", "HIGH", "Appreciates educational content"),
    "YT000070": ("NEUTRAL", "MEDIUM", "Promotional channel message; assess spam separately"),
    "YT000071": ("QUESTION", "HIGH", "Asks what the show is"),
    "YT000072": ("QUESTION", "MEDIUM", "Expresses confusion and implicitly seeks clarification"),
    "YT000073": ("NEGATIVE", "MEDIUM", "Dismissive expression"),
    "YT000074": ("QUESTION", "HIGH", "Asks a health-related question"),
    "YT000075": ("POSITIVE", "MEDIUM", "Expresses support for Canadian assistance"),
    "YT000076": ("NEGATIVE", "HIGH", "Impatient and dismissive demand"),
    "YT000077": ("POSITIVE", "HIGH", "Agrees with educational advice"),
    "YT000078": ("POSITIVE", "HIGH", "Appreciates a car"),
    "YT000079": ("CRITICAL", "MEDIUM", "Suggests an explanatory structure to help others"),
    "YT000080": ("NEGATIVE", "MEDIUM", "Expresses disapproval"),
    "YT000081": ("CRITICAL", "HIGH", "Criticizes the economics of the military industry"),
    "YT000082": ("POSITIVE", "MEDIUM", "Expresses recognition and support"),
    "YT000083": ("POSITIVE", "HIGH", "Expresses appreciation"),
    "YT000084": ("NEGATIVE", "MEDIUM", "Dismissive reaction"),
    "YT000085": ("POSITIVE", "HIGH", "Shows empathy through shared experience"),
    "YT000086": ("NEUTRAL", "HIGH", "Technical observations and speculation"),
    "YT000087": ("NEUTRAL", "MEDIUM", "Shares a quotation without clear evaluation"),
    "YT000088": ("POSITIVE", "HIGH", "Appreciates a tutorial"),
    "YT000089": ("CRITICAL", "HIGH", "Suggests a technical improvement"),
    "YT000090": ("POSITIVE", "HIGH", "Appreciates a recommended book"),
    "YT000091": ("QUESTION", "HIGH", "Asks a technical question after expressing thanks"),
    "YT000092": ("NEUTRAL", "MEDIUM", "General observation about relationships"),
    "YT000093": ("CRITICAL", "MEDIUM", "Makes a political accusation about opponents"),
    "YT000094": ("NEUTRAL", "MEDIUM", "Hypothetical humorous observation"),
    "YT000095": ("UNCLEAR", "LOW", "Short Hindi response; context insufficient"),
    "YT000096": ("POSITIVE", "HIGH", "Expresses gratitude and confirms success"),
    "YT000097": ("QUESTION", "HIGH", "Asks multiple technical questions"),
    "YT000098": ("NEUTRAL", "HIGH", "Shares a personal memory"),
    "YT000099": ("NEGATIVE", "HIGH", "Expresses dislike and rejection"),
    "YT000100": ("NEGATIVE", "HIGH", "Expresses strong disapproval"),
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype={"comment_id": "string"})

    required = {"comment_id", "intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs found in the dataset.")

    # Confirm every target ID exists.
    available_ids = set(df["comment_id"].dropna())
    missing_ids = set(ANNOTATIONS) - available_ids
    if missing_ids:
        raise ValueError(f"These annotation IDs are missing: {sorted(missing_ids)}")

    # Do not overwrite annotations that already contain values.
    existing = df.set_index("comment_id").loc[list(ANNOTATIONS), "intent"]
    conflicts = existing.fillna("").astype("string").str.strip()
    conflicts = conflicts[conflicts != ""]

    if not conflicts.empty:
        raise ValueError(
            "Some comments already have intent labels. "
            f"Resolve manually before saving: {conflicts.to_dict()}"
        )

    # Add metadata columns if they do not exist.
    for column in ["intent_confidence", "annotation_note", "annotator"]:
        if column not in df.columns:
            df[column] = pd.Series("", index=df.index, dtype="string")
        else:
            df[column] = df[column].astype("string").fillna("")

    # Apply approved annotations.
    for comment_id, (intent, confidence, note) in ANNOTATIONS.items():
        mask = df["comment_id"] == comment_id
        df.loc[mask, "intent"] = intent
        df.loc[mask, "intent_confidence"] = confidence
        df.loc[mask, "annotation_note"] = note
        df.loc[mask, "annotator"] = ANNOTATOR

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # Read back and verify saved records.
    saved = pd.read_csv(CSV_PATH, dtype={"comment_id": "string"})
    batch = saved[saved["comment_id"].isin(ANNOTATIONS)]

    if len(batch) != len(ANNOTATIONS):
        raise RuntimeError("Verification failed: unexpected number of saved records.")

    if batch["intent"].isna().any() or batch["intent"].eq("").any():
        raise RuntimeError("Verification failed: some intent labels are empty.")

    print("\nBATCH 3 SAVED SUCCESSFULLY")
    print("-" * 45)
    print(f"Annotations saved: {len(batch)}")
    print("\nIntent distribution:")
    print(batch["intent"].value_counts().to_string())
    print("\nConfidence distribution:")
    print(batch["intent_confidence"].value_counts().to_string())
    print("\nCorrected annotation:")
    print(
        batch.loc[
            batch["comment_id"] == "YT000053",
            ["comment_id", "intent", "intent_confidence"]
        ].to_string(index=False)
    )
    print("\nVerification: PASSED")


if __name__ == "__main__":
    main()
