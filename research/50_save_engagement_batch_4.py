
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

# Approved annotations: YT000101–YT000200
# Format: (intent, confidence, annotation_note)

ANNOTATIONS = {
    "YT000101": ("NEGATIVE", "HIGH", "Sarcastic criticism of administration spending"),
    "YT000102": ("POSITIVE", "HIGH", "Praises clear and useful content"),
    "YT000103": ("POSITIVE", "HIGH", "Expresses affection and appreciation"),
    "YT000104": ("QUESTION", "MEDIUM", "Questions the stated starting year"),
    "YT000105": ("CRITICAL", "HIGH", "Identifies poor instructions and explains the problem"),
    "YT000106": ("CRITICAL", "HIGH", "Criticizes aggressive behavior and expresses concern"),
    "YT000107": ("QUESTION", "HIGH", "Asks why sand is not used as an extinguisher"),
    "YT000108": ("POSITIVE", "HIGH", "Thanks the creator for new videos"),
    "YT000109": ("NEUTRAL", "HIGH", "Observation about an actor"),
    "YT000110": ("POSITIVE", "MEDIUM", "Expresses amusement through laughter"),
    "YT000111": ("UNCLEAR", "LOW", "Foreign-language expression; meaning uncertain"),
    "YT000112": ("NEGATIVE", "HIGH", "Expresses strong disapproval of Russia's actions"),
    "YT000113": ("POSITIVE", "MEDIUM", "Enthusiastic reaction"),
    "YT000114": ("CRITICAL", "HIGH", "Identifies lack of solutions and criticizes the discussion"),
    "YT000115": ("QUESTION", "MEDIUM", "Expresses confusion and implicitly seeks clarification"),
    "YT000116": ("NEUTRAL", "MEDIUM", "Factual observation about appearances"),
    "YT000117": ("NEUTRAL", "MEDIUM", "General statement about human behavior"),
    "YT000118": ("NEGATIVE", "HIGH", "Criticizes declining channel performance"),
    "YT000119": ("CRITICAL", "HIGH", "Challenges an interpretation and provides reasoning"),
    "YT000120": ("NEUTRAL", "MEDIUM", "Humorous hypothetical observation"),
    "YT000121": ("POSITIVE", "MEDIUM", "Expresses respect for physics"),
    "YT000122": ("POSITIVE", "MEDIUM", "Shows enthusiasm about saving learning resources"),
    "YT000123": ("NEUTRAL", "HIGH", "States lack of familiarity with songs"),
    "YT000124": ("QUESTION", "MEDIUM", "Rhetorical question about a group of people"),
    "YT000125": ("NEUTRAL", "HIGH", "Shares historical tax information"),
    "YT000126": ("NEUTRAL", "MEDIUM", "Observes someone's emotional state"),
    "YT000127": ("POSITIVE", "HIGH", "Expresses gratitude for an updated course"),
    "YT000128": ("QUESTION", "MEDIUM", "Playful request about getting a day off"),
    "YT000129": ("NEUTRAL", "HIGH", "Identifies an activity"),
    "YT000130": ("CRITICAL", "HIGH", "Highlights perceived political inconsistency"),
    "YT000131": ("QUESTION", "MEDIUM", "Asks whether others noticed something"),
    "YT000132": ("NEUTRAL", "MEDIUM", "Shares personal entertainment preferences"),
    "YT000133": ("NEUTRAL", "HIGH", "Timestamp only"),
    "YT000134": ("NEUTRAL", "MEDIUM", "Shares slogans and statements without clear interaction intent"),
    "YT000135": ("CRITICAL", "HIGH", "Criticizes lack of preparedness for recurring fires"),
    "YT000136": ("NEGATIVE", "HIGH", "Expresses dislike toward a public figure"),
    "YT000137": ("NEUTRAL", "MEDIUM", "Shares a humorous dialogue"),
    "YT000138": ("QUESTION", "LOW", "Appears to ask who is doing something in 2025; wording unclear"),
    "YT000139": ("POSITIVE", "MEDIUM", "Affectionate expression"),
    "YT000140": ("POSITIVE", "HIGH", "Expresses amusement"),
    "YT000141": ("NEUTRAL", "HIGH", "Provides technical instructions"),
    "YT000142": ("QUESTION", "MEDIUM", "Asks whether others noticed a visual effect"),
    "YT000143": ("QUESTION", "HIGH", "Requests help resolving an npm version issue"),
    "YT000144": ("QUESTION", "HIGH", "Asks for clarification about build and watch scripts"),
    "YT000145": ("CRITICAL", "MEDIUM", "Sarcastically criticizes reliance on previous programming mistakes"),
    "YT000146": ("NEGATIVE", "HIGH", "Expresses distress about fire destruction"),
    "YT000147": ("POSITIVE", "MEDIUM", "Expresses approval"),
    "YT000148": ("QUESTION", "HIGH", "Compliment followed by a genuine question"),
    "YT000149": ("POSITIVE", "HIGH", "Praises the explanation"),
    "YT000150": ("POSITIVE", "HIGH", "Thanks and compliments the creator"),
    "YT000151": ("NEGATIVE", "MEDIUM", "Expresses strong disagreement"),
    "YT000152": ("NEGATIVE", "MEDIUM", "Uses sarcasm to criticize a political figure"),
    "YT000153": ("NEGATIVE", "HIGH", "Complains about camera movement"),
    "YT000154": ("CRITICAL", "HIGH", "Criticizes hate directed at teenagers and advocates letting them be"),
    "YT000155": ("UNCLEAR", "LOW", "Political slogan with unclear intended meaning"),
    "YT000156": ("CRITICAL", "MEDIUM", "Makes serious accusations about surveillance and institutional involvement"),
    "YT000157": ("NEGATIVE", "HIGH", "Expresses frustration about broken earbuds"),
    "YT000158": ("POSITIVE", "HIGH", "Strong appreciation for the course"),
    "YT000159": ("NEUTRAL", "MEDIUM", "Shares a hypothetical observation"),
    "YT000160": ("POSITIVE", "HIGH", "Praises the conversation"),
    "YT000161": ("NEGATIVE", "HIGH", "Shares a hostile accusation; assess potential threat separately"),
    "YT000162": ("QUESTION", "HIGH", "Requests a name"),
    "YT000163": ("QUESTION", "HIGH", "Requests a second part and asks about ML training"),
    "YT000164": ("CRITICAL", "HIGH", "Criticizes political interference in personal matters"),
    "YT000165": ("POSITIVE", "HIGH", "Praises the content idea and creator"),
    "YT000166": ("QUESTION", "HIGH", "Asks for a Flask versus Django comparison"),
    "YT000167": ("CRITICAL", "HIGH", "Presents detailed criticism and argues for more national products"),
    "YT000168": ("NEGATIVE", "MEDIUM", "Expresses hostility toward Putin; assess threat separately"),
    "YT000169": ("QUESTION", "HIGH", "Asks whether someone was pardoned"),
    "YT000170": ("POSITIVE", "HIGH", "Praises the explanation"),
    "YT000171": ("POSITIVE", "HIGH", "Offers encouragement and supportive advice"),
    "YT000172": ("NEGATIVE", "HIGH", "Expresses anger and disapproval"),
    "YT000173": ("NEGATIVE", "MEDIUM", "Disapproving reaction expressed humorously"),
    "YT000174": ("POSITIVE", "MEDIUM", "Compliments someone's personality"),
    "YT000175": ("POSITIVE", "HIGH", "Praises improving animations"),
    "YT000176": ("CRITICAL", "HIGH", "Criticizes the treatment and stress experienced by an animal"),
    "YT000177": ("POSITIVE", "MEDIUM", "Humorous reaction"),
    "YT000178": ("POSITIVE", "HIGH", "Praises content and expresses support"),
    "YT000179": ("QUESTION", "HIGH", "Describes a technical issue and requests a solution"),
    "YT000180": ("POSITIVE", "HIGH", "Compliments the creator and shares a learning takeaway"),
    "YT000181": ("CRITICAL", "HIGH", "Identifies outdated courses and requests updates"),
    "YT000182": ("CRITICAL", "MEDIUM", "Criticizes mainstream media coverage"),
    "YT000183": ("NEGATIVE", "HIGH", "Expresses broad political disapproval"),
    "YT000184": ("CRITICAL", "HIGH", "Criticizes media outlets for alleged racist remarks"),
    "YT000185": ("QUESTION", "MEDIUM", "Questions how someone could manage a time difference"),
    "YT000186": ("QUESTION", "HIGH", "Asks about algorithmic time complexity"),
    "YT000187": ("NEGATIVE", "HIGH", "Expresses disgust and disapproval"),
    "YT000188": ("QUESTION", "HIGH", "Thanks the creator, then asks a genuine spreadsheet-sharing question"),
    "YT000189": ("POSITIVE", "HIGH", "Praises the video as informative"),
    "YT000190": ("POSITIVE", "HIGH", "Praises the interview and interviewee"),
    "YT000191": ("POSITIVE", "HIGH", "Praises the advice"),
    "YT000192": ("CRITICAL", "HIGH", "Gives constructive feedback about audio synchronization"),
    "YT000193": ("NEGATIVE", "MEDIUM", "Expresses a negative reaction to gambling"),
    "YT000194": ("NEGATIVE", "HIGH", "Blames and criticizes Ukraine's president"),
    "YT000195": ("NEGATIVE", "HIGH", "Criticizes the US administration"),
    "YT000196": ("POSITIVE", "HIGH", "Praises the information"),
    "YT000197": ("NEGATIVE", "MEDIUM", "Insulting expression of disapproval"),
    "YT000198": ("NEGATIVE", "MEDIUM", "Makes a dismissive, aggressive humorous remark"),
    "YT000199": ("POSITIVE", "MEDIUM", "Praises an interview and speaker's argument"),
    "YT000200": ("NEGATIVE", "HIGH", "Expresses impatience about a delayed moment"),
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype={"comment_id": "string"})

    required_columns = {"comment_id", "intent"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs found.")

    if len(ANNOTATIONS) != 100:
        raise ValueError(
            f"Expected 100 annotations, found {len(ANNOTATIONS)}."
        )

    available_ids = set(df["comment_id"].dropna())
    missing_ids = set(ANNOTATIONS) - available_ids

    if missing_ids:
        raise ValueError(f"Missing comment IDs: {sorted(missing_ids)}")

    # Prevent accidental overwriting of existing annotations.
    existing = (
        df.set_index("comment_id")
        .loc[list(ANNOTATIONS), "intent"]
        .fillna("")
        .astype("string")
        .str.strip()
    )

    conflicts = existing[existing != ""]

    if not conflicts.empty:
        raise ValueError(
            "Some comments already have intent labels: "
            f"{conflicts.to_dict()}"
        )

    # Use the existing canonical metadata columns.
    metadata_columns = [
        "annotation_confidence",
        "annotation_notes",
        "annotator",
    ]

    for column in metadata_columns:
        if column not in df.columns:
            df[column] = pd.Series("", index=df.index, dtype="string")
        else:
            df[column] = df[column].astype("string").fillna("")

    # Save the approved annotations.
    for comment_id, (intent, confidence, note) in ANNOTATIONS.items():
        mask = df["comment_id"] == comment_id

        df.loc[mask, "intent"] = intent
        df.loc[mask, "annotation_confidence"] = confidence
        df.loc[mask, "annotation_notes"] = note
        df.loc[mask, "annotator"] = ANNOTATOR

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # Reload and verify.
    saved = pd.read_csv(
        CSV_PATH,
        dtype={"comment_id": "string"}
    )

    batch = saved[
        saved["comment_id"].isin(ANNOTATIONS)
    ].copy()

    if len(batch) != 100:
        raise RuntimeError(
            f"Verification failed: expected 100 rows, found {len(batch)}."
        )

    if batch["intent"].isna().any():
        raise RuntimeError("Verification failed: missing intent labels.")

    if batch["intent"].eq("").any():
        raise RuntimeError("Verification failed: empty intent labels.")

    if batch["annotation_confidence"].isna().any():
        raise RuntimeError("Verification failed: missing confidence.")

    if batch["annotation_notes"].isna().any():
        raise RuntimeError("Verification failed: missing annotation notes.")

    print("\nMODERAI — BATCH 4 SAVED SUCCESSFULLY")
    print("=" * 50)
    print(f"Annotations saved: {len(batch)}")

    print("\nIntent distribution:")
    print(batch["intent"].value_counts().to_string())

    print("\nConfidence distribution:")
    print(batch["annotation_confidence"].value_counts().to_string())

    print("\nAnnotator distribution:")
    print(batch["annotator"].value_counts().to_string())

    print("\nVerification: PASSED")
    print(f"Updated file: {CSV_PATH}")


if __name__ == "__main__":
    main()
