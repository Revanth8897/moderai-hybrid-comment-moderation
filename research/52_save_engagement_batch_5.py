
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

# Approved annotations: YT000201–YT000300
# Format: (intent, confidence, annotation_note)

ANNOTATIONS = {
    "YT000201": ("POSITIVE", "HIGH", "Compliments the creator warmly"),
    "YT000202": ("POSITIVE", "HIGH", "Expresses support and solidarity from an OFW"),
    "YT000203": ("POSITIVE", "HIGH", "Praises teaching style and effectiveness"),
    "YT000204": ("NEGATIVE", "HIGH", "Derogatory political labeling and insult"),
    "YT000205": ("POSITIVE", "HIGH", "Expresses appreciation and gratitude to instructor"),
    "YT000206": ("NEGATIVE", "HIGH", "Hostile comment advocating insults and throwing objects"),
    "YT000207": ("CRITICAL", "HIGH", "Critical policy discussion criticizing gun access and legislation"),
    "YT000208": ("POSITIVE", "MEDIUM", "Emotional reaction admiring someone's voice"),
    "YT000209": ("POSITIVE", "HIGH", "Praises clear and useful content"),
    "YT000210": ("POSITIVE", "HIGH", "Enthusiastic expression about starting the course"),
    "YT000211": ("UNCLEAR", "LOW", "Single-digit comment; insufficient context"),
    "YT000212": ("NEGATIVE", "HIGH", "Dismissive political insult accusing others of Trump Derangement Syndrome"),
    "YT000213": ("POSITIVE", "HIGH", "Praises performance with enthusiasm"),
    "YT000214": ("POSITIVE", "HIGH", "Expresses international solidarity and support for Ukraine"),
    "YT000215": ("CRITICAL", "MEDIUM", "Offers unsolicited opinion on learning coding through college"),
    "YT000216": ("POSITIVE", "MEDIUM", "Expresses nostalgia and fondness for a wrestler"),
    "YT000217": ("POSITIVE", "HIGH", "Expresses gratitude for educational knowledge"),
    "YT000218": ("CRITICAL", "HIGH", "Strong critical opinion on an athlete's reputation"),
    "YT000219": ("NEUTRAL", "HIGH", "Informative observation on building design and firefighting physics"),
    "YT000220": ("POSITIVE", "MEDIUM", "Emoji reaction expressing sorrow and prayerful sympathy"),
    "YT000221": ("NEUTRAL", "HIGH", "Shares personal web development experience and tech stack"),
    "YT000222": ("NEGATIVE", "MEDIUM", "Satirical joke mocking political claims about ending the war in Ukraine"),
    "YT000223": ("NEGATIVE", "HIGH", "Expresses grief over occupation and hostility toward occupying forces"),
    "YT000224": ("POSITIVE", "MEDIUM", "Expresses emotional attachment to BMW"),
    "YT000225": ("POSITIVE", "HIGH", "Amused reaction to a humorous altercation"),
    "YT000226": ("NEGATIVE", "HIGH", "Cringing disapproval of an interview response"),
    "YT000227": ("QUESTION", "HIGH", "Requests a future course on PostgreSQL"),
    "YT000228": ("POSITIVE", "MEDIUM", "Humorous reaction regarding a popular streamer"),
    "YT000229": ("NEGATIVE", "HIGH", "Advocates nuclear war; send for separate safety review"),
    "YT000230": ("CRITICAL", "HIGH", "Challenges classifying discipline as a downside"),
    "YT000231": ("POSITIVE", "HIGH", "High praise and appreciation for creator's dedication"),
    "YT000232": ("NEGATIVE", "HIGH", "Insult criticizing someone as rude and ill-mannered"),
    "YT000233": ("NEGATIVE", "HIGH", "Harsh criticism of a politician's demeanor and competence"),
    "YT000234": ("POSITIVE", "MEDIUM", "Playful, humorous complaint about an older phone model"),
    "YT000235": ("NEUTRAL", "MEDIUM", "Humorous observation about a common programming mindset"),
    "YT000236": ("POSITIVE", "HIGH", "Expresses gratitude for troubleshooting assistance"),
    "YT000237": ("POSITIVE", "HIGH", "Compliments creator by comparing them to Bob Ross"),
    "YT000238": ("CRITICAL", "MEDIUM", "Argues for legal action against offensive speech and supports INC"),
    "YT000239": ("CRITICAL", "HIGH", "Raises suspicions of exploitation or abuse and urges an investigation"),
    "YT000240": ("NEGATIVE", "HIGH", "Expresses strong disapproval and moral disgust toward political associates"),
    "YT000241": ("NEGATIVE", "HIGH", "Cynical accusation claiming the WHO spreads lies"),
    "YT000242": ("CRITICAL", "HIGH", "Criticizes infrastructure failures and demands criminal accountability for officials"),
    "YT000243": ("QUESTION", "HIGH", "Compliments video and asks a technical question about array destructuring"),
    "YT000244": ("POSITIVE", "HIGH", "Heart emoji expressing affection"),
    "YT000245": ("NEGATIVE", "MEDIUM", "Sarcastic political commentary mocking the presidential campaign"),
    "YT000246": ("NEUTRAL", "HIGH", "Factual personal observation noting ownership of the same purse"),
    "YT000247": ("POSITIVE", "HIGH", "Playful compliment praising a kid's agility in virtual gaming context"),
    "YT000248": ("CRITICAL", "HIGH", "Criticizes press hypocrisy regarding mental health and honors deceased"),
    "YT000249": ("CRITICAL", "HIGH", "Offers blunt technical advice and criticizes specific sharpening tools"),
    "YT000250": ("POSITIVE", "HIGH", "Celebratory birthday greeting"),
    "YT000251": ("NEGATIVE", "MEDIUM", "Vulgar dismissal of AI technology and state television"),
    "YT000252": ("CRITICAL", "HIGH", "Offers constructive curriculum recommendations for AI/ML roadmaps"),
    "YT000253": ("NEUTRAL", "HIGH", "Provides technical explanation and code example"),
    "YT000254": ("POSITIVE", "MEDIUM", "Expresses enthusiasm for a cryptocurrency project"),
    "YT000255": ("UNCLEAR", "LOW", "List of names without clear interaction intent"),
    "YT000256": ("NEUTRAL", "MEDIUM", "Observes political news with curiosity and bewilderment"),
    "YT000257": ("POSITIVE", "HIGH", "Praises creator's voice and laugh"),
    "YT000258": ("POSITIVE", "HIGH", "Expresses encouragement and solidarity with Ukraine"),
    "YT000259": ("NEGATIVE", "HIGH", "Hostile accusation claiming creator is lying and spreading fake news"),
    "YT000260": ("POSITIVE", "HIGH", "Praises advice as a good strategy"),
    "YT000261": ("QUESTION", "HIGH", "Asks if the tutorial series is beginner-friendly for Node.js"),
    "YT000262": ("POSITIVE", "MEDIUM", "Humorous quote dramatizing a programming workflow"),
    "YT000263": ("NEGATIVE", "HIGH", "Derogatory insult criticizing a politician's appearance and voice"),
    "YT000264": ("POSITIVE", "HIGH", "Expresses strong agreement and resonance with a quote"),
    "YT000265": ("POSITIVE", "MEDIUM", "Enthusiastic celebratory message regarding spiritual or personal growth"),
    "YT000266": ("NEUTRAL", "MEDIUM", "Neutral political observation regarding election outcome"),
    "YT000267": ("NEGATIVE", "HIGH", "Derogatory personal and political criticism of public figures"),
    "YT000268": ("NEUTRAL", "MEDIUM", "Playful remark comparing video to a sitcom"),
    "YT000269": ("NEGATIVE", "HIGH", "Derogatory backronym dismissing CNN"),
    "YT000270": ("CRITICAL", "HIGH", "Political satire highlighting perceived partisan inconsistency"),
    "YT000271": ("POSITIVE", "HIGH", "Expresses affection for hometown"),
    "YT000272": ("POSITIVE", "HIGH", "Praises pacing, delivery, and instructor's voice"),
    "YT000273": ("POSITIVE", "HIGH", "Encourages creator and praises work"),
    "YT000274": ("UNCLEAR", "LOW", "Incoherent string; meaning uncertain"),
    "YT000275": ("NEGATIVE", "HIGH", "Pessimistic political warning predicting economic recession"),
    "YT000276": ("NEUTRAL", "MEDIUM", "Lighthearted personal remark"),
    "YT000277": ("QUESTION", "MEDIUM", "Playful question asking if anyone is watching in 2025"),
    "YT000278": ("POSITIVE", "HIGH", "Expresses heartfelt thanks to the instructor"),
    "YT000279": ("QUESTION", "HIGH", "Compliments communication skills and asks for virtual interview tips"),
    "YT000280": ("NEGATIVE", "HIGH", "Violent language directed at a child; urgent separate safety review"),
    "YT000281": ("CRITICAL", "HIGH", "Detailed critique raising privacy and automated surveillance risks"),
    "YT000282": ("NEGATIVE", "HIGH", "Endorses insults against a public figure"),
    "YT000283": ("QUESTION", "MEDIUM", "Single-word question inquiring about gelatin"),
    "YT000284": ("NEGATIVE", "HIGH", "Criticizes poor listening skills with frustration"),
    "YT000285": ("POSITIVE", "HIGH", "Encouraging call to do good and show kindness"),
    "YT000286": ("NEGATIVE", "HIGH", "Dismissive criticism of someone's humor"),
    "YT000287": ("NEGATIVE", "MEDIUM", "Yawn emojis expressing boredom and disinterest"),
    "YT000288": ("QUESTION", "HIGH", "Requests technical troubleshooting assistance with component export"),
    "YT000289": ("POSITIVE", "MEDIUM", "Expresses political support for Russia and Putin"),
    "YT000290": ("POSITIVE", "HIGH", "Praises instructor for clear explanations of fundamentals"),
    "YT000291": ("QUESTION", "HIGH", "Thanks creator, notes subscription, and asks if they built a similar Slack bot"),
    "YT000292": ("QUESTION", "MEDIUM", "Informal question using Scottish slang asking for tour dates"),
    "YT000293": ("NEUTRAL", "HIGH", "Timestamp reference only"),
    "YT000294": ("CRITICAL", "HIGH", "Criticizes hypocritical behavior during physical confrontations"),
    "YT000295": ("POSITIVE", "HIGH", "Compliments creator's consistent quality"),
    "YT000296": ("NEUTRAL", "MEDIUM", "Humorous gaming reference connecting Dragon Man to Skyrim"),
    "YT000297": ("POSITIVE", "HIGH", "Praises videos enthusiastically and encourages more educational content"),
    "YT000298": ("NEUTRAL", "MEDIUM", "Playful observation about feeding a pet"),
    "YT000299": ("NEGATIVE", "HIGH", "Political declaration assigning blame to Hamas"),
    "YT000300": ("QUESTION", "MEDIUM", "Playful question asking if someone is Dexter's sister"),
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

    print("\nMODERAI — BATCH 5 SAVED SUCCESSFULLY")
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
