
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_FILE = PROJECT_ROOT / "datasets/engagement/raw/youtube_annotation_batch_6.csv"
MASTER_FILE = PROJECT_ROOT / "datasets/engagement/raw/youtube_annotation_queue.csv"

ANNOTATOR = "human_approved_assisted"

# Final 20 annotations: YT000381–YT000400
ANNOTATIONS = {
    "YT000381": ("UNCLEAR", "LOW", "Fragmented wording; insufficient context to identify the intent."),
    "YT000382": ("NEGATIVE", "HIGH", "Disapproves of a politician and disputes whom she represents."),
    "YT000383": ("CRITICAL", "MEDIUM", "Warns about clerks during ticket cash-ins; expresses dissatisfaction with their conduct."),
    "YT000384": ("POSITIVE", "HIGH", "Playful laughter and enjoyment of the scene."),
    "YT000385": ("POSITIVE", "HIGH", "Explicit praise and thanks for a helpful tutorial."),
    "YT000386": ("POSITIVE", "HIGH", "Expresses affection and respect for the creator."),
    "YT000387": ("POSITIVE", "HIGH", "Thanks the creator for the video."),
    "YT000388": ("POSITIVE", "HIGH", "Praises the rescue and the bond between the person and animal."),
    "YT000389": ("POSITIVE", "MEDIUM", "Expresses agreement with the person discussed in the video."),
    "YT000390": ("UNCLEAR", "LOW", "Tagalog text is difficult to interpret reliably without more context."),
    "YT000391": ("NEUTRAL", "MEDIUM", "Comments on a historical military tactic without a clear request or direct video criticism."),
    "YT000392": ("NEGATIVE", "HIGH", "Strongly insults and criticizes a public figure."),
    "YT000393": ("UNCLEAR", "LOW", "Informal, fragmented Tagalog; the intended point is not sufficiently clear."),
    "YT000394": ("CRITICAL", "HIGH", "Criticizes war profiteering and the financial incentives behind conflict."),
    "YT000395": ("CRITICAL", "HIGH", "Gives detailed criticism of AI limitations in responsive front-end development."),
    "YT000396": ("NEUTRAL", "HIGH", "Provides a factual location-related observation."),
    "YT000397": ("CRITICAL", "HIGH", "Criticizes materialism and concern about property value after a fire."),
    "YT000398": ("POSITIVE", "HIGH", "Brief positive evaluation of the movie."),
    "YT000399": ("CRITICAL", "HIGH", "Complains about the video's 720p resolution."),
    "YT000400": ("NEUTRAL", "HIGH", "Acknowledges the person without expressing a clear opinion or request."),
}

EXPECTED_IDS = [f"YT{i:06d}" for i in range(381, 401)]
VALID_INTENTS = {
    "POSITIVE", "NEGATIVE", "CRITICAL",
    "QUESTION", "NEUTRAL", "UNCLEAR"
}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def backup_file(path, timestamp):
    backup_path = path.with_name(
        f"{path.stem}_backup_{timestamp}{path.suffix}"
    )
    shutil.copy2(path, backup_path)
    print(f"Backup created: {backup_path.name}")


def save_csv(df, path):
    temp_path = path.with_name(path.stem + "_temporary" + path.suffix)
    df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)


def apply_annotations(df, id_column="comment_id"):
    required = {id_column}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required column: {id_column}")

    for column in [
        "intent", "annotation_confidence",
        "annotation_notes", "annotator"
    ]:
        if column not in df.columns:
            df[column] = ""

    df[id_column] = df[id_column].astype("string").fillna("")
    matching = df[id_column].isin(ANNOTATIONS)

    if int(matching.sum()) != 20:
        found = df.loc[matching, id_column].tolist()
        raise ValueError(
            f"Expected exactly 20 target rows; found {len(found)}. "
            f"Found IDs: {found}"
        )

    for index in df.index[matching]:
        comment_id = df.at[index, id_column]
        intent, confidence, note = ANNOTATIONS[comment_id]

        df.at[index, "intent"] = intent
        df.at[index, "annotation_confidence"] = confidence
        df.at[index, "annotation_notes"] = note
        df.at[index, "annotator"] = ANNOTATOR

    return df


def main():
    print("=" * 90)
    print("MODERAI — SAVE ANNOTATION BATCH 6 — GROUP 5")
    print("=" * 90)

    if set(ANNOTATIONS) != set(EXPECTED_IDS):
        raise ValueError("Annotation IDs do not match the expected Group 5 IDs.")

    for comment_id, (intent, confidence, _) in ANNOTATIONS.items():
        if intent not in VALID_INTENTS:
            raise ValueError(f"Invalid intent for {comment_id}: {intent}")
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(f"Invalid confidence for {comment_id}: {confidence}")

    batch_df = read_csv(BATCH_FILE)
    master_df = read_csv(MASTER_FILE)

    # Check IDs before changing either file.
    for name, df in [("Batch", batch_df), ("Master queue", master_df)]:
        if "comment_id" not in df.columns:
            raise ValueError(f"{name} file has no comment_id column.")

        duplicates = df["comment_id"][df["comment_id"].duplicated()].tolist()
        if duplicates:
            raise ValueError(f"Duplicate IDs in {name}: {duplicates}")

    batch_ids = set(batch_df["comment_id"])
    master_ids = set(master_df["comment_id"])

    missing_batch = set(EXPECTED_IDS) - batch_ids
    if missing_batch:
        raise ValueError(f"Group 5 IDs missing from batch file: {sorted(missing_batch)}")

    already_in_master = set(EXPECTED_IDS) & master_ids
    if already_in_master:
        raise ValueError(
            "Some Group 5 IDs already exist in master queue: "
            f"{sorted(already_in_master)}"
        )

    # Apply labels to the batch file.
    batch_df = apply_annotations(batch_df)

    # Extract the final 20 rows for appending to the master queue.
    new_rows = batch_df[batch_df["comment_id"].isin(EXPECTED_IDS)].copy()

    # Ensure both files have the same columns before concatenation.
    for column in new_rows.columns:
        if column not in master_df.columns:
            master_df[column] = ""

    for column in master_df.columns:
        if column not in new_rows.columns:
            new_rows[column] = ""

    new_rows = new_rows[master_df.columns]
    updated_master = pd.concat(
        [master_df, new_rows],
        ignore_index=True
    )

    # Final checks before writing.
    if len(new_rows) != 20:
        raise ValueError(f"Expected 20 new rows, got {len(new_rows)}.")

    if updated_master["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs detected after merge.")

    saved_rows = updated_master[
        updated_master["comment_id"].isin(EXPECTED_IDS)
    ]

    if len(saved_rows) != 20:
        raise ValueError("Final master queue verification failed.")

    if saved_rows["intent"].eq("").any():
        raise ValueError("Some new rows have blank intent labels.")

    # Back up both originals before writing.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file(BATCH_FILE, timestamp)
    backup_file(MASTER_FILE, timestamp)

    save_csv(batch_df, BATCH_FILE)
    save_csv(updated_master, MASTER_FILE)

    # Re-read saved files and verify persisted data.
    verified_batch = read_csv(BATCH_FILE)
    verified_master = read_csv(MASTER_FILE)

    verified_rows = verified_master[
        verified_master["comment_id"].isin(EXPECTED_IDS)
    ]

    if len(verified_rows) != 20:
        raise RuntimeError("Saved master queue does not contain all 20 new rows.")

    if set(verified_rows["comment_id"]) != set(EXPECTED_IDS):
        raise RuntimeError("Saved master queue IDs do not match Group 5.")

    if not verified_rows["annotator"].eq(ANNOTATOR).all():
        raise RuntimeError("Annotator verification failed.")

    print("\nGROUP 5 SAVE SUMMARY")
    print("-" * 90)
    print(f"Annotations prepared:       {len(ANNOTATIONS)}")
    print(f"Batch file rows:             {len(verified_batch)}")
    print(f"Master queue rows:           {len(verified_master)}")
    print(f"New real-world comments:     {len(verified_rows)}")
    print(f"Annotator:                   {ANNOTATOR}")

    print("\nGroup 5 intent distribution:")
    print(verified_rows["intent"].value_counts().to_string())

    print("\nGroup 5 confidence distribution:")
    print(verified_rows["annotation_confidence"].value_counts().to_string())

    print("\nVerification: PASSED")
    print("Batch 6 Group 5 successfully saved.")
    print("Master queue updated from 380 to 400 real-world comments.")


if __name__ == "__main__":
    main()
