
import hashlib
import sqlite3
import csv
import pandas as pd

from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "processed" / "civil_comments"

OUTPUT_DIR = ROOT / "datasets" / "processed" / "civil_comments_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = REPORT_DIR / "23_aggregation_temp.sqlite"

OUTPUT_FILE = OUTPUT_DIR / "all_unique_comments.csv"
TEMP_OUTPUT = OUTPUT_DIR / "all_unique_comments.tmp.csv"

FILES = [
    DATA_DIR / "train.csv",
    DATA_DIR / "validation.csv",
    DATA_DIR / "test.csv"
]

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

CHUNK_SIZE = 25_000


# --------------------------------------------------
# HASH FUNCTION
# --------------------------------------------------

def hash_comment(text):

    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=32
    ).digest()


# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------

def create_database(connection):

    label_columns = ", ".join(
        f"sum_{label} REAL NOT NULL"
        for label in LABELS
    )

    connection.execute(f"""
        CREATE TABLE comments (

            text_hash BLOB PRIMARY KEY,

            text TEXT NOT NULL,

            occurrence_count INTEGER NOT NULL,

            {label_columns}
        )
    """)


# --------------------------------------------------
# AGGREGATION
# --------------------------------------------------

def aggregate_comments(connection):

    sum_columns = ", ".join(
        f"sum_{label}"
        for label in LABELS
    )

    insert_columns = ", ".join([
        "text_hash",
        "text",
        "occurrence_count",
        sum_columns
    ])

    placeholders = ", ".join(
        ["?"] * (3 + len(LABELS))
    )

    update_columns = [
        "occurrence_count = occurrence_count + 1"
    ]

    for label in LABELS:

        update_columns.append(
            f"sum_{label} = sum_{label} + excluded.sum_{label}"
        )

    update_clause = ", ".join(update_columns)

    insert_sql = f"""
        INSERT INTO comments (
            {insert_columns}
        )

        VALUES ({placeholders})

        ON CONFLICT(text_hash) DO UPDATE SET

        {update_clause}
    """

    total_rows = 0
    skipped_rows = 0

    for file_path in FILES:

        print(f"\nReading: {file_path.name}")

        reader = pd.read_csv(
            file_path,
            usecols=["text"] + LABELS,
            dtype={"text": "string"},
            chunksize=CHUNK_SIZE
        )

        for chunk in reader:

            records = []

            for row in chunk.itertuples(index=False, name=None):

                raw_text = row[0]

                if pd.isna(raw_text) or str(raw_text) == "":
                    skipped_rows += 1
                    continue

                text = str(raw_text)

                scores = []

                for value in row[1:]:

                    if pd.isna(value):
                        raise ValueError(
                            "Missing annotation score found."
                        )

                    score = float(value)

                    if not 0.0 <= score <= 1.0:
                        raise ValueError(
                            f"Invalid annotation score: {score}"
                        )

                    scores.append(score)

                record = [
                    hash_comment(text),
                    text,
                    1
                ] + scores

                records.append(tuple(record))

            connection.executemany(
                insert_sql,
                records
            )

            connection.commit()

            total_rows += len(chunk)

            print(
                f"Processed: {total_rows:,} rows",
                end="\r"
            )

        print(f"\nCompleted: {file_path.name}")

    return total_rows, skipped_rows


# --------------------------------------------------
# EXPORT UNIQUE COMMENTS
# --------------------------------------------------

def export_unique_comments(connection):

    print("\nExporting aggregated comments...")

    cursor = connection.execute("""
        SELECT
            text,
            occurrence_count,
            sum_toxicity,
            sum_severe_toxicity,
            sum_obscene,
            sum_threat,
            sum_insult,
            sum_identity_attack,
            sum_sexual_explicit

        FROM comments

        ORDER BY text_hash
    """)

    header = [
        "text",
        "occurrence_count"
    ] + LABELS

    exported_rows = 0

    with open(
        TEMP_OUTPUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        while True:

            rows = cursor.fetchmany(5000)

            if not rows:
                break

            output_rows = []

            for row in rows:

                text = row[0]
                occurrence_count = row[1]

                sums = row[2:]

                mean_scores = [
                    score / occurrence_count
                    for score in sums
                ]

                output_rows.append(
                    [text, occurrence_count] + mean_scores
                )

            writer.writerows(output_rows)

            exported_rows += len(rows)

            print(
                f"Exported: {exported_rows:,} unique comments",
                end="\r"
            )

    TEMP_OUTPUT.replace(OUTPUT_FILE)

    print("\nExport completed.")

    return exported_rows


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if DB_FILE.exists():
        DB_FILE.unlink()

    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()

    connection = sqlite3.connect(DB_FILE)

    try:

        print("=" * 65)
        print("MODERAI DUPLICATE COMMENT AGGREGATION")
        print("=" * 65)

        create_database(connection)

        total_rows, skipped_rows = aggregate_comments(
            connection
        )

        unique_rows = export_unique_comments(
            connection
        )

        duplicate_rows = total_rows - skipped_rows - unique_rows

        print("\n" + "=" * 65)
        print("AGGREGATION SUMMARY")
        print("=" * 65)

        print("Original rows:", f"{total_rows:,}")
        print("Skipped rows:", f"{skipped_rows:,}")
        print("Unique comments:", f"{unique_rows:,}")
        print("Duplicate rows consolidated:", f"{duplicate_rows:,}")

        print("\nOutput:", OUTPUT_FILE)

        print("\nOriginal datasets were not modified.")

    finally:

        connection.close()

        if DB_FILE.exists():
            DB_FILE.unlink()

        print("\nTemporary database cleaned up.")


if __name__ == "__main__":
    main()
