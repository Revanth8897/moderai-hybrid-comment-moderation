
import hashlib
import sqlite3
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "processed" / "civil_comments"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = REPORT_DIR / "22_conflict_audit_temp.sqlite"
OUTPUT_FILE = REPORT_DIR / "22_per_label_conflicts.csv"

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

CHUNK_SIZE = 100_000


def hash_comment(text):
    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16
    ).digest()


def main():

    if DB_FILE.exists():
        DB_FILE.unlink()

    connection = sqlite3.connect(DB_FILE)

    try:

        # Build a table with minimum and maximum score
        # and binary label for every unique comment.

        columns = ["text_hash BLOB PRIMARY KEY", "occurrence_count INTEGER"]

        for label in LABELS:
            columns.extend([
                f"{label}_min REAL",
                f"{label}_max REAL",
                f"{label}_binary_min INTEGER",
                f"{label}_binary_max INTEGER"
            ])

        connection.execute(
            f"CREATE TABLE comments ({', '.join(columns)})"
        )

        insert_columns = ["text_hash", "occurrence_count"]

        for label in LABELS:
            insert_columns.extend([
                f"{label}_min",
                f"{label}_max",
                f"{label}_binary_min",
                f"{label}_binary_max"
            ])

        placeholders = ", ".join(["?"] * len(insert_columns))

        update_parts = ["occurrence_count = occurrence_count + 1"]

        for label in LABELS:

            update_parts.extend([
                f"{label}_min = MIN(comments.{label}_min, excluded.{label}_min)",
                f"{label}_max = MAX(comments.{label}_max, excluded.{label}_max)",
                f"{label}_binary_min = MIN(comments.{label}_binary_min, excluded.{label}_binary_min)",
                f"{label}_binary_max = MAX(comments.{label}_binary_max, excluded.{label}_binary_max)"
            ])

        insert_sql = f"""
            INSERT INTO comments ({', '.join(insert_columns)})
            VALUES ({placeholders})

            ON CONFLICT(text_hash) DO UPDATE SET
            {', '.join(update_parts)}
        """

        # Scan all original splits.

        for file_path in FILES:

            print(f"\nScanning: {file_path.name}")

            reader = pd.read_csv(
                file_path,
                usecols=["text"] + LABELS,
                chunksize=CHUNK_SIZE
            )

            processed = 0

            for chunk in reader:

                records = []

                for row in chunk.itertuples(index=False, name=None):

                    text = row[0]

                    if pd.isna(text) or str(text) == "":
                        continue

                    record = [
                        hash_comment(str(text)),
                        1
                    ]

                    for score in row[1:]:

                        score = float(score)
                        binary = int(score >= 0.5)

                        record.extend([
                            score,
                            score,
                            binary,
                            binary
                        ])

                    records.append(tuple(record))

                connection.executemany(insert_sql, records)
                connection.commit()

                processed += len(chunk)

                print(
                    f"Processed {processed:,} rows...",
                    end="\r"
                )

            print(f"\nCompleted: {file_path.name}")

        # Generate per-label results.

        results = []

        for label in LABELS:

            query = f"""
                SELECT
                    COUNT(*) AS duplicated_texts,

                    SUM(
                        CASE
                            WHEN {label}_min != {label}_max
                            THEN 1 ELSE 0
                        END
                    ) AS score_conflict_texts,

                    SUM(
                        CASE
                            WHEN {label}_binary_min != {label}_binary_max
                            THEN 1 ELSE 0
                        END
                    ) AS binary_conflict_texts

                FROM comments

                WHERE occurrence_count > 1
            """

            duplicated, score_conflicts, binary_conflicts = (
                connection.execute(query).fetchone()
            )

            results.append({
                "label": label,
                "duplicated_texts": duplicated or 0,
                "score_conflict_texts": score_conflicts or 0,
                "binary_conflict_texts": binary_conflicts or 0
            })

        summary = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("PER-LABEL DUPLICATE CONFLICT RESULTS")
        print("=" * 70)

        print(summary.to_string(index=False))

        summary.to_csv(OUTPUT_FILE, index=False)

        print("\nReport saved:", OUTPUT_FILE)

    finally:

        connection.close()

        if DB_FILE.exists():
            DB_FILE.unlink()

        print("\nTemporary database cleaned up.")


if __name__ == "__main__":
    main()
