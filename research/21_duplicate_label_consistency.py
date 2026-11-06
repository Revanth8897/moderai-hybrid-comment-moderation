
import hashlib
import sqlite3
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "processed" / "civil_comments"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = REPORT_DIR / "21_duplicate_audit_temp.sqlite"
OUTPUT_FILE = REPORT_DIR / "21_duplicate_label_consistency.csv"

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv"
}

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


def make_signatures(values):

    scores = []
    binary = []

    for value in values:

        if pd.isna(value):
            scores.append("NULL")
            binary.append("NULL")
        else:
            score = float(value)

            scores.append(format(score, ".10g"))
            binary.append(str(int(score >= 0.5)))

    return "|".join(scores), "|".join(binary)


def main():

    if DB_FILE.exists():
        DB_FILE.unlink()

    connection = sqlite3.connect(DB_FILE)

    connection.execute("""
        CREATE TABLE comments (
            text_hash BLOB PRIMARY KEY,
            first_split TEXT,
            score_signature TEXT,
            binary_signature TEXT,
            occurrence_count INTEGER DEFAULT 1,
            score_conflicts INTEGER DEFAULT 0,
            binary_conflicts INTEGER DEFAULT 0,
            split_mask INTEGER DEFAULT 0
        )
    """)

    insert_sql = """
        INSERT INTO comments (
            text_hash,
            first_split,
            score_signature,
            binary_signature,
            occurrence_count,
            split_mask
        )
        VALUES (?, ?, ?, ?, 1, ?)

        ON CONFLICT(text_hash) DO UPDATE SET

            occurrence_count = occurrence_count + 1,

            score_conflicts = score_conflicts +
                CASE
                    WHEN score_signature != excluded.score_signature
                    THEN 1 ELSE 0
                END,

            binary_conflicts = binary_conflicts +
                CASE
                    WHEN binary_signature != excluded.binary_signature
                    THEN 1 ELSE 0
                END,

            split_mask = split_mask | excluded.split_mask
    """

    try:

        for split_index, (split_name, file_path) in enumerate(FILES.items()):

            print(f"\nScanning {split_name}: {file_path.name}")

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

                    text_hash = hash_comment(str(text))

                    score_signature, binary_signature = (
                        make_signatures(row[1:])
                    )

                    records.append((
                        text_hash,
                        split_name,
                        score_signature,
                        binary_signature,
                        1 << split_index
                    ))

                connection.executemany(insert_sql, records)
                connection.commit()

                processed += len(chunk)

                print(
                    f"Processed {processed:,} rows...",
                    end="\r"
                )

            print(f"\nCompleted {split_name}")

        # ------------------------------------------------
        # SUMMARY
        # ------------------------------------------------

        queries = {
            "unique_comment_texts": """
                SELECT COUNT(*) FROM comments
            """,

            "duplicated_texts": """
                SELECT COUNT(*)
                FROM comments
                WHERE occurrence_count > 1
            """,

            "extra_duplicate_rows": """
                SELECT COALESCE(SUM(occurrence_count - 1), 0)
                FROM comments
            """,

            "texts_with_score_conflicts": """
                SELECT COUNT(*)
                FROM comments
                WHERE score_conflicts > 0
            """,

            "rows_with_score_conflicts": """
                SELECT COALESCE(SUM(score_conflicts), 0)
                FROM comments
            """,

            "texts_with_binary_label_conflicts": """
                SELECT COUNT(*)
                FROM comments
                WHERE binary_conflicts > 0
            """,

            "rows_with_binary_label_conflicts": """
                SELECT COALESCE(SUM(binary_conflicts), 0)
                FROM comments
            """
        }

        results = []

        for metric, query in queries.items():

            value = connection.execute(query).fetchone()[0]

            results.append({
                "metric": metric,
                "value": value
            })

        summary = pd.DataFrame(results)

        print("\n" + "=" * 65)
        print("DUPLICATE LABEL CONSISTENCY RESULTS")
        print("=" * 65)

        print(summary.to_string(index=False))

        summary.to_csv(OUTPUT_FILE, index=False)

        print("\nReport saved:", OUTPUT_FILE)

        print("\nInterpretation:")
        print("- Score conflicts: annotation scores differ.")
        print("- Binary conflicts: labels differ at threshold >= 0.5.")
        print("- These are annotation disagreements, not necessarily errors.")

    finally:

        connection.close()

        if DB_FILE.exists():
            DB_FILE.unlink()

        print("\nTemporary database cleaned up.")


if __name__ == "__main__":
    main()
