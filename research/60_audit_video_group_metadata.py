
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "Original YouTube data": (
        ROOT / "datasets" / "engagement" / "raw"
        / "youtube_annotation_queue.csv"
    ),
    "Combined dataset": (
        ROOT / "datasets" / "engagement" / "processed"
        / "engagement_all_330.csv"
    ),
}


def normalize_column(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
    )


def inspect_file(label, path):
    print(f"\n{'=' * 65}")
    print(label)
    print("=" * 65)

    if not path.exists():
        print("ERROR: File not found:", path)
        return

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("\nAvailable columns:")
    for column in df.columns:
        print(f"  - {column}")

    title_columns = [
        column for column in df.columns
        if normalize_column(column) in {
            "videotitle",
            "title",
        }
    ]

    print("\nPossible video-title columns:", title_columns)

    if not title_columns:
        print("No video-title column detected.")
        return

    for column in title_columns:
        values = df[column].astype("string").fillna("").str.strip()
        nonempty = values[values.ne("")]

        print(f"\nColumn: {column}")
        print("Non-empty rows:", len(nonempty))
        print("Unique non-empty titles:", nonempty.nunique())
        print("Missing/blank rows:", values.eq("").sum())

        print("\nMost common titles:")
        print(
            nonempty.value_counts().head(10).to_string()
        )


def main():
    print("MODERAI — VIDEO GROUP METADATA AUDIT")

    for label, path in FILES.items():
        inspect_file(label, path)

    print("\nAudit completed.")
    print("Do not create train/test splits until this output is reviewed.")


if __name__ == "__main__":
    main()
