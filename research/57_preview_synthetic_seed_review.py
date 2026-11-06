
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "engagement_comments.csv"
)

def main():
    df = pd.read_csv(CSV_PATH)

    required = {"comment_id", "text", "intent", "annotator"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("\nMODERAI — SYNTHETIC SEED REVIEW")
    print("=" * 75)

    for _, row in df.iterrows():
        print(f"\nID: {row['comment_id']}")
        print(f"ASSIGNED INTENT: {row['intent']}")
        print(f"COMMENT: {row['text']}")
        print(f"STATUS: {row['annotator']}")
        print("-" * 75)

    print("\nTOTAL EXAMPLES:", len(df))
    print("\nINTENT DISTRIBUTION:")
    print(df["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
