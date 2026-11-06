
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "datasets/engagement/raw/youtube_comment_sample.csv"
)

def main():

    df = pd.read_csv(INPUT_FILE)

    print("=" * 65)
    print("MODERAI - YOUTUBE SAMPLE QUALITY AUDIT")
    print("=" * 65)

    print("\n1. DATASET INFORMATION")
    print("Total records:", len(df))
    print("Columns:", list(df.columns))

    print("\n2. MISSING VALUES")
    print(df.isnull().sum())

    print("\n3. COMMENT LENGTH")

    lengths = df["comment_text"].fillna("").astype(str).str.len()

    print("Minimum:", lengths.min())
    print("Maximum:", lengths.max())
    print("Average:", round(lengths.mean(), 2))
    print("Comments under 10 characters:", (lengths < 10).sum())
    print("Comments over 1000 characters:", (lengths > 1000).sum())

    print("\n4. DUPLICATE COMMENTS")

    normalized = (
        df["comment_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    print("Duplicate normalized comments:", normalized.duplicated().sum())

    print("\n5. ORIGINAL SENTIMENT DISTRIBUTION")
    print(df["original_sentiment"].value_counts(dropna=False))

    print("\n6. SHORT COMMENT EXAMPLES")

    short_comments = df.assign(comment_length=lengths).sort_values(
        "comment_length"
    )

    print(
        short_comments[
            ["comment_text", "original_sentiment", "comment_length"]
        ].head(10).to_string(index=False)
    )

    print("\n7. LONG COMMENT EXAMPLES")

    print(
        short_comments[
            ["comment_text", "original_sentiment", "comment_length"]
        ].tail(5).to_string(index=False)
    )

    print("\nAUDIT COMPLETED")

if __name__ == "__main__":
    main()
