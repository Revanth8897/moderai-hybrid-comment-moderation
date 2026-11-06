
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REAL_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "youtube_annotation_queue.csv"
)

SEED_PATH = (
    ROOT / "datasets" / "engagement" / "raw"
    / "engagement_comments.csv"
)

REPORT_DIR = ROOT / "reports" / "engagement"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_real_dataset():
    df = pd.read_csv(REAL_PATH)

    print("\n" + "=" * 65)
    print("REAL-WORLD ENGAGEMENT DATA ANALYSIS")
    print("=" * 65)

    # 1. Intent distribution
    intent_counts = df["intent"].value_counts()

    intent_summary = pd.DataFrame({
        "intent": intent_counts.index,
        "count": intent_counts.values,
        "percentage": (
            intent_counts.values / len(df) * 100
        ).round(2),
    })

    intent_summary.to_csv(
        REPORT_DIR / "real_intent_distribution.csv",
        index=False,
    )

    print("\n1. INTENT DISTRIBUTION")
    print(intent_summary.to_string(index=False))

    # 2. Confidence by intent
    confidence_table = pd.crosstab(
        df["intent"],
        df["annotation_confidence"],
        margins=True,
    )

    confidence_table.to_csv(
        REPORT_DIR / "confidence_by_intent.csv"
    )

    print("\n2. CONFIDENCE BY INTENT")
    print(confidence_table.to_string())

    # 3. Original sentiment vs MODERAI intent
    sentiment_table = pd.crosstab(
        df["original_sentiment"],
        df["intent"],
        margins=True,
    )

    sentiment_table.to_csv(
        REPORT_DIR / "original_sentiment_vs_intent.csv"
    )

    print("\n3. ORIGINAL SENTIMENT VS MODERAI INTENT")
    print(sentiment_table.to_string())

    # 4. Comment length analysis
    df["comment_length"] = (
        df["comment_text"].astype("string").str.len()
    )

    length_summary = df["comment_length"].describe()

    print("\n4. COMMENT LENGTH")
    print(length_summary.to_string())

    print(
        "\nComments under 10 characters:",
        (df["comment_length"] < 10).sum(),
    )

    print(
        "Comments over 500 characters:",
        (df["comment_length"] > 500).sum(),
    )

    # 5. Class imbalance ratio
    largest_class = intent_counts.max()
    smallest_class = intent_counts.min()

    imbalance_ratio = largest_class / smallest_class

    print("\n5. CLASS IMBALANCE")
    print("Largest class:", largest_class)
    print("Smallest class:", smallest_class)
    print("Imbalance ratio:", round(imbalance_ratio, 2))

    print("\nSaved reports:")
    print("- real_intent_distribution.csv")
    print("- confidence_by_intent.csv")
    print("- original_sentiment_vs_intent.csv")


def analyze_seed_dataset():
    df = pd.read_csv(SEED_PATH)

    print("\n" + "=" * 65)
    print("SYNTHETIC SEED DATASET ANALYSIS")
    print("=" * 65)

    print("Total synthetic examples:", len(df))
    print("\nIntent distribution:")
    print(df["intent"].value_counts().to_string())

    print("\nAnnotator status:")
    print(df["annotator"].value_counts().to_string())

    print(
        "\nNote: These examples are not yet approved "
        "for training."
    )


def main():
    analyze_real_dataset()
    analyze_seed_dataset()

    print("\n" + "=" * 65)
    print("DISTRIBUTION ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
