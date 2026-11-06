
import pandas as pd
from pathlib import Path

OUTPUT = Path(
    "datasets/engagement/raw/engagement_comments.csv"
)

examples = [
    # POSITIVE
    ("ENG000001", "Amazing work! Keep it up.", "POSITIVE"),
    ("ENG000002", "I absolutely love this product.", "POSITIVE"),
    ("ENG000003", "Thank you for sharing this information.", "POSITIVE"),
    ("ENG000004", "This is one of the best experiences I've had.", "POSITIVE"),
    ("ENG000005", "Really impressive work by the entire team.", "POSITIVE"),

    # NEGATIVE
    ("ENG000006", "I really don't like this product.", "NEGATIVE"),
    ("ENG000007", "This was a disappointing experience.", "NEGATIVE"),
    ("ENG000008", "Not worth the money I spent.", "NEGATIVE"),
    ("ENG000009", "I regret purchasing this.", "NEGATIVE"),
    ("ENG000010", "The experience was terrible.", "NEGATIVE"),

    # CRITICAL
    ("ENG000011", "The application needs a better user interface.", "CRITICAL"),
    ("ENG000012", "The product is missing several important features.", "CRITICAL"),
    ("ENG000013", "Delivery takes too long and needs improvement.", "CRITICAL"),
    ("ENG000014", "The documentation should include more practical examples.", "CRITICAL"),
    ("ENG000015", "The pricing structure could be more transparent.", "CRITICAL"),

    # QUESTION
    ("ENG000016", "How much does this product cost?", "QUESTION"),
    ("ENG000017", "Is this service available in Austria?", "QUESTION"),
    ("ENG000018", "Can you explain how this feature works?", "QUESTION"),
    ("ENG000019", "Where can I purchase this product?", "QUESTION"),
    ("ENG000020", "When will the next update be released?", "QUESTION"),

    # NEUTRAL
    ("ENG000021", "The product was released yesterday.", "NEUTRAL"),
    ("ENG000022", "I purchased this last week.", "NEUTRAL"),
    ("ENG000023", "The event starts at 10 AM tomorrow.", "NEUTRAL"),
    ("ENG000024", "This application supports three languages.", "NEUTRAL"),
    ("ENG000025", "The company opened a new office in Vienna.", "NEUTRAL"),

    # UNCLEAR
    ("ENG000026", "Hmm...", "UNCLEAR"),
    ("ENG000027", "Interesting.", "UNCLEAR"),
    ("ENG000028", "Okay then...", "UNCLEAR"),
    ("ENG000029", "Well, let's see.", "UNCLEAR"),
    ("ENG000030", "I see.", "UNCLEAR"),
]

if OUTPUT.exists():
    existing = pd.read_csv(OUTPUT)

    if len(existing) > 0:
        raise RuntimeError(
            "Dataset already contains data. "
            "Refusing to overwrite it."
        )

df = pd.DataFrame(
    examples,
    columns=["comment_id", "text", "intent"]
)

df["source"] = "synthetic_seed"
df["annotator"] = "pending_review"

df.to_csv(OUTPUT, index=False, encoding="utf-8")

print("=" * 55)
print("MODERAI ENGAGEMENT SEED DATASET")
print("=" * 55)

print("\nTotal comments:", len(df))

print("\nClass distribution:")
print(df["intent"].value_counts().sort_index())

print("\nSaved to:", OUTPUT)

print("\nDataset creation completed.")
