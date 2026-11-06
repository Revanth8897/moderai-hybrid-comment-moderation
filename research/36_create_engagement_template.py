import pandas as pd
from pathlib import Path

output = Path("datasets/engagement/raw/engagement_comments.csv")

columns = [
    "comment_id",
    "text",
    "intent",
    "source",
    "annotator"
]

df = pd.DataFrame(columns=columns)

df.to_csv(output, index=False, encoding="utf-8")

print("Dataset template created:", output)
print("Columns:", list(df.columns))
print("Rows:", len(df))
