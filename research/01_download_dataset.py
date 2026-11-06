
import csv
from pathlib import Path
from datasets import load_dataset

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "raw" / "civil_comments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

COLUMNS = ["text"] + LABELS

for split in ["train", "validation", "test"]:

    print(f"\nDownloading {split} split...")

    dataset = load_dataset(
        "google/civil_comments",
        split=split,
        streaming=True
    )

    output_file = OUTPUT_DIR / f"{split}.csv"

    count = 0

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()

        for row in dataset:
            writer.writerow({
                column: row[column]
                for column in COLUMNS
            })

            count += 1

            if count % 100000 == 0:
                print(f"Processed {count:,} comments")

    print(f"Completed {split}: {count:,} comments")
    print(f"Saved to: {output_file}")

print("\nAll dataset splits downloaded successfully!")
