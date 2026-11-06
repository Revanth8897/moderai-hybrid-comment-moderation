
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "civil_comments"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "tfidf"

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

SAMPLE_SIZES = {
    "train": 200_000,
    "validation": 25_000,
    "test": 25_000,
}

def load_sample(split, sample_size):
    file_path = DATA_DIR / f"{split}.csv"

    df = pd.read_csv(
        file_path,
        usecols=["text"] + LABELS,
        nrows=sample_size
    )

    df = df.dropna(subset=["text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


print("Loading dataset samples...")

train_df = load_sample("train", SAMPLE_SIZES["train"])
val_df = load_sample("validation", SAMPLE_SIZES["validation"])
test_df = load_sample("test", SAMPLE_SIZES["test"])

print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Testing records:", len(test_df))


# Configure TF-IDF
vectorizer = TfidfVectorizer(
    max_features=100_000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32
)

print("\nFitting TF-IDF on training data...")

X_train = vectorizer.fit_transform(train_df["text"])

print("Transforming validation data...")
X_val = vectorizer.transform(val_df["text"])

print("Transforming test data...")
X_test = vectorizer.transform(test_df["text"])


# Save feature matrices
save_npz(OUTPUT_DIR / "X_train.npz", X_train)
save_npz(OUTPUT_DIR / "X_validation.npz", X_val)
save_npz(OUTPUT_DIR / "X_test.npz", X_test)

# Save the fitted vectorizer
joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.joblib")

# Save corresponding annotation scores
for split, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:
    df[LABELS].to_csv(
        OUTPUT_DIR / f"y_{split}.csv",
        index=False
    )


# Display feature information
print("\n" + "=" * 55)
print("TF-IDF FEATURE EXTRACTION REPORT")
print("=" * 55)

for name, matrix in [
    ("Training", X_train),
    ("Validation", X_val),
    ("Testing", X_test)
]:
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100

    print(f"\n{name}:")
    print("Matrix shape:", matrix.shape)
    print("Non-zero values:", matrix.nnz)
    print(f"Matrix density: {density:.4f}%")

print("\nVocabulary size:", len(vectorizer.vocabulary_))

print("\nTop vocabulary examples:")
print(list(vectorizer.vocabulary_.keys())[:20])

print("\nTF-IDF extraction completed!")
