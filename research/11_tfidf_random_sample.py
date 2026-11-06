
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "random_sample"
    / "train_sample.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "civil_comments"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "tfidf_random"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# Load training sample
print("Loading random training sample...")

train_df = pd.read_csv(TRAIN_FILE)

# Load complete validation and test splits
print("Loading complete validation split...")

val_df = pd.read_csv(
    PROCESSED_DIR / "validation.csv"
)

print("Loading complete test split...")

test_df = pd.read_csv(
    PROCESSED_DIR / "test.csv"
)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# Separate text and labels
train_text = train_df["text"].fillna("").astype(str)
val_text = val_df["text"].fillna("").astype(str)
test_text = test_df["text"].fillna("").astype(str)

y_train = train_df[LABELS]
y_val = val_df[LABELS]
y_test = test_df[LABELS]

# Configure TF-IDF
vectorizer = TfidfVectorizer(
    max_features=100_000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32,
)

# Fit only on training data
print("\nFitting TF-IDF on training sample...")

X_train = vectorizer.fit_transform(train_text)

print("Transforming validation data...")
X_val = vectorizer.transform(val_text)

print("Transforming test data...")
X_test = vectorizer.transform(test_text)

# Save feature matrices
print("\nSaving feature matrices...")

save_npz(OUTPUT_DIR / "X_train.npz", X_train)
save_npz(OUTPUT_DIR / "X_validation.npz", X_val)
save_npz(OUTPUT_DIR / "X_test.npz", X_test)

# Save labels
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_val.to_csv(OUTPUT_DIR / "y_validation.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# Save fitted vectorizer
joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer_random.joblib"
)

# Report feature information
print("\nTF-IDF FEATURE SUMMARY")
print("=" * 60)

print("Vocabulary size:", len(vectorizer.vocabulary_))

for name, matrix in [
    ("Training", X_train),
    ("Validation", X_val),
    ("Test", X_test),
]:
    density = matrix.nnz / (
        matrix.shape[0] * matrix.shape[1]
    )

    print(f"\n{name}:")
    print("Shape:", matrix.shape)
    print("Non-zero values:", matrix.nnz)
    print("Density:", round(density * 100, 4), "%")

print("\nTF-IDF extraction completed!")
print("Saved to:", OUTPUT_DIR)
