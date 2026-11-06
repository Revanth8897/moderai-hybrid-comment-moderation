
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "moderai_50k"
)

FEATURE_DIR = DATA_DIR / "features"

MODEL_DIR = ROOT / "models"

VECTORIZE_CONFIG = {
    "max_features": 20_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "dtype": np.float32
}


def load_comments(split_name):

    file_path = DATA_DIR / f"{split_name}.csv"

    print(f"\nLoading {split_name} comments...")

    df = pd.read_csv(
        file_path,
        usecols=["text"]
    )

    if df["text"].isna().any():
        raise ValueError(
            f"Missing comment text in {split_name}"
        )

    comments = df["text"].astype(str).tolist()

    print(f"Loaded: {len(comments):,}")

    return comments


def print_matrix_info(name, matrix):

    memory_bytes = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )

    memory_mb = memory_bytes / (1024 * 1024)

    total_elements = (
        matrix.shape[0] * matrix.shape[1]
    )

    density = (
        matrix.nnz / total_elements * 100
    )

    empty_rows = int(
        np.sum(matrix.getnnz(axis=1) == 0)
    )

    print(f"\n{name} MATRIX")
    print("-" * 50)

    print("Shape:", matrix.shape)
    print("Non-zero values:", f"{matrix.nnz:,}")
    print("Density:", f"{density:.4f}%")
    print("Sparse memory:", f"{memory_mb:.2f} MB")
    print("Empty rows:", empty_rows)


def main():

    print("=" * 65)
    print("MODERAI TF-IDF FEATURE ENGINEERING")
    print("=" * 65)

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = [
        FEATURE_DIR / "X_train.npz",
        FEATURE_DIR / "X_validation.npz",
        FEATURE_DIR / "X_test.npz",
        MODEL_DIR / "tfidf_vectorizer_20k.joblib"
    ]

    existing = [
        str(path)
        for path in outputs
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Output files already exist:\n"
            + "\n".join(existing)
            + "\nMove or rename them before rerunning."
        )

    # ------------------------------------------------
    # STEP 1: Load training comments
    # ------------------------------------------------

    train_comments = load_comments("train")

    # ------------------------------------------------
    # STEP 2: Create and fit TF-IDF on TRAIN only
    # ------------------------------------------------

    print("\nCreating TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        **VECTORIZE_CONFIG
    )

    print("\nFitting vectorizer on training data...")

    X_train = vectorizer.fit_transform(
        train_comments
    )

    print("Vocabulary size:", len(vectorizer.vocabulary_))

    print_matrix_info("TRAIN", X_train)

    # Save the fitted vectorizer.
    vectorizer_path = (
        MODEL_DIR / "tfidf_vectorizer_20k.joblib"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
        compress=3
    )

    print("\nVectorizer saved:", vectorizer_path)

    # Save training features.
    save_npz(
        FEATURE_DIR / "X_train.npz",
        X_train,
        compressed=True
    )

    print("Training features saved.")

    # ------------------------------------------------
    # STEP 3: Transform validation data
    # ------------------------------------------------

    validation_comments = load_comments("validation")

    print("\nTransforming validation data...")

    X_validation = vectorizer.transform(
        validation_comments
    )

    print_matrix_info("VALIDATION", X_validation)

    save_npz(
        FEATURE_DIR / "X_validation.npz",
        X_validation,
        compressed=True
    )

    print("Validation features saved.")

    # ------------------------------------------------
    # STEP 4: Transform test data
    # ------------------------------------------------

    test_comments = load_comments("test")

    print("\nTransforming test data...")

    X_test = vectorizer.transform(
        test_comments
    )

    print_matrix_info("TEST", X_test)

    save_npz(
        FEATURE_DIR / "X_test.npz",
        X_test,
        compressed=True
    )

    print("Test features saved.")

    # ------------------------------------------------
    # STEP 5: Final verification
    # ------------------------------------------------

    assert X_train.shape[0] == 40_000
    assert X_validation.shape[0] == 5_000
    assert X_test.shape[0] == 5_000

    assert (
        X_train.shape[1]
        == X_validation.shape[1]
        == X_test.shape[1]
    )

    assert X_train.dtype == np.float32

    print("\n" + "=" * 65)
    print("TF-IDF FEATURE ENGINEERING COMPLETED")
    print("=" * 65)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)
    print("Test shape:", X_test.shape)

    print("Vocabulary size:", len(vectorizer.vocabulary_))

    print("\nFeature directory:", FEATURE_DIR)
    print("Vectorizer:", vectorizer_path)

    print("\nTF-IDF VALIDATION: PASSED")


if __name__ == "__main__":
    main()
