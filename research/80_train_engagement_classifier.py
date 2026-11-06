
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "datasets" / "engagement" / "splits"
MODEL_DIR = ROOT / "models" / "engagement"
REPORT_DIR = ROOT / "reports" / "engagement"

TRAIN_FILE = SPLIT_DIR / "train_v2.csv"
VALIDATION_FILE = SPLIT_DIR / "validation_v2.csv"

MODEL_FILE = MODEL_DIR / "engagement_tfidf_logreg_v1.joblib"
REPORT_FILE = REPORT_DIR / "engagement_validation_report_v1.json"
PREDICTIONS_FILE = REPORT_DIR / "engagement_validation_predictions_v1.csv"
CONFUSION_FILE = REPORT_DIR / "engagement_validation_confusion_v1.csv"

LABELS = [
    "POSITIVE",
    "NEGATIVE",
    "CRITICAL",
    "QUESTION",
    "NEUTRAL",
]


def main():

    print("=" * 85)
    print("MODERAI — ENGAGEMENT CLASSIFIER TRAINING V1")
    print("=" * 85)

    # --------------------------------------------------
    # 1. Check files
    # --------------------------------------------------

    for path in [TRAIN_FILE, VALIDATION_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_files = [
        MODEL_FILE,
        REPORT_FILE,
        PREDICTIONS_FILE,
        CONFUSION_FILE,
    ]

    existing = [
        path for path in output_files if path.exists()
    ]

    if existing:
        raise FileExistsError(
            f"Outputs already exist. Nothing overwritten: {existing}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------

    train = pd.read_csv(
        TRAIN_FILE,
        dtype="string",
        keep_default_na=False,
    )

    validation = pd.read_csv(
        VALIDATION_FILE,
        dtype="string",
        keep_default_na=False,
    )

    required = {"comment_id", "comment_text", "intent"}

    for name, frame in [
        ("Training", train),
        ("Validation", validation),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {sorted(missing)}"
            )

        if frame["comment_text"].str.strip().eq("").any():
            raise ValueError(
                f"{name} dataset contains blank comments."
            )

    if train["intent"].eq("UNCLEAR").any():
        raise ValueError(
            "UNCLEAR must not be present in supervised training."
        )

    if set(train["intent"].unique()) != set(LABELS):
        raise ValueError(
            "Training must contain exactly the five supervised classes."
        )

    if not set(validation["intent"].unique()).issubset(
        set(LABELS + ["UNCLEAR"])
    ):
        raise ValueError("Unexpected validation label found.")

    if validation["intent"].eq("UNCLEAR").sum() == 0:
        raise ValueError(
            "Validation must retain UNCLEAR examples for abstention study."
        )

    X_train = train["comment_text"].astype(str)
    y_train = train["intent"].astype(str)

    X_validation = validation["comment_text"].astype(str)
    y_validation = validation["intent"].astype(str)

    print("\n1. DATASET INFORMATION")
    print("-" * 85)

    print("Training comments:", len(train))
    print("Validation comments:", len(validation))

    print("\nTraining label distribution:")
    print(y_train.value_counts().to_string())

    print("\nValidation label distribution:")
    print(y_validation.value_counts().to_string())

    # --------------------------------------------------
    # 3. Build TF-IDF + Logistic Regression pipeline
    # --------------------------------------------------

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                max_features=20_000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
                dtype="float32",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
                solver="lbfgs",
            ),
        ),
    ])

    # --------------------------------------------------
    # 4. Train
    # --------------------------------------------------

    print("\n2. MODEL TRAINING")
    print("-" * 85)

    print("Training TF-IDF + Logistic Regression...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    print("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))
    print("Classifier classes:", classifier.classes_.tolist())

    # --------------------------------------------------
    # 5. Validation predictions
    # --------------------------------------------------

    print("\n3. VALIDATION EVALUATION")
    print("-" * 85)

    predicted = pipeline.predict(X_validation)
    probabilities = pipeline.predict_proba(X_validation)

    confidence = probabilities.max(axis=1)

    # Five-class metrics apply only to clear validation labels.
    clear_mask = y_validation.isin(LABELS)

    y_clear = y_validation[clear_mask]
    predicted_clear = predicted[clear_mask]

    accuracy = accuracy_score(
        y_clear,
        predicted_clear,
    )

    macro_f1 = f1_score(
        y_clear,
        predicted_clear,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_clear,
        predicted_clear,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_clear,
        predicted_clear,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    print("Clear validation comments:", int(clear_mask.sum()))
    print("UNCLEAR validation comments:", int((~clear_mask).sum()))

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_clear,
            predicted_clear,
            labels=LABELS,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 6. Confusion matrix
    # --------------------------------------------------

    matrix = confusion_matrix(
        y_clear,
        predicted_clear,
        labels=LABELS,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    )

    confusion_df.index.name = "actual_intent"

    print("\n4. CONFUSION MATRIX")
    print("-" * 85)
    print(confusion_df.to_string())

    # --------------------------------------------------
    # 7. Save validation predictions
    # --------------------------------------------------

    predictions_df = validation[
        ["comment_id", "comment_text", "intent"]
    ].copy()

    predictions_df = predictions_df.rename(
        columns={"intent": "actual_intent"}
    )

    predictions_df["predicted_intent"] = predicted
    predictions_df["max_probability"] = confidence

    for index, label in enumerate(classifier.classes_):
        predictions_df[
            f"probability_{label.lower()}"
        ] = probabilities[:, index]

    predictions_df["is_unclear"] = (
        predictions_df["actual_intent"].eq("UNCLEAR")
    )

    predictions_df["correct_clear_prediction"] = (
        predictions_df["actual_intent"].eq(
            predictions_df["predicted_intent"]
        )
        & ~predictions_df["is_unclear"]
    )

    # --------------------------------------------------
    # 8. Save model and reports
    # --------------------------------------------------

    print("\n5. SAVING MODEL AND REPORTS")
    print("-" * 85)

    joblib.dump(pipeline, MODEL_FILE)

    confusion_df.to_csv(CONFUSION_FILE)

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    report_data = {
        "model": "TF-IDF + Logistic Regression",
        "version": "v1",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "validation_clear_rows": int(clear_mask.sum()),
        "validation_unclear_rows": int((~clear_mask).sum()),
        "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
        "classes": classifier.classes_.tolist(),
        "accuracy_clear_validation": float(accuracy),
        "macro_f1_clear_validation": float(macro_f1),
        "weighted_f1_clear_validation": float(weighted_f1),
        "classification_report": report,
        "note": (
            "UNCLEAR is not a supervised class. "
            "UNCLEAR validation examples are retained for "
            "future confidence-based abstention analysis."
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(report_data, indent=4),
        encoding="utf-8",
    )

    print("Model:", MODEL_FILE)
    print("Validation report:", REPORT_FILE)
    print("Predictions:", PREDICTIONS_FILE)
    print("Confusion matrix:", CONFUSION_FILE)

    print("\n6. FINAL STATUS")
    print("-" * 85)

    print("[PASS] Five-class model trained")
    print("[PASS] UNCLEAR excluded from supervised training")
    print("[PASS] Validation evaluated without using test data")
    print("[PASS] Model and reports saved")

    print("\nEngagement classifier training completed.")


if __name__ == "__main__":
    main()
