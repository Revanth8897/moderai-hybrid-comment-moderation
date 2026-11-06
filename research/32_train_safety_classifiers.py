
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "moderai_50k"
)

FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

LABELS = [
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

MODEL_OUTPUT = (
    MODEL_DIR / "moderai_6_safety_models_20k.joblib"
)

REPORT_OUTPUT = (
    REPORT_DIR / "moderai_validation_results_20k.csv"
)

CONFIG = {
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 1000,
    "C": 1.0,
    "random_state": 42
}


def load_features():

    print("\nLoading TF-IDF features...")

    X_train = load_npz(
        FEATURE_DIR / "X_train.npz"
    )

    X_validation = load_npz(
        FEATURE_DIR / "X_validation.npz"
    )

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)

    return X_train, X_validation


def load_labels():

    print("\nLoading annotation labels...")

    train_df = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=LABELS
    )

    validation_df = pd.read_csv(
        DATA_DIR / "validation.csv",
        usecols=LABELS
    )

    y_train = (
        train_df[LABELS].to_numpy() >= 0.5
    ).astype(np.int8)

    y_validation = (
        validation_df[LABELS].to_numpy() >= 0.5
    ).astype(np.int8)

    print("Training label shape:", y_train.shape)
    print("Validation label shape:", y_validation.shape)

    return y_train, y_validation


def train_models(X_train, y_train, X_validation, y_validation):

    print("\n" + "=" * 65)
    print("TRAINING SIX INDEPENDENT SAFETY CLASSIFIERS")
    print("=" * 65)

    models = {}
    results = []

    for index, label in enumerate(LABELS):

        print("\n" + "-" * 65)
        print(f"Training classifier: {label.upper()}")
        print("-" * 65)

        y_train_label = y_train[:, index]
        y_val_label = y_validation[:, index]

        positive_count = int(y_train_label.sum())
        negative_count = int(
            len(y_train_label) - positive_count
        )

        print("Positive samples:", f"{positive_count:,}")
        print("Negative samples:", f"{negative_count:,}")

        if positive_count == 0 or negative_count == 0:
            raise ValueError(
                f"Cannot train {label}: only one class exists."
            )

        model = LogisticRegression(
            **CONFIG
        )

        model.fit(
            X_train,
            y_train_label
        )

        # Evaluate only on validation data.
        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        predictions = (
            probabilities >= 0.50
        ).astype(np.int8)

        precision = precision_score(
            y_val_label,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val_label,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val_label,
            predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            y_val_label,
            probabilities
        )

        results.append({
            "label": label,
            "train_positive": positive_count,
            "validation_positive": int(y_val_label.sum()),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
            "threshold": 0.50,
            "iterations": int(model.n_iter_[0])
        })

        models[label] = model

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Iterations: {model.n_iter_[0]}")

    return models, pd.DataFrame(results)


def main():

    print("=" * 65)
    print("MODERAI SAFETY MODEL TRAINING")
    print("=" * 65)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if MODEL_OUTPUT.exists():
        raise FileExistsError(
            f"Model already exists: {MODEL_OUTPUT}"
        )

    if REPORT_OUTPUT.exists():
        raise FileExistsError(
            f"Report already exists: {REPORT_OUTPUT}"
        )

    X_train, X_validation = load_features()

    y_train, y_validation = load_labels()

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training feature/label mismatch.")

    if X_validation.shape[0] != y_validation.shape[0]:
        raise ValueError("Validation feature/label mismatch.")

    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError("Feature dimensions do not match.")

    models, results_df = train_models(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    # Save all six models together.
    joblib.dump(
        models,
        MODEL_OUTPUT,
        compress=3
    )

    # Save validation metrics.
    results_df.to_csv(
        REPORT_OUTPUT,
        index=False
    )

    print("\n" + "=" * 65)
    print("VALIDATION PERFORMANCE SUMMARY")
    print("=" * 65)

    print(
        results_df[
            [
                "label",
                "precision",
                "recall",
                "f1_score",
                "roc_auc"
            ]
        ].to_string(index=False)
    )

    print("\nModels saved:", MODEL_OUTPUT)
    print("Report saved:", REPORT_OUTPUT)

    print("\n" + "=" * 65)
    print("SIX CLASSIFIERS TRAINED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
