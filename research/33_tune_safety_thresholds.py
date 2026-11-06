
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
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

MODEL_FILE = (
    MODEL_DIR / "moderai_6_safety_models_20k.joblib"
)

THRESHOLD_FILE = (
    MODEL_DIR / "moderai_thresholds_20k.json"
)

REPORT_FILE = (
    REPORT_DIR / "moderai_threshold_summary_20k.csv"
)

TARGET_PRECISION = 0.70
MIN_PREDICTED_POSITIVES = 10
BASELINE_THRESHOLD = 0.50


def load_data():

    print("\nLoading trained models...")

    models = joblib.load(MODEL_FILE)

    print("Loading validation features...")

    X_validation = load_npz(
        FEATURE_DIR / "X_validation.npz"
    )

    print("Loading validation labels...")

    validation_df = pd.read_csv(
        DATA_DIR / "validation.csv",
        usecols=LABELS
    )

    y_validation = (
        validation_df[LABELS].to_numpy() >= 0.5
    ).astype(np.int8)

    return models, X_validation, y_validation


def evaluate_predictions(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    return {
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "predicted_positives": int(
            predictions.sum()
        )
    }


def tune_label(label, model, X_validation, y_true):

    print("\n" + "-" * 65)
    print(f"TUNING: {label.upper()}")
    print("-" * 65)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    baseline = evaluate_predictions(
        y_true,
        probabilities,
        BASELINE_THRESHOLD
    )

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    candidates = []

    for i, threshold in enumerate(thresholds):

        predicted_count = int(
            np.sum(probabilities >= threshold)
        )

        precision = float(precision_values[i])
        recall = float(recall_values[i])

        if (
            precision >= TARGET_PRECISION
            and predicted_count >= MIN_PREDICTED_POSITIVES
        ):

            candidates.append({
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "predicted_positives": predicted_count
            })

    if not candidates:

        print("No threshold meets the tuning conditions.")
        print("Threshold status: UNQUALIFIED")

        return {
            "label": label,
            "status": "UNQUALIFIED",
            "threshold": None,
            "baseline": baseline,
            "tuned": None
        }

    # Highest recall among candidates meeting the precision target.
    # Tie-break using higher precision, then lower threshold.
    best = max(
        candidates,
        key=lambda item: (
            item["recall"],
            item["precision"],
            -item["threshold"]
        )
    )

    tuned = evaluate_predictions(
        y_true,
        probabilities,
        best["threshold"]
    )

    print(f"Selected threshold: {best['threshold']:.6f}")
    print(f"Precision: {tuned['precision']:.4f}")
    print(f"Recall: {tuned['recall']:.4f}")
    print(f"F1-score: {tuned['f1_score']:.4f}")
    print(
        "Predicted positives:",
        tuned["predicted_positives"]
    )

    return {
        "label": label,
        "status": "QUALIFIED",
        "threshold": best["threshold"],
        "baseline": baseline,
        "tuned": tuned
    }


def main():

    print("=" * 65)
    print("MODERAI SAFETY THRESHOLD TUNING")
    print("=" * 65)

    if THRESHOLD_FILE.exists():
        raise FileExistsError(THRESHOLD_FILE)

    if REPORT_FILE.exists():
        raise FileExistsError(REPORT_FILE)

    models, X_validation, y_validation = load_data()

    results = []

    for index, label in enumerate(LABELS):

        result = tune_label(
            label,
            models[label],
            X_validation,
            y_validation[:, index]
        )

        results.append(result)

    # Save thresholds. Unqualified labels receive null.
    threshold_data = {
        "target_precision": TARGET_PRECISION,
        "minimum_predicted_positives": MIN_PREDICTED_POSITIVES,
        "selection_method": "highest_recall_meeting_precision_target",
        "thresholds": {
            item["label"]: item["threshold"]
            for item in results
        }
    }

    with open(
        THRESHOLD_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_data,
            file,
            indent=4
        )

    # Prepare a readable report.
    report_rows = []

    for item in results:

        baseline = item["baseline"]
        tuned = item["tuned"]

        report_rows.append({
            "label": item["label"],
            "status": item["status"],
            "baseline_threshold": BASELINE_THRESHOLD,
            "baseline_precision": baseline["precision"],
            "baseline_recall": baseline["recall"],
            "baseline_f1": baseline["f1_score"],
            "baseline_predicted_positives": (
                baseline["predicted_positives"]
            ),
            "selected_threshold": item["threshold"],
            "tuned_precision": (
                tuned["precision"] if tuned else None
            ),
            "tuned_recall": (
                tuned["recall"] if tuned else None
            ),
            "tuned_f1": (
                tuned["f1_score"] if tuned else None
            ),
            "tuned_predicted_positives": (
                tuned["predicted_positives"] if tuned else None
            )
        })

    report_df = pd.DataFrame(report_rows)

    report_df.to_csv(
        REPORT_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("THRESHOLD TUNING SUMMARY")
    print("=" * 65)

    print(
        report_df[
            [
                "label",
                "status",
                "selected_threshold",
                "tuned_precision",
                "tuned_recall",
                "tuned_f1"
            ]
        ].to_string(index=False)
    )

    print("\nThreshold file:", THRESHOLD_FILE)
    print("Report file:", REPORT_FILE)

    print("\nTHRESHOLD TUNING COMPLETED")


if __name__ == "__main__":
    main()
