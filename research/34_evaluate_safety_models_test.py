
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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
    REPORT_DIR / "moderai_test_results_20k.csv"
)

DEFAULT_THRESHOLD = 0.50


def load_test_data():

    print("\nLoading trained models...")

    models = joblib.load(MODEL_FILE)

    print("Loading test TF-IDF features...")

    X_test = load_npz(
        FEATURE_DIR / "X_test.npz"
    )

    print("Loading test annotation labels...")

    test_df = pd.read_csv(
        DATA_DIR / "test.csv",
        usecols=LABELS
    )

    y_test = (
        test_df[LABELS].to_numpy() >= 0.5
    ).astype(np.int8)

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        threshold_data = json.load(file)

    thresholds = threshold_data["thresholds"]

    return models, X_test, y_test, thresholds


def evaluate_label(
    label,
    y_true,
    probabilities,
    threshold,
    evaluation_type
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "label": label,
        "evaluation_type": evaluation_type,
        "threshold": threshold,
        "test_support": int(y_true.sum()),
        "predicted_positives": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }


def main():

    print("=" * 70)
    print("MODERAI FINAL SAFETY MODEL TEST EVALUATION")
    print("=" * 70)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if REPORT_FILE.exists():
        raise FileExistsError(
            f"Report already exists: {REPORT_FILE}\n"
            "Do not overwrite the original test evaluation."
        )

    models, X_test, y_test, thresholds = load_test_data()

    if X_test.shape[0] != 5_000:
        raise ValueError("Unexpected test feature count.")

    if y_test.shape[0] != 5_000:
        raise ValueError("Unexpected test label count.")

    results = []

    for index, label in enumerate(LABELS):

        print("\n" + "=" * 65)
        print(f"TESTING: {label.upper()}")
        print("=" * 65)

        y_true = y_test[:, index]

        model = models[label]

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # First: evaluate using default threshold.
        baseline = evaluate_label(
            label,
            y_true,
            probabilities,
            DEFAULT_THRESHOLD,
            "default_0.50"
        )

        results.append(baseline)

        print("\nDEFAULT THRESHOLD: 0.50")
        print(f"Precision: {baseline['precision']:.4f}")
        print(f"Recall:    {baseline['recall']:.4f}")
        print(f"F1-score:  {baseline['f1_score']:.4f}")
        print(f"ROC-AUC:   {baseline['roc_auc']:.4f}")
        print(
            "TP / FP / FN / TN:",
            baseline["true_positives"],
            baseline["false_positives"],
            baseline["false_negatives"],
            baseline["true_negatives"]
        )

        # Second: evaluate tuned threshold only if qualified.
        selected_threshold = thresholds.get(label)

        if selected_threshold is not None:

            tuned = evaluate_label(
                label,
                y_true,
                probabilities,
                selected_threshold,
                "validation_tuned"
            )

            results.append(tuned)

            print(
                f"\nTUNED THRESHOLD: {selected_threshold:.6f}"
            )

            print(f"Precision: {tuned['precision']:.4f}")
            print(f"Recall:    {tuned['recall']:.4f}")
            print(f"F1-score:  {tuned['f1_score']:.4f}")
            print(f"ROC-AUC:   {tuned['roc_auc']:.4f}")

            print(
                "TP / FP / FN / TN:",
                tuned["true_positives"],
                tuned["false_positives"],
                tuned["false_negatives"],
                tuned["true_negatives"]
            )

        else:

            print(
                "\nNo qualified validation threshold. "
                "Default result is exploratory only."
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "label",
                "evaluation_type",
                "threshold",
                "test_support",
                "precision",
                "recall",
                "f1_score",
                "roc_auc"
            ]
        ].to_string(index=False)
    )

    print("\nDetailed report:", REPORT_FILE)

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
