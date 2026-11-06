
import json
import joblib
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT / "datasets" / "processed" / "tfidf_random"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_multilabel_safety_models.joblib"
)

THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "multilabel_thresholds.json"
)

MODEL_DIR = PROJECT_ROOT / "models"

LABELS = [
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

print("Loading models, thresholds and test data...")

models = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

X_test = load_npz(DATA_DIR / "X_test.npz")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")

results = []

for label in LABELS:

    print("\n" + "=" * 65)
    print("TEST EVALUATION:", label.upper())
    print("=" * 65)

    model = models[label]

    actual = (y_test[label] >= 0.5).astype(int)

    probabilities = model.predict_proba(X_test)[:, 1]

    # ROC-AUC evaluates ranking, without selecting a threshold.
    auc = roc_auc_score(actual, probabilities)

    config = threshold_config[label]
    threshold = config["threshold"]

    if threshold is None:

        print("No qualified validation threshold.")
        print("Test ROC-AUC:", round(auc, 4))
        print("Precision/recall/F1: Not evaluated at a tuned threshold.")

        results.append({
            "label": label,
            "threshold": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": auc,
            "status": "no_qualified_validation_threshold",
        })

        continue

    # Apply the threshold selected on validation data.
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        actual, predictions, zero_division=0
    )

    recall = recall_score(
        actual, predictions, zero_division=0
    )

    f1 = f1_score(
        actual, predictions, zero_division=0
    )

    print("Selected validation threshold:", round(threshold, 6))
    print("Test positive examples:", int(actual.sum()))

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predictions,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(actual, predictions, labels=[0, 1]))

    print("\nTest ROC-AUC:", round(auc, 4))
    print("Test Precision:", round(precision, 4))
    print("Test Recall:", round(recall, 4))
    print("Test F1:", round(f1, 4))

    results.append({
        "label": label,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "status": "evaluated",
    })

# Save test results
results_df = pd.DataFrame(results)

output_path = MODEL_DIR / "multilabel_test_results.csv"

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION SUMMARY")
print("=" * 65)

print(results_df.round(4).to_string(index=False))

print("\nSaved results:", output_path)
