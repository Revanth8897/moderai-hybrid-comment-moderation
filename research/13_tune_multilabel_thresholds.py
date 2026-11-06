
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT / "datasets" / "processed" / "tfidf_random"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_multilabel_safety_models.joblib"
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

TARGET_PRECISION = 0.70
MIN_PREDICTED_POSITIVES = 30

print("Loading models and validation data...")

models = joblib.load(MODEL_PATH)

X_val = load_npz(DATA_DIR / "X_validation.npz")
y_val = pd.read_csv(DATA_DIR / "y_validation.csv")

threshold_results = {}
summary = []

for label in LABELS:

    print("\n" + "=" * 65)
    print("THRESHOLD ANALYSIS:", label.upper())
    print("=" * 65)

    model = models[label]

    actual = (y_val[label] >= 0.5).astype(int)

    probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        actual,
        probabilities,
    )

    # precision and recall contain one extra final entry.
    precision = precision[:-1]
    recall = recall[:-1]

    predicted_counts = np.array([
        np.sum(probabilities >= threshold)
        for threshold in thresholds
    ])

    # Find thresholds meeting both requirements.
    valid_indices = np.where(
        (precision >= TARGET_PRECISION)
        & (predicted_counts >= MIN_PREDICTED_POSITIVES)
    )[0]

    print("Actual positive examples:", int(actual.sum()))
    print("Target precision:", TARGET_PRECISION)
    print("Minimum predicted positives:", MIN_PREDICTED_POSITIVES)

    if len(valid_indices) > 0:

        # Maximize recall among thresholds meeting requirements.
        best_index = valid_indices[
            np.argmax(recall[valid_indices])
        ]

        best_threshold = float(thresholds[best_index])

        selected_precision = float(precision[best_index])
        selected_recall = float(recall[best_index])
        selected_count = int(predicted_counts[best_index])

        status = "selected"

        print("\nSelected threshold:", best_threshold)
        print("Validation precision:", round(selected_precision, 4))
        print("Validation recall:", round(selected_recall, 4))
        print("Predicted positive comments:", selected_count)

    else:

        best_threshold = None
        selected_precision = None
        selected_recall = None
        selected_count = None

        status = "no_threshold_meets_requirements"

        print("\nNo threshold meets both requirements.")
        print("No threshold will be selected for this category.")

    threshold_results[label] = {
        "threshold": best_threshold,
        "target_precision": TARGET_PRECISION,
        "minimum_predicted_positives": MIN_PREDICTED_POSITIVES,
        "validation_precision": selected_precision,
        "validation_recall": selected_recall,
        "predicted_positive_count": selected_count,
        "status": status,
    }

    summary.append({
        "label": label,
        "threshold": best_threshold,
        "precision": selected_precision,
        "recall": selected_recall,
        "predicted_positives": selected_count,
        "status": status,
    })

# Save threshold configuration.
threshold_path = MODEL_DIR / "multilabel_thresholds.json"

with open(threshold_path, "w", encoding="utf-8") as file:
    json.dump(threshold_results, file, indent=4)

# Save summary.
summary_df = pd.DataFrame(summary)

summary_path = (
    MODEL_DIR / "multilabel_threshold_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 65)
print("THRESHOLD TUNING COMPLETED")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nSaved threshold configuration:", threshold_path)
print("Saved summary:", summary_path)
