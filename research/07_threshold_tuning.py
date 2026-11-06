
import joblib
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "tfidf"
MODEL_PATH = (
    PROJECT_ROOT / "models"
    / "toxicity_logistic_regression.joblib"
)

model = joblib.load(MODEL_PATH)

# Load validation data
X_val = load_npz(DATA_DIR / "X_validation.npz")
y_val = pd.read_csv(DATA_DIR / "y_validation.csv")["toxicity"]
y_val = (y_val >= 0.5).astype(int)

# Obtain probabilities
val_probabilities = model.predict_proba(X_val)[:, 1]

# Calculate precision and recall at different thresholds
precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities
)

# Select a threshold that achieves at least 70% precision
target_precision = 0.70

valid_indices = [
    i for i in range(len(thresholds))
    if precision[i] >= target_precision
]

if not valid_indices:
    raise ValueError(
        "No threshold achieved the target precision."
    )

best_index = max(
    valid_indices,
    key=lambda i: recall[i]
)

best_threshold = thresholds[best_index]

print("Selected threshold:", best_threshold)
print("Validation precision:", precision[best_index])
print("Validation recall:", recall[best_index])

# Save selected threshold
import json

with open(
    PROJECT_ROOT / "models" / "toxicity_threshold.json",
    "w"
) as file:
    json.dump(
        {"threshold": float(best_threshold)},
        file,
        indent=4
    )

# Evaluate once on the untouched test set
X_test = load_npz(DATA_DIR / "X_test.npz")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")["toxicity"]
y_test = (y_test >= 0.5).astype(int)

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL TEST RESULTS AT SELECTED THRESHOLD")

print(classification_report(
    y_test,
    test_predictions,
    target_names=["Non-toxic", "Toxic"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("Toxic precision:", precision_score(
    y_test, test_predictions, zero_division=0
))

print("Toxic recall:", recall_score(
    y_test, test_predictions, zero_division=0
))

print("Toxic F1:", f1_score(
    y_test, test_predictions, zero_division=0
))
