
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "tfidf"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "toxicity"

print("Loading TF-IDF features...")

X_train = load_npz(DATA_DIR / "X_train.npz")
X_val = load_npz(DATA_DIR / "X_validation.npz")
X_test = load_npz(DATA_DIR / "X_test.npz")

y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET]
y_val = pd.read_csv(DATA_DIR / "y_validation.csv")[TARGET]
y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET]

# Convert annotation scores into binary labels
THRESHOLD = 0.5

y_train = (y_train >= THRESHOLD).astype(int)
y_val = (y_val >= THRESHOLD).astype(int)
y_test = (y_test >= THRESHOLD).astype(int)

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train, y_train)

print("\nValidation evaluation...")

val_predictions = model.predict(X_val)
val_probabilities = model.predict_proba(X_val)[:, 1]

print(classification_report(
    y_val,
    val_predictions,
    target_names=["Non-toxic", "Toxic"],
    zero_division=0
))

print("Validation confusion matrix:")
print(confusion_matrix(y_val, val_predictions))

print("Validation ROC-AUC:", roc_auc_score(
    y_val, val_probabilities
))

print("\nFinal test evaluation...")

test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)[:, 1]

print(classification_report(
    y_test,
    test_predictions,
    target_names=["Non-toxic", "Toxic"],
    zero_division=0
))

print("Test confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("Test ROC-AUC:", roc_auc_score(
    y_test, test_probabilities
))

# Save trained model
joblib.dump(
    model,
    MODEL_DIR / "toxicity_logistic_regression.joblib"
)

print("\nModel saved successfully!")
