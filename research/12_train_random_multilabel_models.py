
import joblib
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
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

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# Load feature matrices
print("Loading TF-IDF matrices...")

X_train = load_npz(DATA_DIR / "X_train.npz")
X_val = load_npz(DATA_DIR / "X_validation.npz")

# Load labels
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_val = pd.read_csv(DATA_DIR / "y_validation.csv")

models = {}
results = {}

print("\nStarting training of six safety classifiers...")

for label in LABELS:

    print("\n" + "=" * 65)
    print(f"TRAINING: {label.upper()}")
    print("=" * 65)

    train_target = (y_train[label] >= 0.5).astype(int)
    val_target = (y_val[label] >= 0.5).astype(int)

    positive_count = int(train_target.sum())
    negative_count = int((train_target == 0).sum())

    print("Training samples:", len(train_target))
    print("Positive samples:", positive_count)
    print("Negative samples:", negative_count)
    print("Validation positives:", int(val_target.sum()))

    if positive_count < 10 or val_target.nunique() < 2:
        print("Skipping: insufficient examples for reliable training.")
        continue

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )

    print("\nFitting model...")
    model.fit(X_train, train_target)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nClassification Report:")

    print(
        classification_report(
            val_target,
            predictions,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            val_target,
            predictions,
            labels=[0, 1],
        )
    )

    auc = roc_auc_score(val_target, probabilities)

    precision = precision_score(
        val_target, predictions, zero_division=0
    )

    recall = recall_score(
        val_target, predictions, zero_division=0
    )

    f1 = f1_score(
        val_target, predictions, zero_division=0
    )

    models[label] = model

    results[label] = {
        "training_positives": positive_count,
        "validation_positives": int(val_target.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }

    print("\nROC-AUC:", round(auc, 4))
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

# Save new models separately from the original experiment
model_path = MODEL_DIR / "random_multilabel_safety_models.joblib"

joblib.dump(models, model_path)

# Save validation results
results_df = pd.DataFrame(results).T

results_path = (
    MODEL_DIR / "random_multilabel_validation_results.csv"
)

results_df.to_csv(results_path, index_label="label")

print("\n" + "=" * 65)
print("RANDOM-SAMPLE MULTI-LABEL TRAINING COMPLETED")
print("=" * 65)

print("\nVALIDATION SUMMARY")
print(results_df.round(4))

print("\nSaved models:", model_path)
print("Saved metrics:", results_path)
print("Number of trained classifiers:", len(models))
