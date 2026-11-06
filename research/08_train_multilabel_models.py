
import joblib
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "tfidf"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# Load TF-IDF feature matrices
print("Loading feature matrices...")

X_train = load_npz(DATA_DIR / "X_train.npz")
X_val = load_npz(DATA_DIR / "X_validation.npz")

# Load labels
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_val = pd.read_csv(DATA_DIR / "y_validation.csv")

models = {}
results = {}

print("\nStarting multi-label model training...")

for label in LABELS:

    print("\n" + "=" * 60)
    print(f"Training classifier: {label}")
    print("=" * 60)

    # Convert annotation scores into binary targets
    train_target = (y_train[label] >= 0.5).astype(int)
    val_target = (y_val[label] >= 0.5).astype(int)

    print("Training samples:", len(train_target))
    print("Positive training samples:", train_target.sum())

    # Train an independent binary classifier
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )

    model.fit(X_train, train_target)

    # Validation predictions
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Evaluation
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
    print(confusion_matrix(val_target, predictions, labels=[0, 1]))

    if val_target.nunique() == 2:
        auc = roc_auc_score(val_target, probabilities)
        print("ROC-AUC:", round(auc, 4))
    else:
        auc = None
        print("ROC-AUC unavailable: validation has only one class.")

    # Store model and metrics
    models[label] = model

    results[label] = {
        "positive_samples": int(train_target.sum()),
        "precision": float(
            __import__("sklearn.metrics", fromlist=["precision_score"])
            .precision_score(val_target, predictions, zero_division=0)
        ),
        "recall": float(
            __import__("sklearn.metrics", fromlist=["recall_score"])
            .recall_score(val_target, predictions, zero_division=0)
        ),
        "f1": float(
            __import__("sklearn.metrics", fromlist=["f1_score"])
            .f1_score(val_target, predictions, zero_division=0)
        ),
        "roc_auc": float(auc) if auc is not None else None,
    }

    print(f"\nCompleted: {label}")

# Save all seven classifiers together
model_path = MODEL_DIR / "multilabel_safety_models.joblib"
joblib.dump(models, model_path)

# Save validation metrics
results_df = pd.DataFrame(results).T
results_path = MODEL_DIR / "multilabel_validation_results.csv"
results_df.to_csv(results_path, index_label="label")

print("\n" + "=" * 60)
print("MULTI-LABEL TRAINING COMPLETED")
print("=" * 60)

print("\nValidation summary:")
print(results_df)

print("\nModels saved to:", model_path)
print("Metrics saved to:", results_path)
