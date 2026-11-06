
import joblib
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

# Load validation data
X_val = load_npz(
    ROOT / "datasets/processed/tfidf_random/X_validation.npz"
)

y_val = pd.read_csv(
    ROOT / "datasets/processed/tfidf_random/y_validation.csv"
)

models = joblib.load(
    ROOT / "models/random_multilabel_safety_models.joblib"
)

# Threshold configuration
thresholds = {
    "toxicity": {
        "preliminary": 0.60,
        "high": 0.7623006963643008
    },
    "insult": {
        "preliminary": 0.60,
        "high": 0.7603478650820913
    }
}

# Ground truth: positive if either category is positive.
actual_toxicity = (y_val["toxicity"].values >= 0.5)
actual_insult = (y_val["insult"].values >= 0.5)

y_true = actual_toxicity | actual_insult

# Generate predictions
predictions = {}

for label in ["toxicity", "insult"]:

    scores = models[label].predict_proba(X_val)[:, 1]

    predictions[label] = {
        "preliminary": (
            scores >= thresholds[label]["preliminary"]
        ),
        "high": (
            scores >= thresholds[label]["high"]
        )
    }

# Combine classifier predictions
preliminary = (
    predictions["toxicity"]["preliminary"]
    | predictions["insult"]["preliminary"]
)

high = (
    predictions["toxicity"]["high"]
    | predictions["insult"]["high"]
)

# Additional preliminary-only group
preliminary_only = preliminary & ~high

def evaluate_strategy(name, predicted):

    precision = precision_score(
        y_true, predicted, zero_division=0
    )

    recall = recall_score(
        y_true, predicted, zero_division=0
    )

    f1 = f1_score(
        y_true, predicted, zero_division=0
    )

    matrix = confusion_matrix(y_true, predicted)

    print(f"\n{name}")
    print("-" * 50)

    print("Comments flagged:", int(predicted.sum()))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("Confusion matrix:")
    print(matrix)

print("\nMODERAI REVIEW STRATEGY EVALUATION")
print("=" * 60)

print("Total comments:", len(y_true))
print("Actual positive comments:", int(y_true.sum()))

evaluate_strategy(
    "PRELIMINARY REVIEW",
    preliminary
)

evaluate_strategy(
    "ELEVATED-PRIORITY REVIEW",
    high
)

print("\nPRELIMINARY-ONLY GROUP")
print("-" * 50)

print("Comments:", int(preliminary_only.sum()))

if preliminary_only.sum() > 0:

    correct = int(
        (preliminary_only & y_true).sum()
    )

    print("Actual positive comments:", correct)

    print(
        "Positive rate:",
        round(
            correct / preliminary_only.sum(),
            4
        )
    )

print("\nREVIEW STRATEGY EVALUATION COMPLETED")
