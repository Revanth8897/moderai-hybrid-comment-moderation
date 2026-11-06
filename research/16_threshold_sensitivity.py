
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent

# Load validation data only
X_val = load_npz(
    ROOT / "datasets/processed/tfidf_random/X_validation.npz"
)

y_val = pd.read_csv(
    ROOT / "datasets/processed/tfidf_random/y_validation.csv"
)

models = joblib.load(
    ROOT / "models/random_multilabel_safety_models.joblib"
)

labels = ["toxicity", "insult"]

thresholds = np.arange(0.30, 0.851, 0.05)

for label in labels:

    print("\n" + "=" * 65)
    print(f"THRESHOLD ANALYSIS: {label.upper()}")
    print("=" * 65)

    y_true = (
      y_val[label].values >= 0.5
    ).astype(int)

    probabilities = models[label].predict_proba(X_val)[:, 1]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )

        recall = recall_score(
            y_true, predictions, zero_division=0
        )

        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        predicted_count = int(predictions.sum())

        results.append({
            "threshold": round(float(threshold), 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted_positive": predicted_count
        })

    result_df = pd.DataFrame(results)

    print(result_df.to_string(index=False))

    print("\nValidation actual positive examples:", int(y_true.sum()))
