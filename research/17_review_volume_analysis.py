
import joblib
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent.parent

X_val = load_npz(
    ROOT / "datasets/processed/tfidf_random/X_validation.npz"
)

models = joblib.load(
    ROOT / "models/random_multilabel_safety_models.joblib"
)

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

predictions = {}

for label in ["toxicity", "insult"]:

    scores = models[label].predict_proba(X_val)[:, 1]

    predictions[label] = {
        "scores": scores,
        "preliminary": scores >= thresholds[label]["preliminary"],
        "high": scores >= thresholds[label]["high"]
    }

tox_pre = predictions["toxicity"]["preliminary"]
ins_pre = predictions["insult"]["preliminary"]

tox_high = predictions["toxicity"]["high"]
ins_high = predictions["insult"]["high"]

# Combine categories: one review item per comment.
preliminary_review = tox_pre | ins_pre
high_review = tox_high | ins_high

both_preliminary = tox_pre & ins_pre
both_high = tox_high & ins_high

total = X_val.shape[0]

print("\nMODERAI REVIEW VOLUME ANALYSIS")
print("=" * 55)

print("Total validation comments:", total)

print("\nPRELIMINARY REVIEW")
print("Toxicity flags:", int(tox_pre.sum()))
print("Insult flags:", int(ins_pre.sum()))
print("Both classifiers flagged:", int(both_preliminary.sum()))
print("Unique comments for review:", int(preliminary_review.sum()))
print(
    "Review percentage:",
    round(preliminary_review.mean() * 100, 2), "%"
)

print("\nHIGH-CONFIDENCE REVIEW")
print("Toxicity flags:", int(tox_high.sum()))
print("Insult flags:", int(ins_high.sum()))
print("Both classifiers flagged:", int(both_high.sum()))
print("Unique comments for review:", int(high_review.sum()))
print(
    "Review percentage:",
    round(high_review.mean() * 100, 2), "%"
)

print("\nADDITIONAL PRELIMINARY-ONLY COMMENTS")
print(
    int((preliminary_review & ~high_review).sum())
)

print("\nREVIEW VOLUME ANALYSIS COMPLETED")
