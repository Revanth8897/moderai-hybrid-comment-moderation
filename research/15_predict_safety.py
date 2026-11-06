
import joblib
import json
import re

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR / "random_multilabel_safety_models.joblib"
)

VECTORIZER_PATH = (
    MODEL_DIR / "tfidf_vectorizer_random.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR / "multilabel_thresholds.json"
)

# Load saved artifacts
print("Loading MODERAI safety models...")

models = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

# Only these two categories passed our test precision target.
APPROVED_LABELS = {
    "toxicity",
    "insult",
}

ALL_LABELS = [
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


def preprocess_text(text):
    """
    Apply the same basic text normalization
    used during dataset preprocessing.
    """

    if not isinstance(text, str):
        raise TypeError("Comment must be a string.")

    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def predict_safety(comment):

    cleaned_text = preprocess_text(comment)

    if not cleaned_text:
        raise ValueError("Comment cannot be empty.")

    # Convert comment into TF-IDF features.
    features = vectorizer.transform([cleaned_text])

    scores = {}
    flags = {}

    for label in ALL_LABELS:

        model = models[label]

        probability = float(
            model.predict_proba(features)[0, 1]
        )

        scores[label] = round(probability, 4)

        # Apply only test-evaluated thresholds.
        if label in APPROVED_LABELS:

            threshold = threshold_config[label]["threshold"]

            flags[label] = probability >= threshold

        else:
            flags[label] = None

    # A model flag means review candidate, not a policy violation.
    review_recommended = any(
        flags[label] is True
        for label in APPROVED_LABELS
    )

    return {
        "comment": cleaned_text,

        "safety_scores": scores,

        "validated_flags": flags,

        "review_recommended": review_recommended,

        "model_status": {
            "validated_categories": sorted(APPROVED_LABELS),
            "experimental_categories": [
                label
                for label in ALL_LABELS
                if label not in APPROVED_LABELS
            ],
            "severe_toxicity": "unsupported_by_current_dataset",
        },

        "moderation_action": "no_automatic_action",
    }


if __name__ == "__main__":

    print("\nMODERAI SAFETY PREDICTION ENGINE")
    print("=" * 50)

    while True:

        comment = input(
            "\nEnter a comment (or type 'exit'): "
        )

        if comment.strip().lower() == "exit":
            print("Exiting MODERAI...")
            break

        try:

            result = predict_safety(comment)

            print("\nPREDICTION RESULT")
            print("=" * 50)

            print("\nSafety scores:")

            for label, score in result["safety_scores"].items():
                print(f"{label:20}: {score:.4f}")

            print("\nValidated flags:")

            for label, flag in result["validated_flags"].items():
                print(f"{label:20}: {flag}")

            print(
                "\nHuman review recommended:",
                result["review_recommended"]
            )

            print(
                "Moderation action:",
                result["moderation_action"]
            )

        except (ValueError, TypeError) as error:
            print("Input error:", error)
