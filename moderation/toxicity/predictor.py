
import json
import joblib

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"

VECTOR_FILE = (
    MODEL_DIR / "tfidf_vectorizer_20k.joblib"
)

MODEL_FILE = (
    MODEL_DIR / "moderai_6_safety_models_20k.joblib"
)

THRESHOLD_FILE = (
    MODEL_DIR / "moderai_thresholds_20k.json"
)

LABELS = [
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]


class SafetyPredictor:

    def __init__(self):

        print("Loading MODERAI safety prediction components...")

        self.vectorizer = joblib.load(VECTOR_FILE)

        self.models = joblib.load(MODEL_FILE)

        with open(
            THRESHOLD_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            threshold_data = json.load(file)

        self.thresholds = threshold_data["thresholds"]

        # Verify that all expected classifiers are available.
        missing_models = [
            label for label in LABELS
            if label not in self.models
        ]

        if missing_models:
            raise ValueError(
                f"Missing classifiers: {missing_models}"
            )

        print("Vectorizer loaded.")
        print("Six safety classifiers loaded.")
        print("Threshold configuration loaded.")

    def predict(self, comment: str):

        if not isinstance(comment, str):
            raise TypeError("Comment must be a string.")

        if not comment.strip():
            raise ValueError("Comment cannot be empty.")

        # Step 1: Convert comment into TF-IDF features.
        features = self.vectorizer.transform([comment])

        # Step 2: Generate six independent probability scores.
        safety_scores = {}

        threshold_assessments = {}

        qualified_review_signals = []

        for label in LABELS:

            model = self.models[label]

            probability = float(
                model.predict_proba(features)[0, 1]
            )

            safety_scores[label] = round(
                probability,
                6
            )

            threshold = self.thresholds.get(label)

            # Only validation-qualified thresholds
            # are allowed to generate qualified flags.
            if threshold is not None:

                flag = probability >= threshold

                threshold_assessments[label] = {
                    "status": "qualified",
                    "threshold": threshold,
                    "flag": bool(flag)
                }

                if flag:
                    qualified_review_signals.append(label)

            else:

                threshold_assessments[label] = {
                    "status": "unqualified",
                    "threshold": None,
                    "flag": None
                }

        # Step 3: Return model evidence.
        return {
            "comment": comment,

            "safety_scores": safety_scores,

            "threshold_assessments": threshold_assessments,

            "qualified_review_signals": (
                qualified_review_signals
            ),

            "qualified_review_signal_present": (
                len(qualified_review_signals) > 0
            ),

            # This module never makes moderation decisions.
            "moderation_action": "no_automatic_action"
        }
