
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# Project root:
# moderation/engagement/predictor.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "engagement"
    / "engagement_tfidf_logreg_v1.joblib"
)


class EngagementPredictor:
    """
    Loads the trained MODERAI engagement model and predicts
    the primary intent of a single comment.

    V1 operates in assisted mode:
    every prediction requires human review.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Engagement model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Loaded model does not support predict_proba()."
            )

        if not hasattr(self.model, "classes_"):
            raise TypeError(
                "Loaded model does not expose classes_."
            )

        self.classes = [str(label) for label in self.model.classes_]

        if len(self.classes) < 2:
            raise ValueError(
                "Engagement model must contain at least two classes."
            )

        print("Engagement model loaded successfully.")
        print(f"Model path: {self.model_path}")
        print(f"Supported intents: {self.classes}")

    def classify_comment(self, comment_text: str) -> dict[str, Any]:
        """
        Classify one comment and return its predicted intent,
        class probabilities, confidence, and review information.
        """

        if not isinstance(comment_text, str):
            raise TypeError("comment_text must be a string.")

        cleaned_text = comment_text.strip()

        if not cleaned_text:
            raise ValueError("Comment text cannot be empty.")

        # Step 1: Generate probabilities using the saved pipeline.
        probabilities = self.model.predict_proba(
            [cleaned_text]
        )[0]

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if len(probabilities) != len(self.classes):
            raise ValueError(
                "Probability count does not match model classes."
            )

        # Step 2: Identify the highest and second-highest probabilities.
        sorted_indices = np.argsort(probabilities)

        top_index = int(sorted_indices[-1])
        second_index = int(sorted_indices[-2])

        predicted_intent = self.classes[top_index]

        max_probability = float(probabilities[top_index])
        second_probability = float(probabilities[second_index])

        # Step 3: Calculate confidence margin.
        confidence_margin = (
            max_probability - second_probability
        )

        # Step 4: Return structured prediction.
        # V1 has no validated automatic acceptance threshold.
        return {
            "predicted_intent": predicted_intent,
            "max_probability": max_probability,
            "second_probability": second_probability,
            "confidence_margin": float(confidence_margin),
            "class_probabilities": {
                label: float(probability)
                for label, probability in zip(
                    self.classes,
                    probabilities
                )
            },
            "requires_review": True,
            "review_reason": (
                "V1 assisted mode: predictions require human "
                "confirmation because no reliable automatic "
                "acceptance threshold has been established."
            ),
            "model_version": "engagement_tfidf_logreg_v1"
        }


if __name__ == "__main__":
    predictor = EngagementPredictor()

    sample_comment = (
        "Amazing explanation! Can you share the source code?"
    )

    result = predictor.classify_comment(sample_comment)

    print("\nENGAGEMENT PREDICTION")
    print("-" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")
