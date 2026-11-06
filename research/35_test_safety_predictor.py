
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from moderation.toxicity.predictor import SafetyPredictor


def main():

    print("=" * 65)
    print("MODERAI SAFETY PREDICTOR TEST")
    print("=" * 65)

    predictor = SafetyPredictor()

    test_comments = [
        "Thank you for sharing this wonderful information.",

        "This is a stupid idea and you clearly don't understand it.",

        "I disagree with your opinion, but I respect your point of view."
    ]

    for index, comment in enumerate(test_comments, start=1):

        print("\n" + "=" * 65)
        print(f"TEST COMMENT {index}")
        print("=" * 65)

        result = predictor.predict(comment)

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )

    print("\n" + "=" * 65)
    print("SAFETY PREDICTOR TEST COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
