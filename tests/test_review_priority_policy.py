
import pytest

from moderation.review.priority_policy import ReviewPriorityPolicy


def make_analysis(
    intent="POSITIVE",
    safety_signals=None
):
    return {
        "safety_analysis": {
            "qualified_review_signals": (
                safety_signals if safety_signals is not None else []
            )
        },
        "engagement_analysis": {
            "predicted_intent": intent
        }
    }


def test_qualified_safety_signal_gets_critical_priority():
    analysis = make_analysis(
        safety_signals=["toxicity"]
    )

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "CRITICAL"


def test_multiple_safety_signals_get_critical_priority():
    analysis = make_analysis(
        safety_signals=["toxicity", "insult"]
    )

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "CRITICAL"


@pytest.mark.parametrize(
    "intent",
    ["POSITIVE", "NEGATIVE", "CRITICAL", "QUESTION", "NEUTRAL"]
)
def test_valid_intents_get_normal_priority(intent):
    analysis = make_analysis(intent=intent)

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "NORMAL"


def test_unclear_intent_gets_high_priority():
    analysis = make_analysis(intent="UNCLEAR")

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "HIGH"


def test_invalid_analysis_gets_high_priority():
    result = ReviewPriorityPolicy.evaluate(None)

    assert result["priority"] == "HIGH"


def test_missing_safety_analysis_gets_high_priority():
    analysis = {
        "engagement_analysis": {
            "predicted_intent": "POSITIVE"
        }
    }

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "HIGH"


def test_missing_engagement_analysis_gets_high_priority():
    analysis = {
        "safety_analysis": {
            "qualified_review_signals": []
        }
    }

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "HIGH"


def test_invalid_safety_signals_get_high_priority():
    analysis = make_analysis()
    analysis["safety_analysis"]["qualified_review_signals"] = None

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "HIGH"


def test_invalid_intent_gets_high_priority():
    analysis = make_analysis(intent="UNKNOWN")

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "HIGH"


def test_safety_priority_overrides_unclear_intent():
    analysis = make_analysis(
        intent="UNCLEAR",
        safety_signals=["toxicity"]
    )

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert result["priority"] == "CRITICAL"


def test_every_result_contains_a_reason():
    analysis = make_analysis()

    result = ReviewPriorityPolicy.evaluate(analysis)

    assert isinstance(result["reason"], str)
    assert result["reason"].strip()
