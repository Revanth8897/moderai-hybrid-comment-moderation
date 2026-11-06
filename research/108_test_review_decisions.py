
from moderation.review.manager import ReviewManager


manager = ReviewManager()


def create_review(safety_signals=None):

    if safety_signals is None:
        safety_signals = []

    pipeline_result = {
        "comment": "Can you explain database indexing?",
        "analysis": {
            "requires_human_review": True,
            "engagement_analysis": {
                "predicted_intent": "QUESTION"
            },
            "safety_analysis": {
                "qualified_review_signals": safety_signals
            }
        },
        "routing": {
            "route": "HUMAN_REVIEW",
            "priority": "NORMAL"
        }
    }

    review = manager.add_to_queue(pipeline_result)

    manager.update_status(
        review["review_id"],
        "IN_REVIEW",
        "admin_001"
    )

    return review["review_id"]


# Test 1: Approve for draft generation.
review_id = create_review()

result = manager.decide_review(
    review_id,
    "APPROVE_FOR_DRAFT",
    "admin_001",
    "Comment is safe and deserves an answer."
)

print("APPROVAL:", result["review_status"])


# Test 2: No response.
review_id = create_review()

result = manager.decide_review(
    review_id,
    "NO_RESPONSE",
    "admin_001",
    "No reply is necessary."
)

print("NO RESPONSE:", result["review_status"])


# Test 3: Reject.
review_id = create_review()

result = manager.decide_review(
    review_id,
    "REJECT",
    "admin_001",
    "Proposed action rejected."
)

print("REJECTION:", result["review_status"])


# Test 4: Escalate.
review_id = create_review()

result = manager.decide_review(
    review_id,
    "ESCALATE",
    "admin_001",
    "Requires additional review."
)

print("ESCALATION:", result["review_status"])


# Test 5: Safety protection.
review_id = create_review(["toxicity"])

try:
    manager.decide_review(
        review_id,
        "APPROVE_FOR_DRAFT",
        "admin_001",
        "Trying to approve a safety-flagged comment."
    )

except ValueError as error:
    print("SAFETY PROTECTION:", error)


print("\nALL REVIEW DECISION SMOKE TESTS COMPLETED.")
