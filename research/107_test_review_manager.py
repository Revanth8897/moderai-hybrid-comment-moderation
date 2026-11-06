
from moderation.review.manager import ReviewManager


manager = ReviewManager()

sample_pipeline_result = {
    "comment": "Can you explain database indexing?",
    "analysis": {
        "requires_human_review": True,
        "engagement_analysis": {
            "predicted_intent": "QUESTION"
        },
        "safety_analysis": {
            "qualified_review_signals": []
        }
    },
    "routing": {
        "route": "HUMAN_REVIEW",
        "priority": "NORMAL"
    }
}

# 1. Add review
review = manager.add_to_queue(sample_pipeline_result)

print("\nCREATED REVIEW:")
print(review)

# 2. Retrieve pending reviews
pending = manager.get_pending_reviews()

print("\nPENDING REVIEW COUNT:")
print(len(pending))

# 3. Retrieve individual review
retrieved = manager.get_review(review["review_id"])

print("\nRETRIEVED REVIEW:")
print(retrieved["review_id"])

# 4. Update status
updated = manager.update_status(
    review["review_id"],
    "IN_REVIEW",
    "admin_001"
)

print("\nUPDATED STATUS:")
print(updated["review_status"])

print("\nREVIEW MANAGER TEST COMPLETED.")
