
from copy import deepcopy
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.app.database.models import Review
from backend.app.database.session import SessionLocal

from moderation.response.controller import ResponseGenerationController

from .priority_policy import ReviewPriorityPolicy


class ReviewManager:
    """
    Manages human review records using MySQL and SQLAlchemy.

    Approved response drafts are generated only after
    explicit human approval. No publishing is performed.
    """

    REVIEW_STATUSES = {
        "PENDING",
        "IN_REVIEW",
        "APPROVED",
        "REJECTED",
        "ESCALATED",
        "COMPLETED",
    }

    REVIEW_DECISIONS = {
        "APPROVE_FOR_DRAFT",
        "NO_RESPONSE",
        "REJECT",
        "ESCALATE",
    }

    ALLOWED_TRANSITIONS = {
        "PENDING": {"IN_REVIEW"},
        "IN_REVIEW": {
            "APPROVED",
            "REJECTED",
            "ESCALATED",
            "COMPLETED",
        },
        "APPROVED": {"COMPLETED"},
        "REJECTED": set(),
        "ESCALATED": set(),
        "COMPLETED": set(),
    }

    def __init__(
        self,
        session_factory: sessionmaker = None,
        response_controller: ResponseGenerationController = None,
    ):
        self._session_factory = session_factory or SessionLocal

        self.response_controller = (
            response_controller
            if response_controller is not None
            else ResponseGenerationController()
        )

        print("MODERAI MySQL Review Manager initialized.")

    @staticmethod
    def _format_datetime(value: datetime | None) -> str | None:
        if value is None:
            return None

        return value.isoformat()

    @staticmethod
    def _to_dict(review: Review) -> dict:
        """Convert a SQLAlchemy Review object into a dictionary."""

        analysis = deepcopy(review.analysis_data)
        routing = deepcopy(review.routing_data)

        priority_result = ReviewPriorityPolicy.evaluate(analysis)

        return {
            "review_id": review.review_id,
            "comment": review.comment,
            "analysis": analysis,
            "routing": routing,
            "review_priority": priority_result["priority"],
            "priority_reason": priority_result["reason"],
            "review_status": review.review_status,
            "reviewer_id": review.reviewer_id,
            "review_decision": review.review_decision,
            "decision_reason": review.decision_reason,
            "created_at": ReviewManager._format_datetime(
                review.created_at
            ),
            "updated_at": ReviewManager._format_datetime(
                review.updated_at
            ),
            "automatic_action_allowed": review.automatic_action_allowed,
        }

    def add_to_queue(self, pipeline_result: dict) -> dict:
        """Add a comment requiring human review to MySQL."""

        if not isinstance(pipeline_result, dict):
            raise TypeError("Pipeline result must be a dictionary.")

        analysis = pipeline_result.get("analysis")
        routing = pipeline_result.get("routing")

        if not isinstance(analysis, dict):
            raise ValueError("Valid analysis is required.")

        if not isinstance(routing, dict):
            raise ValueError("Valid routing information is required.")

        if analysis.get("requires_human_review") is not True:
            raise ValueError("Comment does not require human review.")

        comment = pipeline_result.get("comment")

        if not isinstance(comment, str) or not comment.strip():
            raise ValueError("A valid comment is required.")

        review = Review(
            review_id=str(uuid4()),
            comment=comment,
            analysis_data=deepcopy(analysis),
            routing_data=deepcopy(routing),
            review_status="PENDING",
            reviewer_id=None,
            review_decision=None,
            decision_reason=None,
            automatic_action_allowed=False,
        )

        with self._session_factory() as db:
            try:
                db.add(review)
                db.commit()
                db.refresh(review)

                return self._to_dict(review)

            except Exception:
                db.rollback()
                raise

    def get_pending_reviews(self) -> list:
        """Return pending reviews ordered by priority and FIFO."""

        priority_order = {
            "CRITICAL": 1,
            "HIGH": 2,
            "NORMAL": 3,
        }

        with self._session_factory() as db:
            statement = (
                select(Review)
                .where(Review.review_status == "PENDING")
                .order_by(
                    Review.created_at.asc(),
                    Review.id.asc(),
                )
            )

            reviews = db.scalars(statement).all()

            pending_reviews = [
                self._to_dict(review)
                for review in reviews
            ]

        return sorted(
            pending_reviews,
            key=lambda review: priority_order.get(
                review.get("review_priority"),
                99,
            ),
        )

    def get_review(self, review_id: str) -> dict:
        """Retrieve an individual review from MySQL."""

        if not isinstance(review_id, str):
            raise TypeError("Review ID must be a string.")

        with self._session_factory() as db:
            statement = select(Review).where(
                Review.review_id == review_id
            )

            review = db.scalar(statement)

            if review is None:
                raise ValueError("Review record not found.")

            return self._to_dict(review)

    def update_status(
        self,
        review_id: str,
        new_status: str,
        reviewer_id: str,
    ) -> dict:
        """
        Update a review status using allowed transitions.

        Approval must go through decide_review() to enforce
        safety validation and draft generation.
        """

        if not isinstance(review_id, str):
            raise TypeError("Review ID must be a string.")

        if not isinstance(new_status, str):
            raise TypeError("Review status must be a string.")

        if new_status not in self.REVIEW_STATUSES:
            raise ValueError("Invalid review status.")

        if new_status == "APPROVED":
            raise ValueError(
                "Use decide_review() to approve a review."
            )

        if (
            not isinstance(reviewer_id, str)
            or not reviewer_id.strip()
        ):
            raise ValueError("Valid reviewer ID is required.")

        with self._session_factory() as db:
            try:
                statement = (
                    select(Review)
                    .where(Review.review_id == review_id)
                    .with_for_update()
                )

                review = db.scalar(statement)

                if review is None:
                    raise ValueError("Review record not found.")

                current_status = review.review_status

                if (
                    new_status
                    not in self.ALLOWED_TRANSITIONS[current_status]
                ):
                    raise ValueError(
                        f"Invalid status transition: "
                        f"{current_status} -> {new_status}"
                    )

                review.review_status = new_status
                review.reviewer_id = reviewer_id

                db.commit()
                db.refresh(review)

                return self._to_dict(review)

            except Exception:
                db.rollback()
                raise

    def decide_review(
        self,
        review_id: str,
        decision: str,
        reviewer_id: str,
        reason: str = "",
    ) -> dict:
        """
        Record a human review decision.

        APPROVE_FOR_DRAFT:
        1. Validates review state.
        2. Validates safety signals.
        3. Generates a response draft.
        4. Stores the draft in analysis_data.
        5. Commits approval and draft together.

        No response is published or sent.
        """

        if not isinstance(review_id, str):
            raise TypeError("Review ID must be a string.")

        if not isinstance(decision, str):
            raise TypeError("Decision must be a string.")

        if decision not in self.REVIEW_DECISIONS:
            raise ValueError("Invalid reviewer decision.")

        if (
            not isinstance(reviewer_id, str)
            or not reviewer_id.strip()
        ):
            raise ValueError("Valid reviewer ID is required.")

        if not isinstance(reason, str):
            raise TypeError("Decision reason must be a string.")

        with self._session_factory() as db:
            try:
                statement = (
                    select(Review)
                    .where(Review.review_id == review_id)
                    .with_for_update()
                )

                review = db.scalar(statement)

                if review is None:
                    raise ValueError("Review record not found.")

                if review.review_status != "IN_REVIEW":
                    raise ValueError(
                        "Review must be IN_REVIEW before a decision."
                    )

                safety_analysis = review.analysis_data.get(
                    "safety_analysis"
                )

                if not isinstance(safety_analysis, dict):
                    raise ValueError(
                        "Valid safety analysis is required."
                    )

                qualified_signals = safety_analysis.get(
                    "qualified_review_signals"
                )

                if not isinstance(qualified_signals, list):
                    raise ValueError(
                        "Valid qualified safety signals are required."
                    )

                # Qualified safety signals block draft approval.
                if (
                    decision == "APPROVE_FOR_DRAFT"
                    and qualified_signals
                ):
                    raise ValueError(
                        "Cannot approve response generation while "
                        "qualified safety signals are present. "
                        "Escalate for safety resolution."
                    )

                if decision == "APPROVE_FOR_DRAFT":
                    analysis_data = deepcopy(review.analysis_data)

                    # Ensure the controller receives the original comment.
                    analysis_data["comment"] = review.comment

                    pipeline_result = {
                        "comment": review.comment,
                        "analysis": analysis_data,
                        "routing": deepcopy(review.routing_data),
                    }

                    generated_draft = (
                        self.response_controller
                        .generate_after_human_approval(
                            pipeline_result=pipeline_result,
                            approved=True,
                        )
                    )

                    if (
                        not isinstance(generated_draft, dict)
                        or generated_draft.get("status") == "NO_DRAFT"
                        or not isinstance(
                            generated_draft.get("response"),
                            dict,
                        )
                    ):
                        raise ValueError(
                            "Draft generation was blocked. "
                            "Review cannot be approved for draft generation."
                        )

                    # Assign a new dictionary so SQLAlchemy detects
                    # the JSON column modification.
                    updated_analysis = deepcopy(review.analysis_data)
                    updated_analysis["generated_draft"] = generated_draft
                    review.analysis_data = updated_analysis

                decision_status = {
                    "APPROVE_FOR_DRAFT": "APPROVED",
                    "NO_RESPONSE": "COMPLETED",
                    "REJECT": "REJECTED",
                    "ESCALATE": "ESCALATED",
                }

                review.review_status = decision_status[decision]
                review.reviewer_id = reviewer_id
                review.review_decision = decision
                review.decision_reason = reason.strip()

                # Draft and approval are committed together.
                db.commit()
                db.refresh(review)

                return self._to_dict(review)

            except Exception:
                db.rollback()
                raise
