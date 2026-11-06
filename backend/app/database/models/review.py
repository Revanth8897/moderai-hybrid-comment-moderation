from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class Review(Base):
    __tablename__ = "reviews"

    __table_args__ = (
        Index("ix_reviews_status_created", "review_status", "created_at"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    review_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        index=True,
    )

    comment: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    review_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="PENDING",
    )

    reviewer_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    review_decision: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    decision_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    analysis_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    routing_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    automatic_action_allowed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )