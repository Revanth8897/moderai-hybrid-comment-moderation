
"""create reviews table

Revision ID: efdda3d65f38
Revises:
Create Date: 2026-11-04 15:16:52.958680
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# Revision identifiers
revision: str = "efdda3d65f38"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the reviews table."""

    op.create_table(
        "reviews",

        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),

        sa.Column(
            "review_id",
            sa.String(length=36),
            nullable=False,
        ),

        sa.Column(
            "comment",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "review_status",
            sa.String(length=30),
            nullable=False,
        ),

        sa.Column(
            "reviewer_id",
            sa.String(length=100),
            nullable=True,
        ),

        sa.Column(
            "review_decision",
            sa.String(length=30),
            nullable=True,
        ),

        sa.Column(
            "decision_reason",
            sa.Text(),
            nullable=True,
        ),

        sa.Column(
            "analysis_data",
            sa.JSON(),
            nullable=False,
        ),

        sa.Column(
            "routing_data",
            sa.JSON(),
            nullable=False,
        ),

        sa.Column(
            "automatic_action_allowed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),

        sa.PrimaryKeyConstraint("id"),
    )

    # Unique index for review ID
    op.create_index(
        "ix_reviews_review_id",
        "reviews",
        ["review_id"],
        unique=True,
    )

    # Index for review queue filtering and sorting
    op.create_index(
        "ix_reviews_status_created",
        "reviews",
        ["review_status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the reviews table."""

    op.drop_index(
        "ix_reviews_status_created",
        table_name="reviews",
    )

    op.drop_index(
        "ix_reviews_review_id",
        table_name="reviews",
    )

    op.drop_table("reviews")
