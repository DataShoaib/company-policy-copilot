"""initial users and refresh_tokens schema

Revision ID: f010de76e6a1
Revises: 
Create Date: 2026-08-25 14:42:22.795774

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f010de76e6a1'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the initial tables.

    Adopted from an existing DB that was previously created by
    ``Base.metadata.create_all`` — deployments that already have these tables
    run ``alembic stamp head`` (see README). Fresh databases run
    ``alembic upgrade head`` and get the real DDL here.
    """
    op.create_table(
        "users",
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("full_name", sa.String(length=200), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("username"),
    )
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True)
    op.create_index("ix_refresh_tokens_username", "refresh_tokens", ["username"], unique=False)


def downgrade() -> None:
    """Drop the schema."""
    op.drop_index("ix_refresh_tokens_username", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_table("users")
