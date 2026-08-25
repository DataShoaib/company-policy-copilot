"""Shared pytest fixtures.

DB table creation + demo-user seeding used to run at `import users` time as a
module side effect. That made importing the package write to the DB even for
tests that only touch JWT or routing. It's now done here once per session so
auth tests get the demo users they rely on, without spurious writes on import.
"""

import os

# The test-suite is self-contained and doesn't provision a Redis, so disable
# the (fail-closed) rate limiter before pydantic-settings builds the Settings
# object on the first hr_rag import.
os.environ["RATE_LIMIT_ENABLED"] = "false"

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_db_seeded():
    from hr_rag.api.core.database import init_db
    from hr_rag.api.core.users import seed_demo_users

    init_db()
    seed_demo_users()
    yield