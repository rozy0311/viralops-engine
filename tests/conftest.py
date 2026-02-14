"""
Shared test fixtures for ViralOps Engine tests.
Uses a temp SQLite DB so tests don't pollute the production database.
"""
import os
import sys
import pathlib
import tempfile

import pytest

# Ensure project root is on path
_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


@pytest.fixture(autouse=True, scope="session")
def _use_temp_db():
    """Point web.app.DB_PATH to a temporary file for the entire test session."""
    import web.app as webapp

    # Create a temp DB file that persists for the session
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix="viralops_test_")
    tmp.close()
    original_path = webapp.DB_PATH
    webapp.DB_PATH = tmp.name

    # Re-initialize tables on the clean temp DB
    webapp.init_db()

    yield

    webapp.DB_PATH = original_path
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
