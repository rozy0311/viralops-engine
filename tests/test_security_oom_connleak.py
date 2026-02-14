"""
Tests for v3.2 hardening:
- API key authentication middleware
- OOM prevention (deque caps on AlertManager, Dashboard, EngagementTracker)
- SQLite connection leak prevention (get_db_safe context manager)
"""
import os
import sys
import json
import pathlib
import sqlite3
from collections import deque
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

# ── Pre-mock langgraph (same as test_pipeline_api.py) ──
_mock_langgraph = MagicMock()
for mod in ("langgraph", "langgraph.graph", "langgraph.checkpoint",
            "langgraph.checkpoint.memory"):
    sys.modules.setdefault(mod, _mock_langgraph)

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import graph as _graph_module  # noqa: E402
from web.app import app, _alert_manager, get_db_safe

client = TestClient(app)


# ════════════════════════════════════════════════════════════════
# 1. API KEY AUTHENTICATION MIDDLEWARE
# ════════════════════════════════════════════════════════════════

class TestApiKeyAuth:
    """Test ApiKeyMiddleware on /api/ endpoints."""

    def test_dev_mode_no_key_allows_access(self):
        """When VIRALOPS_API_KEY is not set, all requests pass (dev mode)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VIRALOPS_API_KEY", None)
            r = client.get("/api/posts")
            assert r.status_code == 200

    def test_auth_required_returns_401_no_header(self):
        """When key is set but no header provided, return 401."""
        with patch.dict(os.environ, {"VIRALOPS_API_KEY": "test-secret-key-12345"}):
            r = client.get("/api/posts")
            assert r.status_code == 401
            body = r.json()
            assert body["success"] is False
            assert "Unauthorized" in body["error"]

    def test_auth_bearer_token_accepted(self):
        """Correct Bearer token should pass authentication."""
        with patch.dict(os.environ, {"VIRALOPS_API_KEY": "test-secret-key-12345"}):
            r = client.get(
                "/api/posts",
                headers={"Authorization": "Bearer test-secret-key-12345"},
            )
            assert r.status_code == 200

    def test_auth_x_api_key_header_accepted(self):
        """Correct X-API-Key header should pass authentication."""
        with patch.dict(os.environ, {"VIRALOPS_API_KEY": "test-secret-key-12345"}):
            r = client.get(
                "/api/posts",
                headers={"X-API-Key": "test-secret-key-12345"},
            )
            assert r.status_code == 200

    def test_auth_wrong_key_returns_401(self):
        """Wrong API key should return 401."""
        with patch.dict(os.environ, {"VIRALOPS_API_KEY": "correct-key"}):
            r = client.get(
                "/api/posts",
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert r.status_code == 401

    def test_health_endpoint_exempt_from_auth(self):
        """GET /api/health should always work without auth."""
        with patch.dict(os.environ, {"VIRALOPS_API_KEY": "test-secret-key-12345"}):
            r = client.get("/api/health")
            assert r.status_code in (200, 503)  # 503 if tasks not running

    def test_non_api_routes_exempt(self):
        """HTML pages (non /api/ routes) should not require auth."""
        with patch.dict(os.environ, {"VIRALOPS_API_KEY": "test-secret-key-12345"}):
            r = client.get("/compose")
            assert r.status_code in (200, 500)  # 500 if template missing, but NOT 401


# ════════════════════════════════════════════════════════════════
# 2. OOM PREVENTION — DEQUE CAPS
# ════════════════════════════════════════════════════════════════

class TestOomPrevention:
    """Test that in-memory collections are bounded."""

    def test_alert_manager_history_is_deque(self):
        """AlertManager._history should be a deque with maxlen."""
        assert isinstance(_alert_manager._history, deque)
        assert _alert_manager._history.maxlen is not None
        assert _alert_manager._history.maxlen <= 10_000

    def test_alert_manager_history_cap(self):
        """Adding beyond maxlen should evict oldest entries."""
        from monitoring.alerting import AlertManager
        mgr = AlertManager()
        maxlen = mgr._history.maxlen
        # Fill beyond cap
        for i in range(maxlen + 100):
            mgr._history.append({"event": i})
        assert len(mgr._history) == maxlen

    def test_dashboard_publish_history_bounded(self):
        """Dashboard._publish_history should be a deque."""
        from monitoring.dashboard import Dashboard
        dash = Dashboard()
        assert isinstance(dash._publish_history, deque)
        assert dash._publish_history.maxlen is not None
        assert dash._publish_history.maxlen <= 10_000

    def test_dashboard_error_history_bounded(self):
        """Dashboard._error_history should be a deque."""
        from monitoring.dashboard import Dashboard
        dash = Dashboard()
        assert isinstance(dash._error_history, deque)
        assert dash._error_history.maxlen is not None
        assert dash._error_history.maxlen <= 5_000

    def test_engagement_tracker_trim(self):
        """EngagementTracker should trim after MAX_PER_PLATFORM entries."""
        from monitoring.engagement_tracker import EngagementTracker
        tracker = EngagementTracker()
        cap = tracker.MAX_PER_PLATFORM
        assert cap > 0
        # Fill beyond cap
        for i in range(cap + 500):
            tracker.record("test_platform", post_id=f"post-{i}", likes=i)
        assert len(tracker._metrics["test_platform"]) <= cap


# ════════════════════════════════════════════════════════════════
# 3. SQLITE CONNECTION LEAK PREVENTION
# ════════════════════════════════════════════════════════════════

class TestConnectionSafety:
    """Test get_db_safe() context manager."""

    def test_get_db_safe_returns_connection(self):
        """get_db_safe should yield a working SQLite connection."""
        with get_db_safe() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1

    def test_get_db_safe_closes_on_normal_exit(self):
        """Connection should be closed after with-block exits."""
        conn_ref = None
        with get_db_safe() as conn:
            conn_ref = conn
            conn.execute("SELECT 1")
        # After exiting, the connection should be closed
        # Attempting to use it should raise
        with pytest.raises(Exception):
            conn_ref.execute("SELECT 1")

    def test_get_db_safe_closes_on_exception(self):
        """Connection should be closed even when exception occurs."""
        conn_ref = None
        with pytest.raises(ValueError):
            with get_db_safe() as conn:
                conn_ref = conn
                conn.execute("SELECT 1")
                raise ValueError("Simulated error")
        # Connection should still be closed
        with pytest.raises(Exception):
            conn_ref.execute("SELECT 1")

    def test_no_raw_get_db_in_endpoints(self):
        """Verify no 'conn = get_db()' remains in endpoint code (regression guard).

        Only init_db (which is wrapped in get_db_safe now) is allowed.
        This test reads the source to ensure no regressions.
        """
        import inspect
        from web import app as webapp

        source = inspect.getsource(webapp)
        # Count remaining raw get_db() calls (should only be the function def itself)
        import re
        raw_calls = re.findall(r'\bconn\s*=\s*get_db\(\)', source)
        # 0 raw calls expected (init_db now uses get_db_safe)
        assert len(raw_calls) == 0, (
            f"Found {len(raw_calls)} raw 'conn = get_db()' call(s) — "
            "all should use 'with get_db_safe() as conn:'"
        )

    def test_api_endpoints_work_with_safe_connections(self):
        """Basic smoke test: CRUD operations work with get_db_safe."""
        # Create
        r = client.post("/api/posts", json={
            "title": "ConnTest", "body": "testing", "platforms": ["tiktok"],
        })
        assert r.status_code == 200
        post_id = r.json()["post_id"]

        # Read
        r = client.get("/api/posts")
        assert r.status_code == 200
        titles = [p["title"] for p in r.json()]
        assert "ConnTest" in titles

        # Update
        r = client.put(f"/api/posts/{post_id}", json={"title": "ConnTestUpdated"})
        assert r.status_code == 200

        # Delete
        r = client.delete(f"/api/posts/{post_id}")
        assert r.status_code == 200

    def test_stats_endpoint_uses_safe_connection(self):
        """GET /api/stats should work (uses get_db_safe internally)."""
        r = client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "published" in data
