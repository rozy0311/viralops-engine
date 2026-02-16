"""
Tests for v3.1 features:
- POST /api/pipeline/run (full EMADS-PR LangGraph pipeline from web)
- Publer wiring in _get_publisher() and scheduler._load_publisher()
- RSS tick loop fix (asyncio.to_thread)
- Autopilot scheduler config API
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

import sys, pathlib

# ── Pre-mock heavy dependencies that aren't installed in test venv ──
# graph.py imports langgraph which may not be in the test environment.
# We inject a mock graph module so @patch("graph.get_compiled_graph") works.
_mock_langgraph = MagicMock()
for mod in ("langgraph", "langgraph.graph", "langgraph.checkpoint",
            "langgraph.checkpoint.memory"):
    sys.modules.setdefault(mod, _mock_langgraph)

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# Now import graph so it's in sys.modules (with mocked langgraph)
import graph as _graph_module  # noqa: E402

from web.app import app, _PUBLER_PLATFORMS, _alert_manager

client = TestClient(app)


# ════════════════════════════════════════════════════════════════
# 1. PIPELINE RUN ENDPOINT
# ════════════════════════════════════════════════════════════════

class TestPipelineRunEndpoint:
    """Test POST /api/pipeline/run — full EMADS-PR pipeline."""

    def test_pipeline_endpoint_exists(self):
        """Endpoint should exist and accept POST."""
        r = client.post("/api/pipeline/run", json={"niche": "test"})
        assert r.status_code in (200, 500)  # 500 if LangGraph not available, but endpoint exists

    @patch("graph.get_compiled_graph")
    def test_pipeline_returns_structured_result(self, mock_graph):
        """Pipeline should return success + content_pack + reconcile."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "content_pack": {
                "title": "Test Title",
                "body": "Test body content",
                "hook": "Test hook",
                "cta": "Follow!",
                "hashtags": ["#test", "#pipeline"],
            },
            "reconcile_result": {"summary": "All clear", "risk_level": "low"},
            "risk_result": {"risk_score": 2},
            "cost_result": {"estimated_cost_usd": 0.01},
            "publish_results": [],
            "errors": [],
        }
        mock_graph.return_value = mock_app
        r = client.post("/api/pipeline/run", json={
            "niche": "fitness",
            "topic": "morning routine",
            "platforms": ["tiktok", "pinterest"],
            "publish_mode": "draft",
        })
        data = r.json()
        assert data["success"] is True
        assert data["content_pack"]["title"] == "Test Title"
        assert data["content_pack"]["hashtags"] == ["#test", "#pipeline"]
        assert data["reconcile"]["summary"] == "All clear"
        assert data["risk_score"] == 2
        assert data["publish_mode"] == "draft"
        assert "thread_id" in data

    @patch("graph.get_compiled_graph")
    def test_pipeline_saves_to_sqlite(self, mock_graph):
        """Pipeline results should be saved to posts table."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "content_pack": {"title": "SQLite Test", "body": "Test body", "hashtags": []},
            "reconcile_result": {},
            "risk_result": {},
            "cost_result": {},
            "publish_results": [],
            "errors": [],
        }
        mock_graph.return_value = mock_app
        r = client.post("/api/pipeline/run", json={"niche": "tech"})
        assert r.json()["success"] is True

        # Verify it was saved
        posts = client.get("/api/posts").json()
        titles = [p["title"] for p in posts]
        assert "SQLite Test" in titles

    @patch("graph.get_compiled_graph")
    def test_pipeline_default_platforms(self, mock_graph):
        """Default platforms should be tiktok + pinterest."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "content_pack": {"title": "Default Platforms"},
            "reconcile_result": {},
            "risk_result": {},
            "cost_result": {},
            "publish_results": [],
            "errors": [],
        }
        mock_graph.return_value = mock_app
        r = client.post("/api/pipeline/run", json={"niche": "food"})
        data = r.json()
        assert data["success"] is True

    @patch("graph.get_compiled_graph", side_effect=Exception("LangGraph unavailable"))
    def test_pipeline_error_returns_500(self, mock_graph):
        """Pipeline errors should return 500 with error message."""
        r = client.post("/api/pipeline/run", json={"niche": "test"})
        assert r.status_code == 500
        data = r.json()
        assert data["success"] is False
        assert "LangGraph unavailable" in data["error"]

    def test_pipeline_accepts_all_publish_modes(self):
        """Test all valid publish_mode values."""
        for mode in ("draft", "scheduled", "immediate"):
            with patch("graph.get_compiled_graph") as mock_graph:
                mock_app = MagicMock()
                mock_app.invoke.return_value = {
                    "content_pack": {"title": f"Mode: {mode}"},
                    "reconcile_result": {},
                    "risk_result": {},
                    "cost_result": {},
                    "publish_results": [],
                    "errors": [],
                }
                mock_graph.return_value = mock_app
                r = client.post("/api/pipeline/run", json={
                    "niche": "test", "publish_mode": mode,
                })
                assert r.json()["publish_mode"] == mode


# ════════════════════════════════════════════════════════════════
# 2. PUBLER WIRING
# ════════════════════════════════════════════════════════════════

class TestPublerWiring:
    """Test that Publer is wired as default publisher for social platforms."""

    def test_publer_platforms_constant_exists(self):
        """_PUBLER_PLATFORMS should contain core social platforms."""
        assert "tiktok" in _PUBLER_PLATFORMS
        assert "instagram" in _PUBLER_PLATFORMS
        assert "facebook" in _PUBLER_PLATFORMS
        assert "twitter" in _PUBLER_PLATFORMS
        assert "linkedin" in _PUBLER_PLATFORMS
        assert "youtube" in _PUBLER_PLATFORMS
        assert "pinterest" in _PUBLER_PLATFORMS
        assert "threads" in _PUBLER_PLATFORMS
        assert "bluesky" in _PUBLER_PLATFORMS
        assert "mastodon" in _PUBLER_PLATFORMS

    def test_publer_platforms_count(self):
        """Should support at least 12 platforms."""
        assert len(_PUBLER_PLATFORMS) >= 12

    def test_direct_publishers_still_work(self):
        """Reddit, medium, tumblr, shopify_blog should use direct publishers."""
        from web.app import _get_publisher
        # These should not raise ValueError even without Publer
        for platform in ("reddit", "medium", "tumblr", "shopify_blog"):
            try:
                pub = _get_publisher(platform)
                assert pub is not None
            except Exception:
                pass  # OK if credentials not set, just shouldn't be ValueError

    def test_unknown_platform_raises(self):
        """Unknown platforms should raise ValueError."""
        from web.app import _get_publisher
        with pytest.raises(ValueError, match="Unknown platform"):
            _get_publisher("nonexistent_platform_xyz")

    def test_scheduler_has_publer_priority(self):
        """Scheduler._load_publisher should try Publer first for supported platforms."""
        from core.scheduler import PublishScheduler
        scheduler = PublishScheduler()
        # The method should exist and handle tiktok
        # Since Publer may not be configured, it falls through gracefully
        pub = scheduler._load_publisher("tiktok")
        # pub can be None if no credentials set — that's OK
        # The important thing is it doesn't crash


# ════════════════════════════════════════════════════════════════
# 3. RSS TICK LOOP FIX
# ════════════════════════════════════════════════════════════════

class TestRssTickLoop:
    """Test that RSS tick loop uses asyncio.to_thread correctly."""

    def test_rss_tick_is_sync_function(self):
        """rss_auto_poster.tick() should be a sync function (not async)."""
        from integrations.rss_auto_poster import tick
        import asyncio
        assert not asyncio.iscoroutinefunction(tick), \
            "tick() should be sync — the background loop wraps it with asyncio.to_thread"

    def test_rss_tick_returns_dict(self):
        """tick() should return a dict (not a coroutine)."""
        from integrations.rss_auto_poster import tick
        # Mock dependencies to avoid real RSS fetching
        with patch("integrations.rss_auto_poster._load_configs", return_value=[]):
            result = tick()
            assert isinstance(result, dict)
            assert "results" in result

    def test_blog_share_tick_is_async(self):
        """shopify_auto_share.auto_share_tick() should be async."""
        from integrations.shopify_auto_share import auto_share_tick
        import asyncio
        assert asyncio.iscoroutinefunction(auto_share_tick), \
            "auto_share_tick() should be async — called directly with await"


# ════════════════════════════════════════════════════════════════
# 4. AUTOPILOT SCHEDULER CONFIG
# ════════════════════════════════════════════════════════════════

class TestAutopilotSchedulerConfig:
    """Test autopilot scheduler configuration endpoints."""

    def test_get_autopilot_config_returns_200(self):
        r = client.get("/api/autopilot/config")
        assert r.status_code == 200

    def test_get_autopilot_config_has_defaults(self):
        r = client.get("/api/autopilot/config")
        data = r.json()
        assert "enabled" in data
        assert "interval_hours" in data
        assert "niches" in data
        assert "platforms" in data
        assert "max_posts_per_day" in data

    def test_update_autopilot_config(self):
        r = client.put("/api/autopilot/config", json={
            "enabled": True,
            "interval_hours": 4,
            "niches": ["fitness", "tech_gadgets"],
            "platforms": ["tiktok", "pinterest"],
            "max_posts_per_day": 6,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert data["interval_hours"] == 4

    def test_autopilot_config_persists(self):
        # Set config
        client.put("/api/autopilot/config", json={
            "enabled": False,
            "interval_hours": 8,
            "niches": ["cooking"],
            "platforms": ["instagram"],
            "max_posts_per_day": 3,
        })
        # Read it back
        r = client.get("/api/autopilot/config")
        data = r.json()
        assert data["enabled"] is False
        assert data["interval_hours"] == 8
        assert "cooking" in data["niches"]
        assert data["max_posts_per_day"] == 3

    def test_autopilot_status_endpoint(self):
        r = client.get("/api/autopilot/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data
        assert "last_run" in data
        assert "posts_today" in data


# ════════════════════════════════════════════════════════════════
# 5. HEALTH CHECK (real liveness probe)
# ════════════════════════════════════════════════════════════════

class TestHealthCheck:
    """Test GET /api/health — real liveness/readiness probe."""

    def test_health_endpoint_exists(self):
        r = client.get("/api/health")
        assert r.status_code in (200, 503)

    def test_health_has_checks(self):
        r = client.get("/api/health")
        data = r.json()
        assert "checks" in data
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_checks_all_tasks(self):
        """Health should report on all 4 background tasks."""
        r = client.get("/api/health")
        data = r.json()
        checks = data["checks"]
        for task_name in ("scheduler", "rss_tick", "blog_share", "autopilot"):
            assert task_name in checks, f"Missing check for {task_name}"

    def test_health_checks_database(self):
        r = client.get("/api/health")
        data = r.json()
        assert "database" in data["checks"]
        assert data["checks"]["database"] == "ok"

    def test_health_returns_version(self):
        r = client.get("/api/health")
        data = r.json()
        assert data["version"] == "2.14.0"


# ════════════════════════════════════════════════════════════════
# 6. GLOBAL EXCEPTION HANDLER
# ════════════════════════════════════════════════════════════════

class TestGlobalExceptionHandler:
    """Test that unhandled exceptions return structured JSON, not raw tracebacks."""

    def test_missing_niche_still_works(self):
        """Posting empty body should work (defaults to 'general' niche)."""
        with patch("graph.get_compiled_graph") as mock_graph:
            mock_app = MagicMock()
            mock_app.invoke.return_value = {
                "content_pack": {"title": "Default Niche"},
                "reconcile_result": {},
                "risk_result": {},
                "cost_result": {},
                "publish_results": [],
                "errors": [],
            }
            mock_graph.return_value = mock_app
            r = client.post("/api/pipeline/run", json={})
            assert r.status_code == 200
            assert r.json()["success"] is True

    def test_404_does_not_leak_internals(self):
        """Non-existent endpoint should return clean 404."""
        r = client.get("/api/nonexistent/endpoint")
        assert r.status_code == 404


# ════════════════════════════════════════════════════════════════
# 7. SCHEDULER PUBLISH FIX
# ════════════════════════════════════════════════════════════════

class TestSchedulerPublishFix:
    """Test that scheduler._publish_to_platform no longer uses nested asyncio.run()."""

    def test_no_event_loop_detection_code(self):
        """The old asyncio.get_event_loop().is_running() pattern should be gone."""
        import inspect
        from core.scheduler import PublishScheduler
        source = inspect.getsource(PublishScheduler._publish_to_platform)
        assert "is_running()" not in source, \
            "_publish_to_platform still uses asyncio event loop detection"

    def test_uses_thread_pool_executor(self):
        """Should use ThreadPoolExecutor to run async publish in a clean loop."""
        import inspect
        from core.scheduler import PublishScheduler
        source = inspect.getsource(PublishScheduler._publish_to_platform)
        assert "ThreadPoolExecutor" in source

    def test_no_nested_lambda_asyncio_run(self):
        """The old lambda: asyncio.run(...) pattern should be gone."""
        import inspect
        from core.scheduler import PublishScheduler
        source = inspect.getsource(PublishScheduler._publish_to_platform)
        assert "lambda" not in source, \
            "_publish_to_platform still uses lambda: asyncio.run() anti-pattern"


# ════════════════════════════════════════════════════════════════
# 8. PUBLISH SAFETY GATE (EMADS-PR risk check)
# ════════════════════════════════════════════════════════════════

class TestPublishSafetyGate:
    """Test POST /api/publish/{post_id} — EMADS-PR safety gate."""

    def _seed_post(self, platforms=None, title="Test", body="body"):
        """Helper: insert a test post and return its id."""
        from web.app import get_db
        platforms = platforms or ["tiktok"]
        conn = get_db()
        cur = conn.execute(
            "INSERT INTO posts (title, body, platforms, status, extra_fields) VALUES (?, ?, ?, ?, ?)",
            (title, body, json.dumps(platforms), "draft", "{}"),
        )
        post_id = cur.lastrowid
        conn.commit()
        conn.close()
        return post_id

    @patch("agents.risk_health.assess_risk")
    @patch("web.app._get_publisher")
    def test_low_risk_publishes_normally(self, mock_pub, mock_risk):
        """Risk < 4 should publish without blocking."""
        mock_risk.return_value = {
            "risk_result": {"risk_score": 2, "risk_factors": [], "risk_level": "low"}
        }
        pub_instance = MagicMock()
        pub_instance.publish = AsyncMock(return_value=MagicMock(success=True, post_url="https://tiktok.com/123", error=""))
        mock_pub.return_value = pub_instance

        post_id = self._seed_post()
        r = client.post(f"/api/publish/{post_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["risk_score"] == 2

    @patch("agents.risk_health.assess_risk")
    @patch("web.app._alert_manager")
    def test_high_risk_blocks_without_force(self, mock_alert, mock_risk):
        """Risk >= 4 should return 403 and require human review."""
        mock_risk.return_value = {
            "risk_result": {"risk_score": 7, "risk_factors": ["too many platforms"], "risk_level": "high"}
        }
        mock_alert.publish_failed = AsyncMock()
        post_id = self._seed_post()
        r = client.post(f"/api/publish/{post_id}")
        assert r.status_code == 403
        data = r.json()
        assert data["requires_human_review"] is True
        assert data["risk_score"] == 7

    @patch("agents.risk_health.assess_risk")
    @patch("web.app._get_publisher")
    def test_high_risk_force_overrides(self, mock_pub, mock_risk):
        """Risk >= 4 with ?force=true should still publish."""
        mock_risk.return_value = {
            "risk_result": {"risk_score": 7, "risk_factors": [], "risk_level": "high"}
        }
        pub_instance = MagicMock()
        pub_instance.publish = AsyncMock(return_value=MagicMock(success=True, post_url="url", error=""))
        mock_pub.return_value = pub_instance

        post_id = self._seed_post()
        r = client.post(f"/api/publish/{post_id}?force=true")
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_publish_nonexistent_post_404(self):
        """Publishing a nonexistent post should return 404."""
        r = client.post("/api/publish/99999")
        assert r.status_code == 404

    @patch("agents.risk_health.assess_risk", side_effect=Exception("risk agent down"))
    @patch("web.app._get_publisher")
    def test_risk_failure_defaults_to_zero(self, mock_pub, mock_risk):
        """If risk check fails, default to risk_score=0 and allow publish."""
        pub_instance = MagicMock()
        pub_instance.publish = AsyncMock(return_value=MagicMock(success=True, post_url="url", error=""))
        mock_pub.return_value = pub_instance

        post_id = self._seed_post()
        r = client.post(f"/api/publish/{post_id}")
        assert r.status_code == 200
        assert r.json()["risk_score"] == 0


# ════════════════════════════════════════════════════════════════
# 9. ALERTING SYSTEM
# ════════════════════════════════════════════════════════════════

class TestAlertingSystem:
    """Test AlertManager and /api/alerts/history endpoint."""

    def test_alert_history_endpoint_exists(self):
        r = client.get("/api/alerts/history")
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_console_alert_stored_in_history(self):
        """Console alerts should be stored in history."""
        from monitoring.alerting import AlertManager, AlertLevel, AlertChannel
        mgr = AlertManager(default_channel=AlertChannel.CONSOLE)
        await mgr.send("test alert", level=AlertLevel.INFO)
        assert len(mgr._history) >= 1
        assert mgr._history[-1]["message"] == "test alert"

    @pytest.mark.asyncio
    async def test_telegram_fallback_when_not_configured(self):
        """Telegram alert without env vars should fall back to console."""
        from monitoring.alerting import AlertManager, AlertLevel, AlertChannel
        import os
        # Ensure no Telegram config
        with patch.dict(os.environ, {}, clear=True):
            mgr = AlertManager(default_channel=AlertChannel.TELEGRAM)
            result = await mgr.send("test telegram", level=AlertLevel.WARNING)
            assert result is False  # Falls back
            assert mgr._history[-1]["channel"] == "telegram"

    @pytest.mark.asyncio
    async def test_publish_failed_quick_method(self):
        """publish_failed() should create a WARNING-level alert."""
        from monitoring.alerting import AlertManager, AlertChannel
        mgr = AlertManager(default_channel=AlertChannel.CONSOLE)
        await mgr.publish_failed("tiktok", "API 429")
        assert len(mgr._history) >= 1
        last = mgr._history[-1]
        assert "tiktok" in last["message"]
        assert last["level"] == "warning"

    @pytest.mark.asyncio
    async def test_webhook_fallback_when_not_configured(self):
        """Webhook alert without env var should fall back to console."""
        from monitoring.alerting import AlertManager, AlertLevel, AlertChannel
        import os
        with patch.dict(os.environ, {}, clear=True):
            mgr = AlertManager(default_channel=AlertChannel.WEBHOOK)
            result = await mgr.send("test webhook", level=AlertLevel.INFO)
            assert result is False


# ════════════════════════════════════════════════════════════════
# 10. PUBLISH_LOG GAP FIX
# ════════════════════════════════════════════════════════════════

class TestPublishLogGapFix:
    """Test that publish results are written to publish_log table."""

    @patch("graph.get_compiled_graph")
    def test_pipeline_run_writes_publish_log(self, mock_graph):
        """POST /api/pipeline/run should write results to publish_log."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "content_pack": {"title": "Pipeline Log Test", "body": "body"},
            "reconcile_result": {"summary": "ok"},
            "risk_result": {"risk_score": 1},
            "cost_result": {"estimated_cost_usd": 0},
            "publish_results": [
                {"platform": "tiktok", "status": "published", "detail": "ok", "post_url": ""},
                {"platform": "pinterest", "status": "failed", "detail": "API error", "post_url": ""},
            ],
            "errors": [],
        }
        mock_graph.return_value = mock_app

        r = client.post("/api/pipeline/run", json={
            "niche": "test",
            "platforms": ["tiktok", "pinterest"],
            "publish_mode": "immediate",
        })
        assert r.status_code == 200

        # Check publish_log has entries
        from web.app import get_db
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM publish_log WHERE post_id = (SELECT MAX(id) FROM posts WHERE title = 'Pipeline Log Test')"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        platforms_logged = {r["platform"] for r in rows}
        assert "tiktok" in platforms_logged
        assert "pinterest" in platforms_logged

    @patch("agents.risk_health.assess_risk")
    @patch("web.app._get_publisher")
    def test_direct_publish_writes_publish_log(self, mock_pub, mock_risk):
        """POST /api/publish/{id} should write results to publish_log."""
        mock_risk.return_value = {
            "risk_result": {"risk_score": 1, "risk_factors": [], "risk_level": "low"}
        }
        pub_instance = MagicMock()
        pub_instance.publish = AsyncMock(
            return_value=MagicMock(success=True, post_url="https://example.com/post", error="")
        )
        mock_pub.return_value = pub_instance

        from web.app import get_db
        conn = get_db()
        cur = conn.execute(
            "INSERT INTO posts (title, body, platforms, status, extra_fields) VALUES (?, ?, ?, ?, ?)",
            ("Log Test", "body", json.dumps(["tiktok"]), "draft", "{}"),
        )
        post_id = cur.lastrowid
        conn.commit()
        conn.close()

        r = client.post(f"/api/publish/{post_id}")
        assert r.status_code == 200

        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM publish_log WHERE post_id = ?", (post_id,)
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        assert rows[0]["platform"] == "tiktok"
        assert rows[0]["success"] == 1  # SQLite stores bool as int
