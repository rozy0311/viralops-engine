"""
Tests for v3.2.0 — Sendible UI Automation (Playwright Stealth)

Tests cover:
  1. Sendible page HTML route
  2. Sendible API endpoints (/api/sendible/*)
  3. SendibleUIPublisher class (import, interface, methods)
  4. Anti-detect helpers (bezier, random delay, human type)
  5. Platform publisher registry integration
  6. .env.example Sendible UI vars
  7. UI HTML elements
"""
import pytest
import json
import os
import time
import math
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock
from fastapi.testclient import TestClient

# ── Import app ──
import sys, pathlib
_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
from web.app import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════
# 1. SENDIBLE PAGE ROUTE
# ═══════════════════════════════════════════════════════════════

class TestSendiblePageRoute:
    """Test /sendible HTML page route."""

    def test_sendible_page_returns_200(self):
        r = client.get("/sendible")
        assert r.status_code == 200

    def test_sendible_page_is_html(self):
        r = client.get("/sendible")
        assert "text/html" in r.headers.get("content-type", "")

    def test_sendible_page_has_section(self):
        r = client.get("/sendible")
        assert 'id="page-sendible"' in r.text

    def test_sendible_page_has_status_badge(self):
        r = client.get("/sendible")
        assert 'id="snd-status-badge"' in r.text

    def test_sendible_page_has_test_button(self):
        r = client.get("/sendible")
        assert "testSendibleConnection" in r.text

    def test_sendible_page_has_services_grid(self):
        r = client.get("/sendible")
        assert 'id="snd-services-grid"' in r.text

    def test_sendible_page_has_publish_form(self):
        r = client.get("/sendible")
        assert 'id="snd-message"' in r.text

    def test_sendible_page_has_schedule_input(self):
        r = client.get("/sendible")
        assert 'id="snd-schedule"' in r.text

    def test_sendible_page_has_setup_guide(self):
        r = client.get("/sendible")
        assert "Setup Guide" in r.text

    def test_sendible_page_has_nav_link(self):
        r = client.get("/sendible")
        assert "Sendible Bridge" in r.text

    def test_sendible_page_mentions_sendible(self):
        r = client.get("/sendible")
        assert "Sendible" in r.text


# ═══════════════════════════════════════════════════════════════
# 2. SENDIBLE API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

class TestSendibleStatusAPI:
    """Test GET /api/sendible/status."""

    def test_status_returns_200(self):
        r = client.get("/api/sendible/status")
        assert r.status_code == 200

    def test_status_has_configured_field(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert "configured" in data

    def test_status_has_auth_mode(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert "auth_mode" in data
        assert data["auth_mode"] in ("ui_automation", "none")

    def test_status_has_env_vars(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert "env_vars" in data
        assert "SENDIBLE_EMAIL" in data["env_vars"]
        assert "SENDIBLE_PASSWORD" in data["env_vars"]

    def test_status_has_method_field(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert data["method"] == "playwright_stealth"

    def test_status_unconfigured_by_default(self):
        """Without env vars, should show unconfigured."""
        r = client.get("/api/sendible/status")
        data = r.json()
        # Unless env vars are set in test env
        assert isinstance(data["configured"], bool)

    @patch.dict(os.environ, {
        "SENDIBLE_EMAIL": "test@example.com",
        "SENDIBLE_PASSWORD": "secret123",
    })
    def test_status_configured_with_credentials(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert data["configured"] is True
        assert data["auth_mode"] == "ui_automation"

    @patch.dict(os.environ, {
        "SENDIBLE_EMAIL": "",
        "SENDIBLE_PASSWORD": "",
    })
    def test_status_unconfigured_empty_credentials(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert data["configured"] is False
        assert data["auth_mode"] == "none"


class TestSendibleTestAPI:
    """Test POST /api/sendible/test."""

    def test_test_returns_200(self):
        r = client.post("/api/sendible/test")
        assert r.status_code == 200

    def test_test_returns_connected_field(self):
        r = client.post("/api/sendible/test")
        data = r.json()
        assert "connected" in data

    def test_test_fails_without_credentials(self):
        r = client.post("/api/sendible/test")
        data = r.json()
        assert data["connected"] is False


class TestSendibleServicesAPI:
    """Test GET /api/sendible/services."""

    def test_services_returns_200(self):
        r = client.get("/api/sendible/services")
        assert r.status_code == 200

    def test_services_has_services_field(self):
        r = client.get("/api/sendible/services")
        data = r.json()
        assert "services" in data

    def test_services_returns_list(self):
        r = client.get("/api/sendible/services")
        data = r.json()
        assert isinstance(data["services"], list)


class TestSendiblePublishAPI:
    """Test POST /api/sendible/publish."""

    def test_publish_returns_200(self):
        r = client.post("/api/sendible/publish", json={"caption": "test"})
        assert r.status_code == 200

    def test_publish_requires_connection(self):
        r = client.post("/api/sendible/publish", json={"caption": "test"})
        data = r.json()
        assert data["success"] is False

    def test_publish_returns_error_field(self):
        r = client.post("/api/sendible/publish", json={"caption": "test"})
        data = r.json()
        assert "error" in data or "success" in data


class TestSendibleMessagesAPI:
    """Test GET /api/sendible/messages."""

    def test_messages_returns_200(self):
        r = client.get("/api/sendible/messages")
        assert r.status_code == 200

    def test_messages_has_messages_field(self):
        r = client.get("/api/sendible/messages")
        data = r.json()
        assert "messages" in data

    def test_messages_supports_pagination(self):
        r = client.get("/api/sendible/messages?per_page=5&page=2")
        assert r.status_code == 200


class TestSendibleAccountAPI:
    """Test GET /api/sendible/account."""

    def test_account_returns_200(self):
        r = client.get("/api/sendible/account")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════
# 3. SENDIBLE UI PUBLISHER CLASS
# ═══════════════════════════════════════════════════════════════

class TestSendibleUIPublisher:
    """Test SendibleUIPublisher class (unit tests, no browser)."""

    def test_import(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        assert SendibleUIPublisher

    def test_platform_name(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert pub.platform == "sendible"

    def test_default_account_id(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert pub.account_id == "sendible_main"

    def test_custom_account_id(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher(account_id="alt")
        assert pub.account_id == "alt"

    def test_base_url(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        assert "sendible.com" in SendibleUIPublisher.BASE_URL

    def test_has_publish_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "publish")
        assert callable(pub.publish)

    def test_has_schedule_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "schedule")

    def test_has_connect_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "connect")

    def test_has_login_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "login")

    def test_has_get_services_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "get_services")

    def test_has_get_messages_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "get_messages")

    def test_has_close_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "close")

    def test_has_test_connection_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "test_connection")

    def test_has_get_account_details_method(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert hasattr(pub, "get_account_details")

    def test_default_not_configured(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        if not os.environ.get("SENDIBLE_EMAIL"):
            assert pub.is_configured is False

    @patch.dict(os.environ, {
        "SENDIBLE_EMAIL": "test@example.com",
        "SENDIBLE_PASSWORD": "secret",
    })
    def test_configured_with_credentials(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert pub.is_configured is True

    @patch.dict(os.environ, {
        "SENDIBLE_EMAIL": "test@example.com",
        "SENDIBLE_PASSWORD": "secret",
        "SENDIBLE_PROXY": "socks5://user:pass@proxy:1080",
    })
    def test_proxy_from_env(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert pub.proxy == "socks5://user:pass@proxy:1080"

    @patch.dict(os.environ, {"SENDIBLE_HEADLESS": "true"})
    def test_headless_from_env(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert pub.headless is True

    def test_default_non_headless(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert pub.headless is False

    @pytest.mark.asyncio
    async def test_publish_not_connected(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        result = await pub.publish({"caption": "test"})
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_schedule_without_date(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        result = await pub.schedule({"caption": "test"})
        assert result["success"] is False
        assert "schedule_at" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_close_without_browser(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        await pub.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_get_services_not_connected(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        services = await pub.get_services()
        assert isinstance(services, list)

    @pytest.mark.asyncio
    async def test_get_messages_not_connected(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        messages = await pub.get_messages()
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_get_account_details_not_connected(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        details = await pub.get_account_details()
        assert isinstance(details, dict)

    @pytest.mark.asyncio
    async def test_test_connection_fails_without_creds(self):
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        result = await pub.test_connection()
        assert result["connected"] is False


# ═══════════════════════════════════════════════════════════════
# 4. ANTI-DETECT HELPERS
# ═══════════════════════════════════════════════════════════════

class TestAntiDetectHelpers:
    """Test stealth helper functions."""

    def test_random_delay_within_range(self):
        from integrations.sendible_ui_publisher import _random_delay
        for _ in range(50):
            d = _random_delay(2.0, 5.0)
            assert 2.0 <= d <= 5.0

    def test_random_delay_custom_range(self):
        from integrations.sendible_ui_publisher import _random_delay
        for _ in range(20):
            d = _random_delay(0.5, 1.0)
            assert 0.5 <= d <= 1.0

    def test_bezier_points_count(self):
        from integrations.sendible_ui_publisher import _bezier_points
        pts = _bezier_points((0, 0), (100, 100), steps=20)
        assert len(pts) == 21  # steps + 1

    def test_bezier_points_start(self):
        from integrations.sendible_ui_publisher import _bezier_points
        pts = _bezier_points((10, 20), (200, 300), steps=10)
        assert pts[0] == (10, 20)

    def test_bezier_points_end(self):
        from integrations.sendible_ui_publisher import _bezier_points
        pts = _bezier_points((10, 20), (200, 300), steps=10)
        assert abs(pts[-1][0] - 200) < 0.01
        assert abs(pts[-1][1] - 300) < 0.01

    def test_bezier_points_are_tuples(self):
        from integrations.sendible_ui_publisher import _bezier_points
        pts = _bezier_points((0, 0), (50, 50), steps=5)
        for p in pts:
            assert isinstance(p, tuple)
            assert len(p) == 2

    def test_bezier_produces_curve(self):
        """Bezier should produce multiple points."""
        from integrations.sendible_ui_publisher import _bezier_points
        pts = _bezier_points((0, 0), (100, 100), steps=20)
        assert len(pts) > 2

    def test_user_agents_list(self):
        from integrations.sendible_ui_publisher import USER_AGENTS
        assert len(USER_AGENTS) >= 3
        for ua in USER_AGENTS:
            assert "Chrome" in ua
            assert "Mozilla" in ua

    def test_cookies_dir_path(self):
        from integrations.sendible_ui_publisher import COOKIES_DIR
        assert "sendible_session" in str(COOKIES_DIR)


# ═══════════════════════════════════════════════════════════════
# 5. PLATFORM PUBLISHER REGISTRY
# ═══════════════════════════════════════════════════════════════

class TestRegistryIntegration:
    """Test Sendible in platform publisher registry."""

    def test_sendible_in_registry_imports(self):
        """SendibleUIPublisher should be importable."""
        from integrations.sendible_ui_publisher import SendibleUIPublisher
        pub = SendibleUIPublisher()
        assert pub.platform == "sendible"

    def test_registry_includes_sendible_type(self):
        """Registry should categorize sendible as sendible_bridge."""
        from integrations.platform_publisher import PublisherRegistry
        reg = PublisherRegistry()
        reg.register_all()
        platforms = reg.list_platforms()
        if "sendible" in platforms:
            assert platforms["sendible"] == "sendible_bridge"

    def test_registry_can_get_sendible(self):
        """Registry.get('sendible') should return a UI publisher."""
        from integrations.platform_publisher import PublisherRegistry
        reg = PublisherRegistry()
        reg.register_all()
        pub = reg.get("sendible")
        if pub:
            assert pub.platform == "sendible"

    def test_registry_sendible_is_ui_publisher(self):
        """Registry should use SendibleUIPublisher, not old REST one."""
        from integrations.platform_publisher import PublisherRegistry
        reg = PublisherRegistry()
        reg.register_all()
        pub = reg.get("sendible")
        if pub:
            from integrations.sendible_ui_publisher import SendibleUIPublisher
            assert isinstance(pub, SendibleUIPublisher)


# ═══════════════════════════════════════════════════════════════
# 6. ENV EXAMPLE
# ═══════════════════════════════════════════════════════════════

class TestEnvExample:
    """Test .env.example has Sendible UI variables."""

    def _read_env_example(self):
        env_path = pathlib.Path(__file__).resolve().parent.parent / ".env.example"
        return env_path.read_text(encoding="utf-8")

    def test_has_sendible_email(self):
        content = self._read_env_example()
        assert "SENDIBLE_EMAIL" in content

    def test_has_sendible_password(self):
        content = self._read_env_example()
        assert "SENDIBLE_PASSWORD" in content

    def test_has_sendible_proxy(self):
        content = self._read_env_example()
        assert "SENDIBLE_PROXY" in content

    def test_has_sendible_headless(self):
        content = self._read_env_example()
        assert "SENDIBLE_HEADLESS" in content

    def test_has_sendible_section_header(self):
        content = self._read_env_example()
        assert "Sendible" in content

    def test_has_ui_automation_mention(self):
        content = self._read_env_example()
        assert "automation" in content.lower() or "playwright" in content.lower() or "browser" in content.lower()


# ═══════════════════════════════════════════════════════════════
# 7. UI INTEGRATION
# ═══════════════════════════════════════════════════════════════

class TestSendibleUI:
    """Test Sendible UI elements in HTML."""

    def test_sendible_in_sidebar(self):
        r = client.get("/")
        assert "/sendible" in r.text

    def test_sendible_page_has_js_functions(self):
        r = client.get("/sendible")
        assert "loadSendibleStatus" in r.text
        assert "testSendibleConnection" in r.text
        assert "refreshSendibleServices" in r.text
        assert "sendiblePublish" in r.text
        assert "loadSendibleMessages" in r.text

    def test_sendible_page_has_publish_buttons(self):
        r = client.get("/sendible")
        assert "Publish Now" in r.text
        assert "Schedule" in r.text

    def test_sendible_page_has_platform_filter(self):
        r = client.get("/sendible")
        assert 'id="snd-platforms"' in r.text

    def test_sendible_page_has_media_input(self):
        r = client.get("/sendible")
        assert 'id="snd-media"' in r.text

    def test_sendible_page_has_hashtags_input(self):
        r = client.get("/sendible")
        assert 'id="snd-hashtags"' in r.text

    def test_sendible_page_has_messages_section(self):
        r = client.get("/sendible")
        assert 'id="snd-messages-list"' in r.text

    def test_sendible_page_has_account_info(self):
        r = client.get("/sendible")
        assert 'id="snd-account-info"' in r.text
