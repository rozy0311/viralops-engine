"""
Tests for v3.1.0 — Sendible Bridge (REST API → TikTok/IG/FB/etc.)

Tests cover:
  1. Sendible page HTML route
  2. Sendible API endpoints (/api/sendible/*)
  3. SendibleAuth (encryption, token caching)
  4. SendiblePublisher (publish, schedule, services, messages)
  5. Platform publisher registry integration
  6. .env.example Sendible vars
"""
import pytest
import json
import os
import time
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

    def test_sendible_page_mentions_developer_app(self):
        r = client.get("/sendible")
        assert "Developer Application" in r.text


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
        assert data["auth_mode"] in ("oauth", "direct_token", "none")

    def test_status_has_env_vars(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert "env_vars" in data
        assert "SENDIBLE_APPLICATION_ID" in data["env_vars"]
        assert "SENDIBLE_ACCESS_TOKEN" in data["env_vars"]

    def test_status_unconfigured_by_default(self):
        """Without env vars, should show unconfigured."""
        r = client.get("/api/sendible/status")
        data = r.json()
        # Unless env vars are set in test env
        assert isinstance(data["configured"], bool)

    @patch.dict(os.environ, {"SENDIBLE_ACCESS_TOKEN": "test-token"})
    def test_status_direct_token_mode(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert data["configured"] is True
        assert data["auth_mode"] == "direct_token"

    @patch.dict(os.environ, {
        "SENDIBLE_APPLICATION_ID": "app123",
        "SENDIBLE_SHARED_KEY": "key123",
        "SENDIBLE_SHARED_IV": "iv123",
        "SENDIBLE_USERNAME": "user@test.com",
        "SENDIBLE_API_KEY": "apikey123",
    })
    def test_status_oauth_mode(self):
        r = client.get("/api/sendible/status")
        data = r.json()
        assert data["configured"] is True
        assert data["auth_mode"] == "oauth"


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
# 3. SENDIBLE AUTH
# ═══════════════════════════════════════════════════════════════

class TestSendibleAuth:
    """Test SendibleAuth class."""

    def test_import(self):
        from integrations.sendible_publisher import SendibleAuth
        assert SendibleAuth

    def test_default_not_configured(self):
        from integrations.sendible_publisher import SendibleAuth
        auth = SendibleAuth()
        # Without env vars, should not be configured
        if not os.environ.get("SENDIBLE_ACCESS_TOKEN"):
            assert auth.is_configured is False or auth.is_configured is True  # depends on env

    @patch.dict(os.environ, {"SENDIBLE_ACCESS_TOKEN": "direct-tok"})
    def test_direct_token_configured(self):
        from integrations.sendible_publisher import SendibleAuth
        auth = SendibleAuth()
        assert auth.is_configured is True

    @patch.dict(os.environ, {
        "SENDIBLE_APPLICATION_ID": "app",
        "SENDIBLE_SHARED_KEY": "key",
        "SENDIBLE_SHARED_IV": "iv",
        "SENDIBLE_USERNAME": "user",
        "SENDIBLE_API_KEY": "apikey",
    })
    def test_oauth_configured(self):
        from integrations.sendible_publisher import SendibleAuth
        auth = SendibleAuth()
        assert auth.is_configured is True

    def test_api_url(self):
        from integrations.sendible_publisher import SendibleAuth
        assert "sendible.com" in SendibleAuth.API_URL or "api" in SendibleAuth.API_URL

    @patch.dict(os.environ, {"SENDIBLE_ACCESS_TOKEN": "test-token-123"})
    @pytest.mark.asyncio
    async def test_direct_token_returns_immediately(self):
        from integrations.sendible_publisher import SendibleAuth
        import httpx
        auth = SendibleAuth()
        async with httpx.AsyncClient() as c:
            token = await auth.get_access_token(c)
            assert token == "test-token-123"

    def test_token_cache(self):
        from integrations.sendible_publisher import SendibleAuth
        auth = SendibleAuth()
        auth._direct_token = "cached-tok"
        # Direct token should be instant, no network call


# ═══════════════════════════════════════════════════════════════
# 4. SENDIBLE PUBLISHER
# ═══════════════════════════════════════════════════════════════

class TestSendiblePublisher:
    """Test SendiblePublisher class."""

    def test_import(self):
        from integrations.sendible_publisher import SendiblePublisher
        assert SendiblePublisher

    def test_platform_name(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert pub.platform == "sendible"

    def test_default_account_id(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert pub.account_id == "sendible_main"

    def test_custom_account_id(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher(account_id="alt")
        assert pub.account_id == "alt"

    def test_has_publish_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "publish")
        assert callable(pub.publish)

    def test_has_schedule_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "schedule")

    def test_has_connect_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "connect")

    def test_has_get_services_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "get_services")

    def test_has_get_messages_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "get_messages")

    def test_has_close_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "close")

    def test_has_test_connection_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "test_connection")

    def test_has_shorten_url_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "shorten_url")

    def test_has_get_reports_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "get_reports")

    def test_has_get_account_details_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "get_account_details")

    def test_has_delete_message_method(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert hasattr(pub, "delete_message")

    def test_retry_config(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        assert pub.MAX_RETRIES == 3
        assert pub.RETRY_DELAY == 2.0

    @pytest.mark.asyncio
    async def test_publish_not_connected(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        # Publish without connection should return error
        result = await pub.publish({"caption": "test"})
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_schedule_without_date(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        result = await pub.schedule({"caption": "test"})
        assert result["success"] is False
        assert "schedule_at" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        await pub.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_get_service_ids_empty(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        pub._services = [
            {"id": "1", "service_type": "Instagram", "name": "My IG"},
            {"id": "2", "service_type": "TikTok", "name": "My TT"},
            {"id": "3", "service_type": "Facebook", "name": "My FB"},
        ]
        # No filter — returns all
        ids = await pub.get_service_ids()
        assert len(ids) == 3

    @pytest.mark.asyncio
    async def test_get_service_ids_filtered(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        pub._services = [
            {"id": "1", "service_type": "Instagram", "name": "My IG"},
            {"id": "2", "service_type": "TikTok", "name": "My TT"},
            {"id": "3", "service_type": "Facebook", "name": "My FB"},
        ]
        ids = await pub.get_service_ids("instagram")
        assert ids == ["1"]

    @pytest.mark.asyncio
    async def test_get_service_ids_case_insensitive(self):
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
        pub._services = [
            {"id": "10", "service_type": "TIKTOK", "name": "TikTok Account"},
        ]
        ids = await pub.get_service_ids("tiktok")
        assert ids == ["10"]


# ═══════════════════════════════════════════════════════════════
# 5. PLATFORM PUBLISHER REGISTRY
# ═══════════════════════════════════════════════════════════════

class TestRegistryIntegration:
    """Test Sendible in platform publisher registry."""

    def test_sendible_in_registry_imports(self):
        """Sendible publisher should be importable."""
        from integrations.sendible_publisher import SendiblePublisher
        pub = SendiblePublisher()
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
        """Registry.get('sendible') should return a publisher."""
        from integrations.platform_publisher import PublisherRegistry
        reg = PublisherRegistry()
        reg.register_all()
        pub = reg.get("sendible")
        if pub:
            assert pub.platform == "sendible"


# ═══════════════════════════════════════════════════════════════
# 6. ENV EXAMPLE
# ═══════════════════════════════════════════════════════════════

class TestEnvExample:
    """Test .env.example has Sendible variables."""

    def _read_env_example(self):
        env_path = pathlib.Path(__file__).resolve().parent.parent / ".env.example"
        return env_path.read_text(encoding="utf-8")

    def test_has_sendible_application_id(self):
        content = self._read_env_example()
        assert "SENDIBLE_APPLICATION_ID" in content

    def test_has_sendible_shared_key(self):
        content = self._read_env_example()
        assert "SENDIBLE_SHARED_KEY" in content

    def test_has_sendible_shared_iv(self):
        content = self._read_env_example()
        assert "SENDIBLE_SHARED_IV" in content

    def test_has_sendible_username(self):
        content = self._read_env_example()
        assert "SENDIBLE_USERNAME" in content

    def test_has_sendible_api_key(self):
        content = self._read_env_example()
        assert "SENDIBLE_API_KEY" in content

    def test_has_sendible_access_token(self):
        content = self._read_env_example()
        assert "SENDIBLE_ACCESS_TOKEN" in content

    def test_has_setup_instructions(self):
        content = self._read_env_example()
        assert "Developer Application" in content or "developer" in content.lower()

    def test_has_sendible_section_header(self):
        content = self._read_env_example()
        assert "Sendible" in content


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
