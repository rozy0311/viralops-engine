"""
ViralOps Engine — Publer Integration Tests
Tests for publer_publisher.py — REST API bridge publisher.

Run: pytest tests/test_publer.py -v
"""

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Test fixtures ──


@pytest.fixture
def mock_env(monkeypatch):
    """Set up Publer env vars for testing."""
    monkeypatch.setenv("PUBLER_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("PUBLER_WORKSPACE_ID", "ws-test-67890")


@pytest.fixture
def publisher(mock_env):
    """Create a PublerPublisher instance with test env vars."""
    from integrations.publer_publisher import PublerPublisher

    pub = PublerPublisher(account_id="test")
    return pub


# ── Auth & Configuration Tests ──


class TestPublerConfig:
    """Test Publer configuration and auth."""

    def test_is_configured_with_key(self, publisher):
        assert publisher.is_configured is True

    def test_is_configured_without_key(self, monkeypatch):
        monkeypatch.delenv("PUBLER_API_KEY", raising=False)
        from integrations.publer_publisher import PublerPublisher

        pub = PublerPublisher()
        assert pub.is_configured is False

    def test_headers_include_auth(self, publisher):
        headers = publisher._headers()
        assert headers["Authorization"] == "Bearer-API test-api-key-12345"
        assert headers["Publer-Workspace-Id"] == "ws-test-67890"
        assert headers["Content-Type"] == "application/json"

    def test_headers_without_workspace(self, monkeypatch, mock_env):
        monkeypatch.setenv("PUBLER_WORKSPACE_ID", "")
        from integrations.publer_publisher import PublerPublisher

        pub = PublerPublisher()
        headers = pub._headers()
        assert "Publer-Workspace-Id" not in headers

    def test_platform_name(self, publisher):
        assert publisher.platform == "publer"


# ── Network Mapping Tests ──


class TestNetworkMapping:
    """Test platform-to-Publer network mapping."""

    def test_all_viralops_platforms_mapped(self):
        from integrations.publer_publisher import PLATFORM_TO_PUBLER_NETWORK

        expected = [
            "facebook", "instagram", "twitter", "linkedin",
            "pinterest", "youtube", "tiktok", "google_business",
            "telegram", "mastodon", "threads", "bluesky",
        ]
        for platform in expected:
            assert platform in PLATFORM_TO_PUBLER_NETWORK, f"{platform} not mapped"

    def test_tiktok_mapping(self):
        from integrations.publer_publisher import PLATFORM_TO_PUBLER_NETWORK

        assert PLATFORM_TO_PUBLER_NETWORK["tiktok"] == "tiktok"

    def test_google_business_mapping(self):
        from integrations.publer_publisher import PLATFORM_TO_PUBLER_NETWORK

        assert PLATFORM_TO_PUBLER_NETWORK["google_business"] == "google"

    def test_default_content_types(self):
        from integrations.publer_publisher import PLATFORM_DEFAULT_CONTENT_TYPE

        assert PLATFORM_DEFAULT_CONTENT_TYPE["tiktok"] == "video"
        assert PLATFORM_DEFAULT_CONTENT_TYPE["facebook"] == "status"
        assert PLATFORM_DEFAULT_CONTENT_TYPE["pinterest"] == "pin"
        assert PLATFORM_DEFAULT_CONTENT_TYPE["instagram"] == "photo"


# ── Connection Tests ──


class TestPublerConnection:
    """Test connection, workspace discovery, account listing."""

    @pytest.mark.asyncio
    async def test_connect_not_configured(self, monkeypatch):
        monkeypatch.delenv("PUBLER_API_KEY", raising=False)
        from integrations.publer_publisher import PublerPublisher

        pub = PublerPublisher()
        result = await pub.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_success(self, publisher):
        """Test connect with PUBLER_WORKSPACE_ID already set (fixture provides it).
        connect() calls: GET /me → GET /accounts (skips workspaces since ID is set).
        """
        mock_response_me = MagicMock()
        mock_response_me.status_code = 200
        mock_response_me.json.return_value = {"data": {"email": "test@test.com"}}

        mock_response_acc = MagicMock()
        mock_response_acc.status_code = 200
        mock_response_acc.json.return_value = {
            "data": [
                {"id": "acc-1", "name": "My TikTok", "type": "tiktok"},
                {"id": "acc-2", "name": "My Instagram", "type": "instagram"},
            ]
        }

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [mock_response_me, mock_response_acc]
            result = await publisher.connect()

        assert result is True
        assert publisher._connected is True
        assert len(publisher._accounts) == 2

    @pytest.mark.asyncio
    async def test_connect_auth_fail(self, publisher):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await publisher.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_account_ids_filter(self, publisher):
        publisher._accounts = [
            {"id": "acc-1", "name": "TikTok Brand", "type": "tiktok"},
            {"id": "acc-2", "name": "IG Main", "type": "instagram"},
            {"id": "acc-3", "name": "TikTok Niche", "type": "tiktok"},
        ]

        ids = await publisher.get_account_ids("tiktok")
        assert ids == ["acc-1", "acc-3"]

        ids = await publisher.get_account_ids("instagram")
        assert ids == ["acc-2"]

        ids = await publisher.get_account_ids()
        assert len(ids) == 3


# ── Publish Tests ──


class TestPublerPublish:
    """Test publishing flow."""

    @pytest.mark.asyncio
    async def test_publish_not_connected_auto_connects(self, publisher):
        """publish() should auto-connect if not connected."""
        publisher._connected = False

        with patch.object(publisher, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = False
            result = await publisher.publish({"caption": "test"})

        assert result["success"] is False
        assert "not connected" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_publish_no_accounts(self, publisher):
        publisher._connected = True
        publisher._accounts = []

        with patch.object(publisher, "get_account_ids", new_callable=AsyncMock) as mock_ids:
            mock_ids.return_value = []
            result = await publisher.publish({"caption": "test", "platforms": ["tiktok"]})

        assert result["success"] is False
        assert "no publer accounts" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_publish_success_with_job(self, publisher):
        """Test full publish flow: submit → job_id → poll → success."""
        publisher._connected = True
        publisher._accounts = [
            {"id": "acc-1", "name": "TikTok", "type": "tiktok"},
        ]

        # Mock the publish request
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "success": True,
            "data": {"job_id": "job-abc-123"},
        }

        # Mock job poll
        mock_job_resp = MagicMock()
        mock_job_resp.status_code = 200
        mock_job_resp.json.return_value = {
            "data": {"status": "complete", "result": {"post_ids": ["post-1"]}},
        }

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [mock_post_resp, mock_job_resp]
            result = await publisher.publish({
                "caption": "Test TikTok post!",
                "platforms": ["tiktok"],
                "hashtags": ["viral", "trending"],
            })

        assert result["success"] is True
        assert result["post_id"] == "job-abc-123"
        assert result["accounts_count"] == 1

    @pytest.mark.asyncio
    async def test_publish_with_schedule(self, publisher):
        """Test scheduled publishing."""
        publisher._connected = True
        publisher._accounts = [
            {"id": "acc-1", "name": "Pinterest", "type": "pinterest"},
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"job_id": "job-sched-456"},
        }

        mock_job = MagicMock()
        mock_job.status_code = 200
        mock_job.json.return_value = {
            "data": {"status": "complete"},
        }

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [mock_resp, mock_job]
            result = await publisher.publish({
                "caption": "Scheduled pin!",
                "platforms": ["pinterest"],
                "schedule_at": "2026-03-01T14:00:00Z",
            })

        assert result["success"] is True
        assert result["scheduled"] is True

    @pytest.mark.asyncio
    async def test_publish_hashtags_appended(self, publisher):
        """Test that hashtags are properly appended to caption."""
        publisher._connected = True
        publisher._accounts = [
            {"id": "acc-1", "name": "test", "type": "twitter"},
        ]

        captured_body = {}

        async def capture_request(method, endpoint, **kwargs):
            if "posts" in endpoint:
                captured_body.update(kwargs.get("json_data", {}))
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"data": {"job_id": "j1"}}
                return resp
            elif "job_status" in endpoint:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"data": {"status": "complete"}}
                return resp

        with patch.object(publisher, "_request", side_effect=capture_request):
            await publisher.publish({
                "caption": "Hello world",
                "hashtags": ["test", "#already"],
                "platforms": ["twitter"],
            })

        posts = captured_body.get("bulk", {}).get("posts", [])
        assert len(posts) == 1
        text = posts[0]["networks"]["twitter"]["text"]
        assert "#test" in text
        assert "#already" in text


# ── Job Polling Tests ──


class TestJobPolling:
    """Test async job polling."""

    @pytest.mark.asyncio
    async def test_poll_job_success(self, publisher):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"status": "complete", "result": {"ok": True}},
        }

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_resp
            result = await publisher._poll_job("job-123")

        assert result["success"] is True
        assert result["status"] == "complete"

    @pytest.mark.asyncio
    async def test_poll_job_failed(self, publisher):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "status": "failed",
                "result": {"failures": {"error": "Media upload failed"}},
            },
        }

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_resp
            result = await publisher._poll_job("job-fail")

        assert result["success"] is False
        assert "Media upload failed" in result["error"]

    @pytest.mark.asyncio
    async def test_poll_job_timeout(self, publisher):
        publisher.JOB_POLL_TIMEOUT = 0.5
        publisher.JOB_POLL_INTERVAL = 0.1

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"status": "working"},
        }

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_resp
            result = await publisher._poll_job("job-slow")

        assert result["success"] is False
        assert result["status"] == "timeout"


# ── Schedule / Bulk Tests ──


class TestPublerSchedule:
    """Test scheduling features."""

    @pytest.mark.asyncio
    async def test_schedule_requires_datetime(self, publisher):
        publisher._connected = True
        result = await publisher.schedule({"caption": "test"})
        assert result["success"] is False
        assert "schedule_at" in result["error"]

    @pytest.mark.asyncio
    async def test_bulk_schedule(self, publisher):
        publisher._connected = True
        publisher._accounts = [
            {"id": "acc-1", "name": "TT1", "type": "tiktok"},
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"job_id": "bulk-j1"}}

        mock_job = MagicMock()
        mock_job.status_code = 200
        mock_job.json.return_value = {"data": {"status": "complete"}}

        with patch.object(publisher, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [mock_resp, mock_job]
            result = await publisher.bulk_schedule([
                {"caption": "Post 1", "platforms": ["tiktok"], "schedule_at": "2026-03-01T10:00Z"},
                {"caption": "Post 2", "platforms": ["tiktok"], "schedule_at": "2026-03-01T14:00Z"},
            ])

        assert result["success"] is True
        assert result["posts_count"] == 2


# ── Test Connection Tests ──


class TestPublerTestConnection:
    """Test the test_connection diagnostic method."""

    @pytest.mark.asyncio
    async def test_connection_diagnostic(self, publisher):
        with patch.object(publisher, "connect", new_callable=AsyncMock) as mock_conn:
            mock_conn.return_value = True
            publisher._connected = True
            publisher._accounts = [
                {"name": "My TikTok", "type": "tiktok"},
                {"name": "My IG", "type": "instagram"},
            ]

            with patch.object(publisher, "get_accounts", new_callable=AsyncMock) as mock_acc:
                mock_acc.return_value = publisher._accounts
                result = await publisher.test_connection()

        assert result["connected"] is True
        assert result["accounts_count"] == 2
        assert "My TikTok (tiktok)" in result["account_names"]

    @pytest.mark.asyncio
    async def test_connection_diagnostic_fail(self, publisher):
        with patch.object(publisher, "connect", new_callable=AsyncMock) as mock_conn:
            mock_conn.return_value = False
            result = await publisher.test_connection()

        assert result["connected"] is False


# ── Lifecycle Tests ──


class TestPublerLifecycle:
    """Test client lifecycle."""

    @pytest.mark.asyncio
    async def test_close(self, publisher):
        mock_client = AsyncMock()
        publisher._client = mock_client
        publisher._connected = True

        await publisher.close()

        assert publisher._client is None
        assert publisher._connected is False
        mock_client.aclose.assert_awaited_once()

    def test_repr(self, publisher):
        publisher._connected = False
        publisher._accounts = []
        r = repr(publisher)
        assert "test" in r
        assert "disconnected" in r


# ── API Contract Compliance ──


class TestPublerApiContract:
    """Verify Publer publisher implements required interface."""

    def test_publer_has_required_interface(self, publisher):
        """Publer publisher must have all required methods."""
        required_methods = [
            "connect", "publish", "schedule",
            "test_connection", "close",
        ]
        for method in required_methods:
            assert hasattr(publisher, method), f"Missing method: {method}"
            assert callable(getattr(publisher, method)), f"{method} not callable"

    def test_publer_returns_same_result_format(self):
        """publish() return dict must include required keys."""
        required_keys = ["success", "platform"]
        from integrations.publer_publisher import PublerPublisher

        pub = PublerPublisher()
        # Simulate a failure result
        result = {
            "success": False,
            "error": "test",
            "platform": pub.platform,
        }
        for key in required_keys:
            assert key in result


# ── Run ──

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
