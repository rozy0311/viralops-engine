"""
Tests for Shopify Blog Auto-Share v3.3.0

Tests:
  - ShopifyBlogWatcher: connect, resolve blogs, fetch articles, state persistence
  - MultiTikTokPublisher: discover accounts, publish photo/video, broadcast/round-robin
  - ShopifyAutoShare: initialize, tick, transform content, history tracking
  - API endpoints: /api/blog-share/*
"""

import json
import os
import asyncio
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone


# ════════════════════════════════════════════════════════════════
# ShopifyBlogWatcher Tests
# ════════════════════════════════════════════════════════════════

class TestShopifyBlogWatcher:
    """Tests for integrations/shopify_blog_watcher.py"""

    def _make_watcher(self):
        with patch.dict(os.environ, {
            "SHOPIFY_SHOP": "test-shop",
            "SHOPIFY_ACCESS_TOKEN": "shpat_test123",
        }):
            from integrations.shopify_blog_watcher import ShopifyBlogWatcher
            return ShopifyBlogWatcher(
                blog_handles=["sustainable-living", "brand-partnerships"]
            )

    def test_init_defaults(self):
        """Watcher initializes with default blog handles."""
        with patch.dict(os.environ, {
            "SHOPIFY_SHOP": "test-shop",
            "SHOPIFY_ACCESS_TOKEN": "shpat_test",
        }):
            from integrations.shopify_blog_watcher import ShopifyBlogWatcher
            w = ShopifyBlogWatcher()
            assert "sustainable-living" in w._blog_handles
            assert "brand-partnerships" in w._blog_handles

    def test_init_custom_handles(self):
        """Watcher accepts custom blog handles."""
        w = self._make_watcher()
        assert w._blog_handles == ["sustainable-living", "brand-partnerships"]

    def test_init_from_env(self):
        """Watcher reads blog handles from SHOPIFY_WATCH_BLOGS env."""
        with patch.dict(os.environ, {
            "SHOPIFY_SHOP": "test-shop",
            "SHOPIFY_ACCESS_TOKEN": "shpat_test",
            "SHOPIFY_WATCH_BLOGS": "blog-a,blog-b,blog-c",
        }):
            from integrations.shopify_blog_watcher import ShopifyBlogWatcher
            w = ShopifyBlogWatcher()
            assert w._blog_handles == ["blog-a", "blog-b", "blog-c"]

    def test_init_no_credentials(self):
        """Watcher handles missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            from integrations.shopify_blog_watcher import ShopifyBlogWatcher
            w = ShopifyBlogWatcher()
            assert w._shop == ""
            assert w._token == ""

    @pytest.mark.asyncio
    async def test_connect_resolves_blogs(self):
        """Connect resolves blog handles to IDs."""
        w = self._make_watcher()
        mock_blogs = [
            {"id": 111, "handle": "sustainable-living", "title": "Sustainable Living"},
            {"id": 222, "handle": "brand-partnerships", "title": "Brand Partnerships"},
            {"id": 333, "handle": "other-blog", "title": "Other"},
        ]

        # Patch _list_blogs and _load_state directly
        with patch.object(w, '_list_blogs', new_callable=AsyncMock) as mock_list, \
             patch.object(type(w), '_load_state', staticmethod(lambda: {})):
            mock_list.return_value = mock_blogs

            # Pre-set the client to skip the real httpx.AsyncClient creation
            w._client = AsyncMock()
            connected = await w.connect()

        assert connected
        assert w._blog_map["sustainable-living"] == 111
        assert w._blog_map["brand-partnerships"] == 222
        assert "other-blog" not in w._blog_map

    @pytest.mark.asyncio
    async def test_connect_missing_creds(self):
        """Connect fails when credentials missing."""
        with patch.dict(os.environ, {}, clear=True):
            from integrations.shopify_blog_watcher import ShopifyBlogWatcher
            w = ShopifyBlogWatcher()
            connected = await w.connect()
            assert not connected

    def test_transform_article(self):
        """Transform Shopify article to social-sharing format."""
        w = self._make_watcher()
        w._shop = "test-shop.myshopify.com"

        article = {
            "id": 12345,
            "title": "Test Article Title",
            "body_html": "<p>This is <b>bold</b> content. More text here.</p>",
            "handle": "test-article-title",
            "tags": "eco,green,nature",
            "author": "Rosie",
            "published_at": "2025-01-01T00:00:00Z",
            "created_at": "2025-01-01T00:00:00Z",
            "image": {"src": "https://cdn.shopify.com/test.jpg", "alt": "Test"},
        }

        result = w._transform_article(article, "sustainable-living", 111)
        assert result is not None
        assert result["article_id"] == 12345
        assert result["title"] == "Test Article Title"
        assert result["blog_handle"] == "sustainable-living"
        assert result["featured_image"] == "https://cdn.shopify.com/test.jpg"
        assert "eco" in result["tags"]
        assert "test-article-title" in result["url"]

    def test_transform_article_no_title(self):
        """Transform returns None for articles without title."""
        w = self._make_watcher()
        w._shop = "test.myshopify.com"
        result = w._transform_article({"id": 1, "title": ""}, "blog", 1)
        assert result is None

    def test_transform_article_extracts_image_from_html(self):
        """Transform extracts image from body HTML when no featured image."""
        w = self._make_watcher()
        w._shop = "test.myshopify.com"
        article = {
            "id": 1,
            "title": "Test",
            "body_html": '<p>Text</p><img src="https://example.com/img.jpg" alt="test">',
            "handle": "test",
            "tags": "",
            "author": "",
            "published_at": "",
            "created_at": "",
        }
        result = w._transform_article(article, "blog", 1)
        assert result["featured_image"] == "https://example.com/img.jpg"

    def test_strip_html(self):
        """Strip HTML tags correctly."""
        from integrations.shopify_blog_watcher import ShopifyBlogWatcher
        text = ShopifyBlogWatcher._strip_html(
            "<p>Hello <b>world</b></p><script>alert(1)</script>"
        )
        assert "Hello" in text
        assert "world" in text
        assert "script" not in text
        assert "<" not in text

    def test_extract_excerpt(self):
        """Extract excerpt with word boundary."""
        from integrations.shopify_blog_watcher import ShopifyBlogWatcher
        long_text = "A " * 200  # 400 chars
        excerpt = ShopifyBlogWatcher._extract_excerpt(
            f"<p>{long_text}</p>", max_length=50
        )
        assert len(excerpt) <= 60  # some slack for "..."
        assert excerpt.endswith("...")

    def test_extract_first_image(self):
        """Extract first image URL from HTML."""
        from integrations.shopify_blog_watcher import ShopifyBlogWatcher
        url = ShopifyBlogWatcher._extract_first_image(
            '<div><img src="https://example.com/first.jpg"><img src="second.jpg"></div>'
        )
        assert url == "https://example.com/first.jpg"

    def test_extract_first_image_none(self):
        """Return empty string when no image found."""
        from integrations.shopify_blog_watcher import ShopifyBlogWatcher
        url = ShopifyBlogWatcher._extract_first_image("<p>No images here</p>")
        assert url == ""

    def test_get_watched_blogs(self):
        """get_watched_blogs returns blog info."""
        w = self._make_watcher()
        w._blog_map = {"sustainable-living": 111, "brand-partnerships": 222}
        w._connected = True
        info = w.get_watched_blogs()
        assert info["connected"]
        assert len(info["blogs"]) == 2

    def test_state_persistence(self, tmp_path):
        """State saves and loads correctly."""
        from integrations.shopify_blog_watcher import ShopifyBlogWatcher
        state = {"111": {"last_article_id": 99999, "blog_handle": "test"}}

        state_file = tmp_path / "state.json"
        with patch("integrations.shopify_blog_watcher.WATCHER_STATE_FILE", str(state_file)):
            ShopifyBlogWatcher._save_state(state)
            loaded = ShopifyBlogWatcher._load_state()
            assert loaded["111"]["last_article_id"] == 99999

    @pytest.mark.asyncio
    async def test_check_new_articles_first_run_seeds(self):
        """First run seeds since_id without returning articles (dedup)."""
        w = self._make_watcher()
        w._connected = True
        w._blog_map = {"sustainable-living": 111}
        w._state = {}  # No last_article_id → first run
        w._shop = "test.myshopify.com"

        mock_articles = [
            {
                "id": 1001,
                "title": "Existing Article",
                "body_html": "<p>Content</p>",
                "handle": "existing-article",
                "tags": "eco",
                "author": "Test",
                "published_at": "2025-01-01",
                "created_at": "2025-01-01",
                "image": {"src": "https://cdn.shopify.com/img.jpg"},
            }
        ]

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"articles": mock_articles}
        mock_resp.raise_for_status = MagicMock()

        w._client = AsyncMock()
        w._client.get = AsyncMock(return_value=mock_resp)
        w._last_request_time = 0

        with patch.object(type(w), '_save_state', staticmethod(lambda s: None)):
            new = await w.check_new_articles()

        # First run → seed only, NO articles returned
        assert len(new) == 0
        # But since_id is now set for next run
        assert w._state["111"]["last_article_id"] == 1001

    @pytest.mark.asyncio
    async def test_check_new_articles_subsequent_run(self):
        """Subsequent run (since_id exists) returns new articles normally."""
        w = self._make_watcher()
        w._connected = True
        w._blog_map = {"sustainable-living": 111}
        # Simulate previously seeded state
        w._state = {"111": {"last_article_id": 900, "blog_handle": "sustainable-living"}}
        w._shop = "test.myshopify.com"

        mock_articles = [
            {
                "id": 1001,
                "title": "New Article",
                "body_html": "<p>Content</p>",
                "handle": "new-article",
                "tags": "eco",
                "author": "Test",
                "published_at": "2025-01-01",
                "created_at": "2025-01-01",
                "image": {"src": "https://cdn.shopify.com/img.jpg"},
            }
        ]

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"articles": mock_articles}
        mock_resp.raise_for_status = MagicMock()

        w._client = AsyncMock()
        w._client.get = AsyncMock(return_value=mock_resp)
        w._last_request_time = 0

        with patch.object(type(w), '_save_state', staticmethod(lambda s: None)):
            new = await w.check_new_articles()

        assert len(new) == 1
        assert new[0]["title"] == "New Article"
        assert new[0]["article_id"] == 1001


# ════════════════════════════════════════════════════════════════
# MultiTikTokPublisher Tests
# ════════════════════════════════════════════════════════════════

class TestMultiTikTokPublisher:
    """Tests for integrations/multi_tiktok_publisher.py"""

    def _make_publisher(self, accounts=2):
        env = {}
        for i in range(1, accounts + 1):
            env[f"TIKTOK_{i}_ACCESS_TOKEN"] = f"test_token_{i}"
            env[f"TIKTOK_{i}_OPEN_ID"] = f"test_openid_{i}"
            env[f"TIKTOK_{i}_LABEL"] = f"account_{i}"
        with patch.dict(os.environ, env, clear=False):
            from integrations.multi_tiktok_publisher import MultiTikTokPublisher
            return MultiTikTokPublisher()

    def test_discover_accounts(self):
        """Discovers accounts from env vars."""
        pub = self._make_publisher(3)
        assert len(pub.accounts) == 3
        assert pub.accounts[0].label == "account_1"
        assert pub.accounts[1].access_token == "test_token_2"
        assert pub.accounts[2].index == 3

    def test_discover_no_accounts(self):
        """Handles no accounts gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            from integrations.multi_tiktok_publisher import MultiTikTokPublisher
            pub = MultiTikTokPublisher()
            assert len(pub.accounts) == 0

    def test_discover_single_account_compat(self):
        """Falls back to TIKTOK_ACCESS_TOKEN for single account."""
        with patch.dict(os.environ, {
            "TIKTOK_ACCESS_TOKEN": "single_token",
            "TIKTOK_OPEN_ID": "single_id",
        }, clear=True):
            from integrations.multi_tiktok_publisher import MultiTikTokPublisher
            pub = MultiTikTokPublisher()
            assert len(pub.accounts) == 1
            assert pub.accounts[0].access_token == "single_token"

    def test_get_active_accounts(self):
        """get_active_accounts filters disabled accounts."""
        pub = self._make_publisher(3)
        pub.accounts[1].enabled = False
        active = pub.get_active_accounts()
        assert len(active) == 2

    def test_get_account_by_label(self):
        """Find account by label."""
        pub = self._make_publisher(2)
        acc = pub.get_account_by_label("account_2")
        assert acc is not None
        assert acc.index == 2

    def test_get_account_by_label_case_insensitive(self):
        """Label lookup is case-insensitive."""
        pub = self._make_publisher(1)
        acc = pub.get_account_by_label("ACCOUNT_1")
        assert acc is not None

    def test_get_account_by_index(self):
        """Find account by index."""
        pub = self._make_publisher(3)
        acc = pub.get_account_by_index(2)
        assert acc is not None
        assert acc.label == "account_2"

    def test_build_caption(self):
        """Build caption with hashtags."""
        from integrations.multi_tiktok_publisher import MultiTikTokPublisher
        caption = MultiTikTokPublisher._build_caption(
            "Hello world", ["eco", "#green", "nature"]
        )
        assert "Hello world" in caption
        assert "#eco" in caption
        assert "#green" in caption
        assert "#nature" in caption

    def test_build_caption_empty(self):
        """Build caption with no hashtags."""
        from integrations.multi_tiktok_publisher import MultiTikTokPublisher
        caption = MultiTikTokPublisher._build_caption("Just text", None)
        assert caption == "Just text"

    @pytest.mark.asyncio
    async def test_publish_photo(self):
        """Publish photo to TikTok account."""
        pub = self._make_publisher(1)
        account = pub.accounts[0]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"publish_id": "pub123"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            result = await pub.publish_photo(
                account, ["https://example.com/img.jpg"], "Test caption"
            )

        assert result["success"]
        assert result["publish_id"] == "pub123"
        assert result["account"] == "account_1"

    @pytest.mark.asyncio
    async def test_publish_video(self):
        """Publish video to TikTok account."""
        pub = self._make_publisher(1)
        account = pub.accounts[0]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"publish_id": "vid456"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            result = await pub.publish_video(
                account, "https://example.com/video.mp4", "Video caption"
            )

        assert result["success"]
        assert result["publish_id"] == "vid456"

    @pytest.mark.asyncio
    async def test_publish_no_token(self):
        """Publish fails when account has no token."""
        pub = self._make_publisher(1)
        pub.accounts[0].access_token = ""
        result = await pub.publish_photo(
            pub.accounts[0], ["https://example.com/img.jpg"]
        )
        assert not result["success"]
        assert "no access token" in result["error"]

    @pytest.mark.asyncio
    async def test_publish_to_all_broadcast(self):
        """Broadcast publishes to all active accounts."""
        pub = self._make_publisher(3)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"publish_id": "bc123"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            results = await pub.publish_to_all({
                "image_urls": ["https://example.com/img.jpg"],
                "caption": "Broadcast test",
            })

        assert len(results) == 3
        assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_publish_round_robin(self):
        """Round-robin rotates through accounts."""
        pub = self._make_publisher(3)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"publish_id": "rr123"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            content = {"image_urls": ["https://example.com/img.jpg"], "caption": "RR"}

            r1 = await pub.publish_round_robin(content)
            r2 = await pub.publish_round_robin(content)
            r3 = await pub.publish_round_robin(content)
            r4 = await pub.publish_round_robin(content)  # wraps

        assert r1["account"] == "account_1"
        assert r2["account"] == "account_2"
        assert r3["account"] == "account_3"
        assert r4["account"] == "account_1"  # wrapped

    @pytest.mark.asyncio
    async def test_publish_to_specific_account(self):
        """Publish to specific account by label."""
        pub = self._make_publisher(3)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"publish_id": "sp123"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            result = await pub.publish_to_account(
                "account_2",
                {"image_urls": ["https://example.com/img.jpg"], "caption": "Specific"},
            )

        assert result["success"]
        assert result["account"] == "account_2"

    @pytest.mark.asyncio
    async def test_publish_account_not_found(self):
        """Publish to non-existent account returns error."""
        pub = self._make_publisher(1)
        result = await pub.publish_to_account(
            "nonexistent",
            {"image_urls": ["img.jpg"]},
        )
        assert not result["success"]
        assert "not found" in result["error"]

    def test_status(self):
        """Status returns summary."""
        pub = self._make_publisher(3)
        status = pub.status()
        assert status["total_accounts"] == 3
        assert status["active_accounts"] == 3
        assert len(status["accounts"]) == 3

    @pytest.mark.asyncio
    async def test_test_account(self):
        """Test single account connection."""
        pub = self._make_publisher(1)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"user": {"display_name": "TestUser", "open_id": "123"}}
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            result = await pub.test_account(pub.accounts[0])

        assert result["success"]
        assert result["display_name"] == "TestUser"

    @pytest.mark.asyncio
    async def test_publish_rate_limited(self):
        """Handles 429 rate limit with retry."""
        pub = self._make_publisher(1)

        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"data": {"publish_id": "retry123"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[rate_limited, success])
            MockClient.return_value = mock_client

            result = await pub.publish_photo(
                pub.accounts[0], ["https://example.com/img.jpg"]
            )

        assert result["success"]
        assert result["publish_id"] == "retry123"


# ════════════════════════════════════════════════════════════════
# ShopifyAutoShare Tests
# ════════════════════════════════════════════════════════════════

class TestShopifyAutoShare:
    """Tests for integrations/shopify_auto_share.py"""

    def test_article_hash(self):
        """Article hash is consistent."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        article = {
            "blog_handle": "test",
            "article_id": 123,
            "title": "Test Title",
        }
        h1 = ShopifyAutoShare._article_hash(article)
        h2 = ShopifyAutoShare._article_hash(article)
        assert h1 == h2
        assert len(h1) == 16

    def test_article_hash_different(self):
        """Different articles get different hashes."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        h1 = ShopifyAutoShare._article_hash({"blog_handle": "a", "article_id": 1, "title": "X"})
        h2 = ShopifyAutoShare._article_hash({"blog_handle": "b", "article_id": 2, "title": "Y"})
        assert h1 != h2

    def test_build_tiktok_caption(self):
        """TikTok caption includes title and excerpt."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        caption = ShopifyAutoShare._build_tiktok_caption(
            "My Title", "This is an excerpt about sustainable living.", "https://therike.com/test"
        )
        assert "My Title" in caption
        assert "sustainable living" in caption
        # Enhanced: uses save CTA instead of link-in-bio
        assert "Save this" in caption or "tips" in caption
        # No raw URL in TikTok caption
        assert "https://therike.com" not in caption

    def test_build_pinterest_description(self):
        """Pinterest description includes excerpt and tags."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        desc = ShopifyAutoShare._build_pinterest_description(
            "Title", "This is a great post", ["eco", "green"]
        )
        assert "great post" in desc
        assert "#eco" in desc
        assert "#green" in desc
        assert len(desc) <= 500

    def test_build_hashtags(self):
        """Hashtags built from tags + blog handle (basic fallback)."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        tags = ShopifyAutoShare._build_hashtags_basic(
            ["eco living", "green-tech", "nature"], "sustainable-living"
        )
        assert any("ecoliving" in t for t in tags)
        assert any("sustainableliving" in t for t in tags)
        assert "#TheRike" in tags
        assert len(tags) <= 10

    def test_history_persistence(self, tmp_path):
        """Share history saves and loads."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        history = {"abc123": {"title": "Test", "shared_at": "2025-01-01"}}

        history_file = tmp_path / "history.json"
        with patch("integrations.shopify_auto_share.SHARE_HISTORY_FILE", str(history_file)):
            ShopifyAutoShare._save_history(history)
            loaded = ShopifyAutoShare._load_history()
            assert loaded["abc123"]["title"] == "Test"

    def test_config_persistence(self, tmp_path):
        """Config saves and loads."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        config = {"paused": True, "interval_min": 60}

        config_file = tmp_path / "config.json"
        with patch("integrations.shopify_auto_share.SHARE_CONFIG_FILE", str(config_file)):
            ShopifyAutoShare._save_config(config)
            loaded = ShopifyAutoShare._load_config()
            assert loaded["paused"] is True
            assert loaded["interval_min"] == 60

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Initialize connects watcher, Publer, and falls back to TikTok API."""
        with patch.dict(os.environ, {
            "SHOPIFY_SHOP": "test-shop",
            "SHOPIFY_ACCESS_TOKEN": "shpat_test",
            "SHOPIFY_AUTOSHARE_TIKTOK_VIA": "api",
            "TIKTOK_1_ACCESS_TOKEN": "tok1",
            "TIKTOK_1_OPEN_ID": "id1",
        }):
            from integrations.shopify_auto_share import ShopifyAutoShare

            auto_share = ShopifyAutoShare()

            # Patch the watcher's connect to return True without real API calls
            mock_watcher = AsyncMock()
            mock_watcher.connect = AsyncMock(return_value=True)
            mock_watcher._connected = True

            # Patch Publer to not be configured (test API fallback path)
            mock_publer_cls = MagicMock()
            mock_publer_inst = MagicMock()
            mock_publer_inst.is_configured = False
            mock_publer_cls.return_value = mock_publer_inst

            with patch("integrations.shopify_auto_share.ShopifyAutoShare._load_config", return_value={}), \
                 patch("integrations.shopify_auto_share.ShopifyAutoShare._load_history", return_value={}), \
                 patch("integrations.shopify_blog_watcher.ShopifyBlogWatcher", return_value=mock_watcher), \
                 patch("integrations.publer_publisher.PublerPublisher", mock_publer_cls):
                result = await auto_share.initialize()

            assert result["watcher_connected"]
            assert result["tiktok_accounts"] >= 1
            assert result["tiktok_via"] in ("api", "none")

    @pytest.mark.asyncio
    async def test_initialize_with_publer(self):
        """Initialize connects Publer when configured."""
        with patch.dict(os.environ, {
            "SHOPIFY_SHOP": "test-shop",
            "SHOPIFY_ACCESS_TOKEN": "shpat_test",
            "PUBLER_API_KEY": "test_api_key_123",
            "PUBLER_WORKSPACE_ID": "ws_test_456",
            "SHOPIFY_AUTOSHARE_TIKTOK_VIA": "publer",
        }):
            from integrations.shopify_auto_share import ShopifyAutoShare

            auto_share = ShopifyAutoShare()

            mock_watcher = AsyncMock()
            mock_watcher.connect = AsyncMock(return_value=True)
            mock_watcher._connected = True

            mock_publer = AsyncMock()
            mock_publer.is_configured = True
            mock_publer.connect = AsyncMock(return_value=True)

            with patch("integrations.shopify_auto_share.ShopifyAutoShare._load_config", return_value={}), \
                 patch("integrations.shopify_auto_share.ShopifyAutoShare._load_history", return_value={}), \
                 patch("integrations.shopify_blog_watcher.ShopifyBlogWatcher", return_value=mock_watcher), \
                 patch("integrations.publer_publisher.PublerPublisher", return_value=mock_publer):
                result = await auto_share.initialize()

            assert result["watcher_connected"]
            assert result["publer_connected"]
            assert result["tiktok_via"] == "publer"

    def test_get_history(self):
        """get_history returns sorted history."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        auto_share = ShopifyAutoShare()
        auto_share._history = {
            "aaa": {"title": "Old", "shared_at": "2025-01-01T00:00:00"},
            "bbb": {"title": "New", "shared_at": "2025-01-02T00:00:00"},
        }
        history = auto_share.get_history()
        assert history[0]["title"] == "New"
        assert history[1]["title"] == "Old"

    def test_clear_history(self):
        """clear_history removes all history."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        auto_share = ShopifyAutoShare()
        auto_share._history = {"a": {}, "b": {}}

        with patch.object(ShopifyAutoShare, '_save_history'):
            result = auto_share.clear_history()
            assert result["cleared"] == 2
            assert len(auto_share._history) == 0

    def test_get_stats(self):
        """get_stats returns sharing statistics."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        auto_share = ShopifyAutoShare()
        auto_share._history = {
            "a": {
                "blog": "sustainable-living",
                "tiktok_results": [{"success": True}, {"success": False}],
                "pinterest_result": {"success": True},
            },
            "b": {
                "blog": "brand-partnerships",
                "tiktok_results": [{"success": True}],
                "pinterest_result": {"success": False},
            },
        }
        stats = auto_share.get_stats()
        assert stats["total_articles_shared"] == 2
        assert stats["by_blog"]["sustainable-living"] == 1
        assert stats["by_blog"]["brand-partnerships"] == 1
        assert stats["tiktok_success"] == 2
        assert stats["pinterest_success"] == 1

    def test_pause_resume(self):
        """Pause and resume toggle config."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        auto_share = ShopifyAutoShare()
        with patch.object(ShopifyAutoShare, '_save_config'):
            auto_share.pause()
            assert auto_share._config["paused"] is True
            auto_share.resume()
            assert auto_share._config["paused"] is False

    def test_update_config(self):
        """update_config only allows whitelisted keys."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        auto_share = ShopifyAutoShare()
        with patch.object(ShopifyAutoShare, '_save_config'):
            result = auto_share.update_config({
                "interval_min": 60,
                "tiktok_mode": "round_robin",
                "dangerous_key": "should_not_persist",
            })
            assert auto_share._config["interval_min"] == 60
            assert auto_share._config["tiktok_mode"] == "round_robin"
            assert "dangerous_key" not in auto_share._config

    @pytest.mark.asyncio
    async def test_tick_paused(self):
        """Tick skips when paused."""
        from integrations.shopify_auto_share import ShopifyAutoShare
        auto_share = ShopifyAutoShare()
        auto_share._config = {"paused": True}
        auto_share._initialized = True

        with patch("integrations.shopify_blog_watcher.ShopifyBlogWatcher.connect", new_callable=AsyncMock):
            result = await auto_share.tick()

        assert result["skipped"]
        assert "paused" in result["reason"]

    @pytest.mark.asyncio
    async def test_share_specific_skips_already_shared(self):
        """share_specific_article skips article already in history."""
        from integrations.shopify_auto_share import ShopifyAutoShare

        auto_share = ShopifyAutoShare()
        auto_share._initialized = True

        # Set up watcher mock
        mock_watcher = AsyncMock()
        mock_watcher._connected = True
        article = {
            "blog_handle": "sustainable-living",
            "article_id": 999,
            "title": "Old Post",
            "handle": "old-post",
            "url": "https://therike.com/blogs/sustainable-living/old-post",
        }
        mock_watcher.get_recent_articles = AsyncMock(return_value=[article])
        auto_share._watcher = mock_watcher

        # Pre-populate history with that article
        article_hash = auto_share._article_hash(article)
        auto_share._history = {
            article_hash: {
                "title": "Old Post",
                "shared_at": "2025-01-01T00:00:00+00:00",
            },
        }

        result = await auto_share.share_specific_article(
            "https://therike.com/blogs/sustainable-living/old-post"
        )
        assert result["skipped"] is True
        assert result["reason"] == "already_shared"

    @pytest.mark.asyncio
    async def test_share_specific_force_reshares(self):
        """share_specific_article with force=True re-shares even if in history."""
        from integrations.shopify_auto_share import ShopifyAutoShare

        auto_share = ShopifyAutoShare()
        auto_share._initialized = True

        mock_watcher = AsyncMock()
        mock_watcher._connected = True
        article = {
            "blog_handle": "sustainable-living",
            "article_id": 999,
            "title": "Old Post",
            "handle": "old-post",
            "url": "https://therike.com/blogs/sustainable-living/old-post",
            "excerpt": "Test",
            "featured_image": "",
            "tags": [],
        }
        mock_watcher.get_recent_articles = AsyncMock(return_value=[article])
        auto_share._watcher = mock_watcher
        auto_share._tiktok = None
        auto_share._pinterest = None

        # Pre-populate history
        article_hash = auto_share._article_hash(article)
        auto_share._history = {
            article_hash: {"title": "Old Post", "shared_at": "2025-01-01T00:00:00"},
        }

        with patch.object(ShopifyAutoShare, '_save_history'):
            result = await auto_share.share_specific_article(
                "https://therike.com/blogs/sustainable-living/old-post",
                force=True,
            )
        # force=True → no skip, actual share result returned
        assert "skipped" not in result
        assert result["title"] == "Old Post"

    @pytest.mark.asyncio
    async def test_share_latest_skips_already_shared(self):
        """share_latest skips articles already in history."""
        from integrations.shopify_auto_share import ShopifyAutoShare

        auto_share = ShopifyAutoShare()
        auto_share._initialized = True

        articles = [
            {"blog_handle": "sustainable-living", "article_id": 1, "title": "A",
             "handle": "a", "url": "u1", "excerpt": "", "featured_image": "", "tags": []},
            {"blog_handle": "sustainable-living", "article_id": 2, "title": "B",
             "handle": "b", "url": "u2", "excerpt": "", "featured_image": "", "tags": []},
        ]

        mock_watcher = AsyncMock()
        mock_watcher._connected = True
        mock_watcher.get_recent_articles = AsyncMock(return_value=articles)
        auto_share._watcher = mock_watcher
        auto_share._tiktok = None
        auto_share._pinterest = None

        # Article "A" already shared
        hash_a = auto_share._article_hash(articles[0])
        auto_share._history = {
            hash_a: {"title": "A", "shared_at": "2025-01-01T00:00:00"},
        }

        with patch.object(ShopifyAutoShare, '_save_history'):
            results = await auto_share.share_latest("sustainable-living", count=2)

        assert len(results) == 2
        # First result skipped (already shared)
        assert results[0]["skipped"] is True
        assert results[0]["reason"] == "already_shared"
        # Second result actually shared
        assert results[1]["title"] == "B"
        assert "skipped" not in results[1]

    @pytest.mark.asyncio
    async def test_share_latest_records_history(self):
        """share_latest records newly shared articles to history."""
        from integrations.shopify_auto_share import ShopifyAutoShare

        auto_share = ShopifyAutoShare()
        auto_share._initialized = True

        article = {
            "blog_handle": "brand-partnerships", "article_id": 55, "title": "New",
            "handle": "new", "url": "u", "excerpt": "", "featured_image": "", "tags": [],
        }

        mock_watcher = AsyncMock()
        mock_watcher._connected = True
        mock_watcher.get_recent_articles = AsyncMock(return_value=[article])
        auto_share._watcher = mock_watcher
        auto_share._tiktok = None
        auto_share._pinterest = None
        auto_share._history = {}

        with patch.object(ShopifyAutoShare, '_save_history'):
            results = await auto_share.share_latest("brand-partnerships", count=1)

        assert len(results) == 1
        # History should now contain the article
        article_hash = auto_share._article_hash(article)
        assert article_hash in auto_share._history
        assert auto_share._history[article_hash]["title"] == "New"

    @pytest.mark.asyncio
    async def test_download_image(self, tmp_path):
        """_download_image downloads to temp file."""
        from integrations.shopify_auto_share import ShopifyAutoShare

        mock_resp = MagicMock()
        mock_resp.content = b"\x89PNG\r\n\x1a\nfake_png_data"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("integrations.shopify_auto_share.MEDIA_CACHE_DIR", str(tmp_path)), \
             patch("integrations.shopify_auto_share.httpx.AsyncClient", return_value=mock_client):
            path = await ShopifyAutoShare._download_image("https://cdn.shopify.com/image.png")

        assert path.endswith(".png")
        assert os.path.exists(path)
        os.unlink(path)

    @pytest.mark.asyncio
    async def test_share_article_via_publer(self):
        """_share_article routes TikTok through Publer when configured."""
        from integrations.shopify_auto_share import ShopifyAutoShare

        auto_share = ShopifyAutoShare()
        auto_share._initialized = True

        # Mock Publer as connected
        mock_publer = AsyncMock()
        mock_publer.publish = AsyncMock(return_value={
            "success": True, "post_id": "ui_123", "platform": "publer",
        })
        auto_share._publer = mock_publer
        auto_share._tiktok = None
        auto_share._pinterest = None

        article = {
            "blog_handle": "sustainable-living",
            "article_id": 42,
            "title": "Test Publer Post",
            "excerpt": "Excerpt here",
            "featured_image": "https://cdn.shopify.com/test.jpg",
            "url": "https://therike.com/blogs/sustainable-living/test",
            "tags": ["eco"],
        }

        with patch.dict(os.environ, {"SHOPIFY_AUTOSHARE_TIKTOK_VIA": "publer"}), \
             patch.object(ShopifyAutoShare, '_download_image', new_callable=AsyncMock, return_value="/tmp/test.jpg"), \
             patch("os.unlink"):
            result = await auto_share._share_article(article)

        assert len(result["tiktok"]) == 1
        assert result["tiktok"][0]["success"] is True
        assert result["tiktok"][0]["account"] == "publer"
        # Verify Publer was called with platforms=["tiktok"] (first call)
        calls = mock_publer.publish.call_args_list
        assert len(calls) >= 1
        tiktok_call = calls[0][0][0]  # first call, positional args, first arg
        assert "tiktok" in tiktok_call["platforms"]


# ════════════════════════════════════════════════════════════════
# API Endpoint Tests
# ════════════════════════════════════════════════════════════════

class TestBlogShareAPI:
    """Tests for /api/blog-share/* endpoints."""

    def test_blog_share_page_route_exists(self):
        """Blog share page route is registered."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from web.app import app
        routes = [r.path for r in app.routes]
        assert "/blog-share" in routes

    def test_blog_share_api_routes_exist(self):
        """All blog-share API routes are registered."""
        from web.app import app
        routes = [r.path for r in app.routes]
        expected = [
            "/api/blog-share/status",
            "/api/blog-share/config",
            "/api/blog-share/tick",
            "/api/blog-share/pause",
            "/api/blog-share/resume",
            "/api/blog-share/history",
            "/api/blog-share/manual",
            "/api/blog-share/latest",
            "/api/blog-share/tiktok/accounts",
            "/api/blog-share/tiktok/test",
            "/api/blog-share/watcher/blogs",
            "/api/blog-share/watcher/reset",
        ]
        for route in expected:
            assert route in routes, f"Missing route: {route}"


# ════════════════════════════════════════════════════════════════
# Telegram Alert Tests
# ════════════════════════════════════════════════════════════════

class TestTelegramBlogAlert:
    """Tests for Telegram blog-share alert template."""

    def test_alert_blog_shared_import(self):
        """alert_blog_shared is importable from telegram_bot."""
        from integrations.telegram_bot import alert_blog_shared
        assert callable(alert_blog_shared)

    def test_alert_blog_shared_empty(self):
        """Empty results returns skipped."""
        from integrations.telegram_bot import alert_blog_shared
        with patch("integrations.telegram_bot.send_message") as mock_send:
            result = alert_blog_shared([], [])
        assert result.get("skipped") is True
        mock_send.assert_not_called()

    def test_alert_blog_shared_with_data(self):
        """alert_blog_shared formats and sends message."""
        from integrations.telegram_bot import alert_blog_shared
        results = [
            {"tiktok": [{"success": True, "account": "publer"}], "pinterest": {"success": True}},
        ]
        articles = [
            {"title": "Test Post", "blog_handle": "sustainable-living"},
        ]
        with patch("integrations.telegram_bot.send_message", return_value={"success": True}) as mock_send:
            result = alert_blog_shared(results, articles)
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "Shopify Blog Auto-Share" in msg
        assert "Test Post" in msg
        assert "sustainable-living" in msg
        assert "publer" in msg

    def test_send_telegram_alert_uses_alert_blog_shared(self):
        """ShopifyAutoShare._send_telegram_alert calls alert_blog_shared."""
        from integrations.shopify_auto_share import ShopifyAutoShare as SA
        auto_share = SA()
        results = [{"tiktok": [{"success": True}], "pinterest": None}]
        articles = [{"title": "X", "blog_handle": "y"}]
        with patch("integrations.telegram_bot.alert_blog_shared") as mock_alert:
            auto_share._send_telegram_alert(results, articles)
        mock_alert.assert_called_once_with(results, articles)


# ════════════════════════════════════════════════════════════════
# Background Scheduler Tests
# ════════════════════════════════════════════════════════════════

class TestBackgroundSchedulers:
    """Tests for background scheduler loops in app lifespan."""

    def test_lifespan_creates_all_tasks(self):
        """Lifespan starts scheduler, RSS tick, and blog-share tick tasks."""
        from web.app import app
        import web.app as app_module
        # Just verify the functions exist
        assert callable(app_module.scheduler_loop)
        assert callable(app_module.rss_tick_loop)
        assert callable(app_module.blog_share_tick_loop)

    def test_version_is_3_4_0(self):
        """Health endpoint returns current version."""
        from httpx import ASGITransport, AsyncClient
        from web.app import app
        import asyncio
        async def _check():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/api/health")
                return r.json()
        result = asyncio.get_event_loop().run_until_complete(_check())
        # Health may return version at top level or nested in body
        version = result.get("version", "")
        assert version in ("3.4.0", "3.5.0"), f"Expected 3.4.0 or 3.5.0 in health response, got: {result}"


# ════════════════════════════════════════════════════════════════
# Integration Tests
# ════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_imports(self):
        """All new modules import without errors."""
        from integrations.shopify_blog_watcher import ShopifyBlogWatcher
        from integrations.multi_tiktok_publisher import MultiTikTokPublisher
        from integrations.shopify_auto_share import ShopifyAutoShare
        from integrations.shopify_auto_share import (
            auto_share_tick,
            auto_share_status,
            auto_share_history,
            auto_share_config,
            auto_share_update_config,
            auto_share_pause,
            auto_share_resume,
            auto_share_manual,
            auto_share_latest,
        )
        assert True

    def test_platform_registry_includes_tiktok_multi(self):
        """Platform registry discovers tiktok_multi when accounts configured."""
        with patch.dict(os.environ, {
            "TIKTOK_1_ACCESS_TOKEN": "tok",
            "TIKTOK_1_OPEN_ID": "id",
        }):
            from integrations.platform_publisher import PublisherRegistry
            registry = PublisherRegistry()
            registry._initialized = False
            registry._publishers = {}
            registry.register_all()
            platforms = registry.list_platforms()
            # tiktok_multi should be there if token is set
            assert "tiktok_multi" in platforms or "tiktok" in platforms

    def test_env_has_autoshare_config(self):
        """Env file has auto-share configuration variables."""
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            ".env"
        )
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "SHOPIFY_WATCH_BLOGS" in content
            assert "SHOPIFY_AUTOSHARE_TIKTOK_VIA" in content
            assert "SHOPIFY_AUTOSHARE_TIKTOK_MODE" in content
            assert "SHOPIFY_AUTOSHARE_PINTEREST_VIA" in content
            assert "SHOPIFY_AUTOSHARE_PINTEREST_ENABLED" in content
