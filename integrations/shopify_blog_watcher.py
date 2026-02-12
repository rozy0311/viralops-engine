"""
Shopify Blog Watcher — Poll Shopify Admin REST API for new blog articles.

Watches specific blogs by handle (e.g. "sustainable-living", "brand-partnerships")
and returns newly published articles for auto-sharing to social platforms.

Features:
  - Resolve blog handles → Shopify blog IDs automatically
  - Track last-seen article per blog (persistent JSON)
  - Rate-limited requests (2 req/sec Shopify limit)
  - Multi-blog support — watch any number of blogs simultaneously
  - Extract featured image, tags, excerpt for social sharing
  - Configurable poll interval

API: Shopify Admin REST API 2025-01
Docs: https://shopify.dev/docs/api/admin-rest/2025-01/resources/article
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger("viralops.shopify_blog_watcher")

# ── Persistence ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
WATCHER_STATE_FILE = os.path.join(DATA_DIR, "shopify_blog_watcher_state.json")

# ── Shopify API ──────────────────────────────────────────────
API_VERSION = "2025-01"
RATE_LIMIT_DELAY = 0.5  # 2 req/sec


class ShopifyBlogWatcher:
    """
    Watches Shopify blogs for new articles.

    Usage:
        watcher = ShopifyBlogWatcher()
        await watcher.connect()
        new_articles = await watcher.check_new_articles()
    """

    def __init__(
        self,
        blog_handles: list[str] | None = None,
        shop: str | None = None,
        token: str | None = None,
    ):
        self._shop = shop or os.environ.get("SHOPIFY_SHOP", "")
        self._token = token or os.environ.get("SHOPIFY_ACCESS_TOKEN", "")
        self._blog_handles = blog_handles or self._default_blog_handles()
        self._base_url: str = ""
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._last_request_time: float = 0

        # blog_handle → blog_id mapping (resolved on connect)
        self._blog_map: dict[str, int] = {}

        # Persistent state: blog_id → last_seen_article_id
        self._state: dict[str, Any] = {}

    @staticmethod
    def _default_blog_handles() -> list[str]:
        """Default blogs to watch from env or hardcoded."""
        env_val = os.environ.get("SHOPIFY_WATCH_BLOGS", "")
        if env_val:
            return [h.strip() for h in env_val.split(",") if h.strip()]
        return ["sustainable-living", "brand-partnerships"]

    # ════════════════════════════════════════════
    # Connection
    # ════════════════════════════════════════════

    async def connect(self) -> bool:
        """Connect to Shopify and resolve blog handles → IDs."""
        if not self._shop or not self._token:
            logger.error("ShopifyBlogWatcher: Missing SHOPIFY_SHOP or SHOPIFY_ACCESS_TOKEN")
            return False

        shop = self._shop.strip()
        if not shop.endswith(".myshopify.com"):
            shop = f"{shop}.myshopify.com"
        self._shop = shop
        self._base_url = f"https://{self._shop}/admin/api/{API_VERSION}"

        self._client = httpx.AsyncClient(
            headers={
                "X-Shopify-Access-Token": self._token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

        # Load persisted state
        self._state = self._load_state()

        # Resolve blog handles → IDs
        try:
            blogs = await self._list_blogs()
            for blog in blogs:
                handle = blog.get("handle", "")
                blog_id = blog.get("id")
                if handle in self._blog_handles:
                    self._blog_map[handle] = blog_id
                    logger.info(
                        "ShopifyBlogWatcher: Mapped blog '%s' → ID %s (title: %s)",
                        handle, blog_id, blog.get("title", ""),
                    )

            missing = set(self._blog_handles) - set(self._blog_map.keys())
            if missing:
                logger.warning(
                    "ShopifyBlogWatcher: Could not find blogs: %s", missing
                )

            self._connected = bool(self._blog_map)
            if self._connected:
                logger.info(
                    "ShopifyBlogWatcher: Connected — watching %d blogs: %s",
                    len(self._blog_map), list(self._blog_map.keys()),
                )
            return self._connected

        except Exception as e:
            logger.error("ShopifyBlogWatcher: Connection failed: %s", e)
            return False

    async def _list_blogs(self) -> list[dict]:
        """GET /admin/api/2025-01/blogs.json — list all blogs in store."""
        resp = await self._rate_limited_get(f"{self._base_url}/blogs.json")
        resp.raise_for_status()
        return resp.json().get("blogs", [])

    # ════════════════════════════════════════════
    # Core — Check for New Articles
    # ════════════════════════════════════════════

    async def check_new_articles(
        self, max_per_blog: int = 10
    ) -> list[dict]:
        """
        Check all watched blogs for new articles since last check.

        Returns list of new articles with metadata for social sharing:
        [
            {
                "blog_handle": "sustainable-living",
                "blog_id": 12345,
                "article_id": 67890,
                "title": "...",
                "body_html": "...",
                "excerpt": "...",
                "featured_image": "https://...",
                "tags": ["tag1", "tag2"],
                "author": "...",
                "published_at": "2025-01-01T00:00:00Z",
                "url": "https://therike.com/blogs/sustainable-living/article-handle",
                "handle": "article-handle",
            }
        ]
        """
        if not self._connected or not self._client:
            logger.error("ShopifyBlogWatcher: Not connected")
            return []

        all_new = []

        for handle, blog_id in self._blog_map.items():
            try:
                new_articles = await self._fetch_new_for_blog(
                    handle, blog_id, max_per_blog
                )
                all_new.extend(new_articles)
            except Exception as e:
                logger.error(
                    "ShopifyBlogWatcher: Error checking blog '%s': %s",
                    handle, e,
                )

        if all_new:
            self._save_state(self._state)
            logger.info(
                "ShopifyBlogWatcher: Found %d new articles across %d blogs",
                len(all_new), len(self._blog_map),
            )

        return all_new

    async def _fetch_new_for_blog(
        self, handle: str, blog_id: int, limit: int
    ) -> list[dict]:
        """Fetch new articles for a single blog."""
        state_key = str(blog_id)
        last_seen_id = self._state.get(state_key, {}).get("last_article_id")

        params: dict[str, Any] = {
            "limit": min(limit, 250),
            "published_status": "published",
            "order": "created_at desc",
        }
        if last_seen_id:
            params["since_id"] = last_seen_id

        resp = await self._rate_limited_get(
            f"{self._base_url}/blogs/{blog_id}/articles.json",
            params=params,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        if not articles:
            return []

        # Update last seen ID (highest = newest)
        newest_id = max(a["id"] for a in articles)
        self._state[state_key] = {
            "last_article_id": newest_id,
            "blog_handle": handle,
            "last_checked": datetime.now(timezone.utc).isoformat(),
            "articles_found": len(articles),
        }

        # Transform to social-sharing format
        new_articles = []
        for article in articles:
            transformed = self._transform_article(article, handle, blog_id)
            if transformed:
                new_articles.append(transformed)

        logger.info(
            "ShopifyBlogWatcher: Blog '%s' — %d new articles (since_id=%s)",
            handle, len(new_articles), last_seen_id or "none",
        )
        return new_articles

    def _transform_article(
        self, article: dict, blog_handle: str, blog_id: int
    ) -> dict | None:
        """Transform Shopify article → social-sharing format."""
        article_id = article.get("id")
        title = article.get("title", "").strip()
        if not title:
            return None

        body_html = article.get("body_html", "")
        excerpt = self._extract_excerpt(body_html, max_length=300)
        plain_text = self._strip_html(body_html)

        # Featured image
        image = article.get("image", {})
        featured_image = ""
        if image:
            featured_image = image.get("src", "")
        if not featured_image:
            # Try extracting first image from body HTML
            featured_image = self._extract_first_image(body_html)

        # Tags
        tags_raw = article.get("tags", "")
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        else:
            tags = list(tags_raw) if tags_raw else []

        # Build public URL
        article_handle = article.get("handle", "")
        store_domain = self._shop.replace(".myshopify.com", "")
        # TheRike custom domain: therike.com
        custom_domain = os.environ.get("SHOPIFY_CUSTOM_DOMAIN", "therike.com")
        public_url = f"https://{custom_domain}/blogs/{blog_handle}/{article_handle}"

        return {
            "blog_handle": blog_handle,
            "blog_id": blog_id,
            "article_id": article_id,
            "title": title,
            "body_html": body_html,
            "body_text": plain_text,
            "excerpt": excerpt,
            "featured_image": featured_image,
            "tags": tags,
            "author": article.get("author", ""),
            "published_at": article.get("published_at", ""),
            "created_at": article.get("created_at", ""),
            "url": public_url,
            "handle": article_handle,
            "admin_url": f"https://{self._shop}/admin/articles/{article_id}",
        }

    # ════════════════════════════════════════════
    # Blog Info Helpers
    # ════════════════════════════════════════════

    async def list_all_blogs(self) -> list[dict]:
        """List all blogs in the Shopify store."""
        if not self._connected:
            return []
        blogs = await self._list_blogs()
        return [
            {
                "id": b.get("id"),
                "handle": b.get("handle"),
                "title": b.get("title"),
                "tags": b.get("tags", ""),
                "commentable": b.get("commentable"),
            }
            for b in blogs
        ]

    async def get_recent_articles(
        self, blog_handle: str, limit: int = 5
    ) -> list[dict]:
        """Get recent articles from a specific blog."""
        if not self._connected:
            return []
        blog_id = self._blog_map.get(blog_handle)
        if not blog_id:
            return []

        resp = await self._rate_limited_get(
            f"{self._base_url}/blogs/{blog_id}/articles.json",
            params={"limit": limit, "published_status": "published"},
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            self._transform_article(a, blog_handle, blog_id)
            for a in articles
            if self._transform_article(a, blog_handle, blog_id)
        ]

    def get_watched_blogs(self) -> dict:
        """Return currently watched blogs and their state."""
        return {
            "blogs": [
                {
                    "handle": handle,
                    "blog_id": blog_id,
                    "last_article_id": self._state.get(str(blog_id), {}).get(
                        "last_article_id"
                    ),
                    "last_checked": self._state.get(str(blog_id), {}).get(
                        "last_checked"
                    ),
                }
                for handle, blog_id in self._blog_map.items()
            ],
            "connected": self._connected,
        }

    def reset_state(self, blog_handle: str | None = None) -> dict:
        """Reset watcher state (re-fetch all articles on next check)."""
        if blog_handle:
            blog_id = self._blog_map.get(blog_handle)
            if blog_id:
                self._state.pop(str(blog_id), None)
                self._save_state(self._state)
                return {"reset": blog_handle}
        else:
            self._state = {}
            self._save_state(self._state)
            return {"reset": "all"}
        return {"error": f"Blog '{blog_handle}' not found"}

    # ════════════════════════════════════════════
    # State Persistence
    # ════════════════════════════════════════════

    @staticmethod
    def _load_state() -> dict:
        try:
            if os.path.exists(WATCHER_STATE_FILE):
                with open(WATCHER_STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("ShopifyBlogWatcher: Failed to load state: %s", e)
        return {}

    @staticmethod
    def _save_state(state: dict):
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(WATCHER_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("ShopifyBlogWatcher: Failed to save state: %s", e)

    # ════════════════════════════════════════════
    # Rate-Limited HTTP
    # ════════════════════════════════════════════

    async def _rate_limited_get(
        self, url: str, **kwargs: Any
    ) -> httpx.Response:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()
        return await self._client.get(url, **kwargs)

    # ════════════════════════════════════════════
    # Text Processing Helpers
    # ════════════════════════════════════════════

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        html = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", "", html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _extract_excerpt(html: str, max_length: int = 300) -> str:
        """Extract clean text excerpt from HTML."""
        text = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", "", html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_length:
            return text
        excerpt = text[:max_length]
        last_space = excerpt.rfind(" ")
        if last_space > max_length * 0.7:
            excerpt = excerpt[:last_space]
        return excerpt.rstrip(".") + "..."

    @staticmethod
    def _extract_first_image(html: str) -> str:
        """Extract first image URL from HTML content."""
        match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
        return match.group(1) if match else ""

    # ════════════════════════════════════════════
    # Cleanup
    # ════════════════════════════════════════════

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
