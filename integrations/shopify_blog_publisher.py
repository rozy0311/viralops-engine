"""
ViralOps Engine -- Shopify Blog Publisher (REAL Implementation)

Shopify Admin REST API -- blog article CRUD.
Syncs content as published/draft blog articles to a Shopify store.

API Docs: https://shopify.dev/docs/api/admin-rest/2025-01/resources/article
Auth: Custom App Admin API access token (X-Shopify-Access-Token header)
Rate limit: 2 requests/second (bucket leak, 40 burst capacity)

SETUP:
1. Go to Shopify Admin -> Settings -> Apps and sales channels -> Develop apps
2. Create a Custom App with "read_content" + "write_content" scopes
3. Install the app and copy the Admin API access token
4. Set env vars: SHOPIFY_SHOP, SHOPIFY_ACCESS_TOKEN, SHOPIFY_BLOG_ID
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from core.models import PublishResult, QueueItem

logger = logging.getLogger("viralops.publisher.shopify_blog")


class ShopifyBlogPublisher:
    """Real Shopify Admin REST API publisher for blog articles."""

    platform = "shopify_blog"
    API_VERSION = "2025-01"

    def __init__(self, account_id: str = "shopify_main"):
        self.account_id = account_id
        self._shop: str | None = None
        self._token: str | None = None
        self._blog_id: str | None = None
        self._base_url: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0

    async def connect(self) -> bool:
        """Set up Shopify Admin API connection and verify blog exists."""
        prefix = self.account_id.upper().replace("-", "_")
        self._shop = os.environ.get(
            f"{prefix}_SHOP", os.environ.get("SHOPIFY_SHOP")
        )
        self._token = os.environ.get(
            f"{prefix}_ACCESS_TOKEN",
            os.environ.get("SHOPIFY_ACCESS_TOKEN"),
        )
        self._blog_id = os.environ.get(
            f"{prefix}_BLOG_ID", os.environ.get("SHOPIFY_BLOG_ID")
        )

        if not all([self._shop, self._token, self._blog_id]):
            logger.error(
                "Shopify [%s]: Missing SHOPIFY_SHOP, SHOPIFY_ACCESS_TOKEN, "
                "or SHOPIFY_BLOG_ID",
                self.account_id,
            )
            return False

        # Normalize shop domain
        shop = self._shop.strip()
        if not shop.endswith(".myshopify.com"):
            shop = f"{shop}.myshopify.com"
        self._shop = shop
        self._base_url = (
            f"https://{self._shop}/admin/api/{self.API_VERSION}"
        )

        self._client = httpx.AsyncClient(
            headers={
                "X-Shopify-Access-Token": self._token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

        try:
            # Verify blog exists
            resp = await self._rate_limited_get(
                f"{self._base_url}/blogs/{self._blog_id}.json"
            )
            resp.raise_for_status()
            blog_data = resp.json().get("blog", {})
            blog_title = blog_data.get("title", "Unknown")

            self._connected = True
            logger.info(
                "Shopify [%s]: Connected to store '%s', blog '%s' (id=%s)",
                self.account_id,
                self._shop,
                blog_title,
                self._blog_id,
            )
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error(
                    "Shopify [%s]: Invalid access token", self.account_id
                )
            elif e.response.status_code == 404:
                logger.error(
                    "Shopify [%s]: Blog ID %s not found",
                    self.account_id,
                    self._blog_id,
                )
            else:
                logger.error(
                    "Shopify [%s]: Connection failed: HTTP %s",
                    self.account_id,
                    e.response.status_code,
                )
            return False
        except httpx.HTTPError as e:
            logger.error(
                "Shopify [%s]: Connection error: %s", self.account_id, e
            )
            return False

    async def publish(self, item: QueueItem, content: dict) -> PublishResult:
        """
        Create a blog article in the Shopify store.

        content keys:
            title (str): Article title (required)
            body_html (str): Article body in HTML (required)
            body (str): Fallback for body_html (plain text or HTML)
            summary_html (str): Article excerpt/summary HTML
            tags (str | list): Comma-separated tags or list
            handle (str): URL-friendly slug (auto-generated if empty)
            published (bool): True=published, False=draft (default True)
            image_url (str): Featured image URL
            image_alt (str): Featured image alt text
            author (str): Author name
            seo_title (str): Meta title for SEO
            seo_description (str): Meta description for SEO
            template_suffix (str): Custom Liquid template suffix
        """
        if not self._connected or not self._client:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Shopify not connected",
            )

        title = content.get("title", content.get("caption", ""))
        body_html = content.get(
            "body_html", content.get("body", content.get("text", ""))
        )

        if not title:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Article title is required",
            )

        # Build article payload
        article: dict[str, Any] = {
            "title": title,
            "body_html": body_html,
            "published": content.get("published", True),
        }

        # Handle
        handle = content.get("handle", "")
        if not handle:
            handle = self._slugify(title)
        article["handle"] = handle

        # Tags
        tags = content.get("tags", "")
        if isinstance(tags, list):
            tags = ", ".join(tags)
        if tags:
            article["tags"] = tags

        # Summary/excerpt
        if content.get("summary_html"):
            article["summary_html"] = content["summary_html"]

        # Author
        if content.get("author"):
            article["author"] = content["author"]

        # Featured image
        if content.get("image_url"):
            article["image"] = {
                "src": content["image_url"],
                "alt": content.get("image_alt", title),
            }

        # SEO metafields
        if content.get("seo_title"):
            article["metafields_global_title_tag"] = content["seo_title"]
        if content.get("seo_description"):
            article["metafields_global_description_tag"] = content[
                "seo_description"
            ]

        # Template suffix
        if content.get("template_suffix"):
            article["template_suffix"] = content["template_suffix"]

        payload = {"article": article}

        try:
            resp = await self._rate_limited_post(
                f"{self._base_url}/blogs/{self._blog_id}/articles.json",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json().get("article", {})

            article_id = str(data.get("id", ""))
            article_handle = data.get("handle", handle)
            # Build public URL
            store_domain = self._shop.replace(".myshopify.com", "")
            post_url = f"https://{self._shop}/blogs/{self._blog_id}/{article_handle}"

            # If store has custom domain, URL might differ
            # But myshopify.com URL always works for admin
            admin_url = (
                f"https://{self._shop}/admin/articles/{article_id}"
            )

            logger.info(
                "Shopify [%s]: Article '%s' created (id=%s, handle=%s)",
                self.account_id,
                title[:50],
                article_id,
                article_handle,
            )
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=True,
                published_at=datetime.now(timezone.utc),
                post_url=post_url,
                post_id=article_id,
                metadata={
                    "admin_url": admin_url,
                    "handle": article_handle,
                    "published": content.get("published", True),
                },
            )

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = e.response.text[:300]
            if status == 429:
                retry_after = e.response.headers.get("Retry-After", "2")
                logger.warning(
                    "Shopify [%s]: Rate limited, retry after %ss",
                    self.account_id,
                    retry_after,
                )
            elif status == 422:
                logger.error(
                    "Shopify [%s]: Validation error: %s",
                    self.account_id,
                    detail,
                )
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=f"HTTP {status}: {detail}",
            )
        except Exception as e:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=str(e),
            )

    async def update_article(
        self, article_id: str, content: dict
    ) -> PublishResult:
        """Update an existing blog article."""
        if not self._connected or not self._client:
            return PublishResult(
                queue_item_id="",
                platform=self.platform,
                success=False,
                error="Shopify not connected",
            )

        article: dict[str, Any] = {"id": int(article_id)}

        if "title" in content:
            article["title"] = content["title"]
        if "body_html" in content or "body" in content:
            article["body_html"] = content.get("body_html", content.get("body"))
        if "tags" in content:
            tags = content["tags"]
            if isinstance(tags, list):
                tags = ", ".join(tags)
            article["tags"] = tags
        if "published" in content:
            article["published"] = content["published"]
        if "image_url" in content:
            article["image"] = {
                "src": content["image_url"],
                "alt": content.get("image_alt", ""),
            }
        if "seo_title" in content:
            article["metafields_global_title_tag"] = content["seo_title"]
        if "seo_description" in content:
            article["metafields_global_description_tag"] = content["seo_description"]

        try:
            resp = await self._rate_limited_put(
                f"{self._base_url}/articles/{article_id}.json",
                json={"article": article},
            )
            resp.raise_for_status()
            data = resp.json().get("article", {})
            logger.info(
                "Shopify [%s]: Article %s updated", self.account_id, article_id
            )
            return PublishResult(
                queue_item_id="",
                platform=self.platform,
                success=True,
                post_id=article_id,
                post_url=f"https://{self._shop}/admin/articles/{article_id}",
            )
        except Exception as e:
            return PublishResult(
                queue_item_id="",
                platform=self.platform,
                success=False,
                error=str(e),
            )

    async def list_articles(
        self,
        limit: int = 50,
        published_status: str = "any",
        since_id: str | None = None,
    ) -> list[dict]:
        """List blog articles with pagination support."""
        if not self._connected or not self._client:
            return []

        params: dict[str, Any] = {
            "limit": min(limit, 250),
            "published_status": published_status,
        }
        if since_id:
            params["since_id"] = since_id

        try:
            resp = await self._rate_limited_get(
                f"{self._base_url}/blogs/{self._blog_id}/articles.json",
                params=params,
            )
            resp.raise_for_status()
            return resp.json().get("articles", [])
        except Exception as e:
            logger.error(
                "Shopify [%s]: Failed to list articles: %s",
                self.account_id, e,
            )
            return []

    async def get_article(self, article_id: str) -> dict | None:
        """Get a single article by ID."""
        if not self._connected or not self._client:
            return None
        try:
            resp = await self._rate_limited_get(
                f"{self._base_url}/articles/{article_id}.json"
            )
            resp.raise_for_status()
            return resp.json().get("article")
        except Exception as e:
            logger.error(
                "Shopify [%s]: Failed to get article %s: %s",
                self.account_id, article_id, e,
            )
            return None

    async def delete_article(self, article_id: str) -> bool:
        """Delete a blog article."""
        if not self._connected or not self._client:
            return False
        try:
            resp = await self._rate_limited_request(
                "DELETE", f"{self._base_url}/articles/{article_id}.json"
            )
            resp.raise_for_status()
            logger.info(
                "Shopify [%s]: Article %s deleted", self.account_id, article_id
            )
            return True
        except Exception as e:
            logger.error(
                "Shopify [%s]: Failed to delete article %s: %s",
                self.account_id, article_id, e,
            )
            return False

    # --- Rate limiting helpers ---

    async def _rate_limited_get(
        self, url: str, **kwargs: Any
    ) -> httpx.Response:
        """GET request with 2 req/sec rate limit."""
        return await self._rate_limited_request("GET", url, **kwargs)

    async def _rate_limited_post(
        self, url: str, **kwargs: Any
    ) -> httpx.Response:
        """POST request with 2 req/sec rate limit."""
        return await self._rate_limited_request("POST", url, **kwargs)

    async def _rate_limited_put(
        self, url: str, **kwargs: Any
    ) -> httpx.Response:
        """PUT request with 2 req/sec rate limit."""
        return await self._rate_limited_request("PUT", url, **kwargs)

    async def _rate_limited_request(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        """Execute HTTP request with Shopify 2 req/sec rate limiting."""
        import asyncio

        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < 0.5:  # 2 req/sec = 0.5s between requests
            await asyncio.sleep(0.5 - elapsed)

        self._last_request_time = time.monotonic()
        return await self._client.request(method, url, **kwargs)

    # --- Utility ---

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL-friendly handle."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text[:200]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False