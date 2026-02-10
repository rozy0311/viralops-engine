"""
ViralOps Engine -- Tumblr Publisher (REAL Implementation)

Tumblr API v2 -- direct blog posting via OAuth2.
Supports: text (NPF), link, photo, quote posts.

API Docs: https://www.tumblr.com/docs/en/api/v2
Auth: OAuth2 Bearer token
Rate limit: 250 new posts/day, 150 photo uploads/day

SETUP:
1. Go to https://www.tumblr.com/oauth/apps
2. Register an application
3. Generate OAuth2 token (or use API key + secret for OAuth1)
4. Set env vars: TUMBLR_TOKEN, TUMBLR_BLOG_NAME
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import httpx

from core.models import PublishResult, QueueItem

logger = logging.getLogger("viralops.publisher.tumblr")


class TumblrPublisher:
    """Real Tumblr API v2 publisher using OAuth2 Bearer token."""

    platform = "tumblr"
    API_BASE = "https://api.tumblr.com/v2"

    def __init__(self, account_id: str = "tumblr_main"):
        self.account_id = account_id
        self._token: str | None = None
        self._blog_name: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        """Set up OAuth2 bearer token and verify blog access."""
        prefix = self.account_id.upper().replace("-", "_")
        self._token = os.environ.get(
            f"{prefix}_TOKEN", os.environ.get("TUMBLR_TOKEN")
        )
        self._blog_name = os.environ.get(
            f"{prefix}_BLOG_NAME", os.environ.get("TUMBLR_BLOG_NAME")
        )

        if not self._token or not self._blog_name:
            logger.error(
                "Tumblr [%s]: Missing TUMBLR_TOKEN or TUMBLR_BLOG_NAME",
                self.account_id,
            )
            return False

        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        try:
            # Verify blog exists and we have access
            resp = await self._client.get(f"/blog/{self._blog_name}/info")
            resp.raise_for_status()
            blog_info = resp.json().get("response", {}).get("blog", {})
            blog_title = blog_info.get("title", self._blog_name)

            self._connected = True
            logger.info(
                "Tumblr [%s]: Connected to blog '%s' (%s.tumblr.com)",
                self.account_id,
                blog_title,
                self._blog_name,
            )
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Tumblr [%s]: Invalid or expired token", self.account_id)
            elif e.response.status_code == 404:
                logger.error(
                    "Tumblr [%s]: Blog '%s' not found",
                    self.account_id,
                    self._blog_name,
                )
            else:
                logger.error(
                    "Tumblr [%s]: Connection failed: HTTP %s",
                    self.account_id,
                    e.response.status_code,
                )
            return False
        except httpx.HTTPError as e:
            logger.error("Tumblr [%s]: Connection error: %s", self.account_id, e)
            return False

    async def publish(self, item: QueueItem, content: dict) -> PublishResult:
        """
        Create a post on the configured Tumblr blog.

        content keys:
            post_type (str): "text" | "link" | "photo" | "quote"
                             Default: "text"
            title (str): Post title (for text/link posts)
            body (str): Post body HTML (for text posts)
            caption (str): Fallback for body
            url (str): Link URL (for link posts)
            media_url (str): Image URL (for photo posts)
            quote_text (str): Quote text (for quote posts)
            quote_source (str): Quote source/attribution
            tags (list[str] | str): Comma-separated or list of tags
            state (str): "published" | "draft" | "queue" | "private"
                         Default: "published"
            slug (str): Custom URL slug
            format (str): "html" | "markdown" (default "html")
        """
        if not self._connected or not self._client:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Tumblr not connected",
            )

        post_type = content.get("post_type", "text")
        state = content.get("state", "published")
        tags_raw = content.get("tags", [])
        if isinstance(tags_raw, list):
            tags_str = ",".join(tags_raw)
        else:
            tags_str = str(tags_raw)

        # Build NPF (Neue Post Format) content blocks for text posts
        # or use legacy params for other types
        if post_type == "text":
            payload = self._build_text_post(content, state, tags_str)
        elif post_type == "link":
            payload = self._build_link_post(content, state, tags_str)
        elif post_type == "photo":
            payload = self._build_photo_post(content, state, tags_str)
        elif post_type == "quote":
            payload = self._build_quote_post(content, state, tags_str)
        else:
            payload = self._build_text_post(content, state, tags_str)

        try:
            resp = await self._client.post(
                f"/blog/{self._blog_name}/posts", json=payload
            )
            resp.raise_for_status()
            data = resp.json().get("response", {})

            post_id = str(data.get("id", ""))
            post_url = (
                f"https://{self._blog_name}.tumblr.com/post/{post_id}"
                if post_id
                else ""
            )

            logger.info(
                "Tumblr [%s]: Published %s post -> %s",
                self.account_id,
                post_type,
                post_url,
            )
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=True,
                published_at=datetime.utcnow(),
                post_url=post_url,
                post_id=post_id,
            )

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = e.response.text[:300]
            if status == 429:
                logger.warning("Tumblr [%s]: Rate limited (250/day)", self.account_id)
            elif status == 401:
                logger.error("Tumblr [%s]: Token expired", self.account_id)
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

    def _build_text_post(
        self, content: dict, state: str, tags: str
    ) -> dict[str, Any]:
        """Build NPF text post payload."""
        body = content.get("body", content.get("text", content.get("caption", "")))
        title = content.get("title", "")

        # NPF content blocks
        content_blocks = []
        if title:
            content_blocks.append(
                {"type": "text", "text": title, "subtype": "heading1"}
            )
        if body:
            content_blocks.append({"type": "text", "text": body})

        return {
            "content": content_blocks,
            "state": state,
            "tags": tags,
            "slug": content.get("slug", ""),
        }

    def _build_link_post(
        self, content: dict, state: str, tags: str
    ) -> dict[str, Any]:
        """Build NPF link post payload."""
        url = content.get("url", content.get("media_url", ""))
        title = content.get("title", "")
        description = content.get("body", content.get("caption", ""))

        content_blocks = [
            {
                "type": "link",
                "url": url,
                "title": title,
                "description": description,
            }
        ]
        return {
            "content": content_blocks,
            "state": state,
            "tags": tags,
        }

    def _build_photo_post(
        self, content: dict, state: str, tags: str
    ) -> dict[str, Any]:
        """Build NPF photo post payload."""
        media_url = content.get("media_url", "")
        caption = content.get("caption", content.get("body", ""))

        content_blocks = [
            {
                "type": "image",
                "media": [{"url": media_url}],
            }
        ]
        if caption:
            content_blocks.append({"type": "text", "text": caption})

        return {
            "content": content_blocks,
            "state": state,
            "tags": tags,
        }

    def _build_quote_post(
        self, content: dict, state: str, tags: str
    ) -> dict[str, Any]:
        """Build NPF quote post payload."""
        quote_text = content.get("quote_text", content.get("body", ""))
        source = content.get("quote_source", "")

        content_blocks = [
            {"type": "text", "text": quote_text, "subtype": "indented"}
        ]
        if source:
            content_blocks.append(
                {"type": "text", "text": f"-- {source}"}
            )

        return {
            "content": content_blocks,
            "state": state,
            "tags": tags,
        }

    async def get_metrics(self, post_id: str) -> dict | None:
        """Get post notes count and engagement."""
        if not self._connected or not self._client:
            return None
        try:
            resp = await self._client.get(
                f"/blog/{self._blog_name}/posts",
                params={"id": post_id, "notes_info": True},
            )
            resp.raise_for_status()
            posts = resp.json().get("response", {}).get("posts", [])
            if not posts:
                return None
            post = posts[0]
            return {
                "note_count": post.get("note_count", 0),
                "reblog_count": sum(
                    1
                    for n in post.get("notes", [])
                    if n.get("type") == "reblog"
                ),
                "like_count": sum(
                    1
                    for n in post.get("notes", [])
                    if n.get("type") == "like"
                ),
                "reply_count": sum(
                    1
                    for n in post.get("notes", [])
                    if n.get("type") == "reply"
                ),
            }
        except Exception as e:
            logger.error("Tumblr [%s]: Metrics failed: %s", self.account_id, e)
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False