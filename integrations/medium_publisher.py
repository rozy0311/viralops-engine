"""
ViralOps Engine -- Medium Publisher (REAL Implementation)

Medium REST API v1 -- direct article publishing.
Supports: HTML articles with tags, canonical URL, publish status.

API Docs: https://github.com/Medium/medium-api-docs
Auth: Integration token (Bearer)
Rate limit: Undocumented, ~100 posts/day recommended max

SETUP:
1. Go to https://medium.com/me/settings/security
2. Generate an Integration Token
3. Set env var: MEDIUM_TOKEN
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from core.models import PublishResult, QueueItem

logger = logging.getLogger("viralops.publisher.medium")


class MediumPublisher:
    """Real Medium API v1 publisher."""

    platform = "medium"
    API_BASE = "https://api.medium.com/v1"

    def __init__(self, account_id: str = "medium_main"):
        self.account_id = account_id
        self._token: str | None = None
        self._user_id: str | None = None
        self._username: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        """Authenticate with Medium integration token and fetch user ID."""
        prefix = self.account_id.upper().replace("-", "_")
        self._token = os.environ.get(
            f"{prefix}_TOKEN", os.environ.get("MEDIUM_TOKEN")
        )

        if not self._token:
            logger.error(
                "Medium [%s]: Missing MEDIUM_TOKEN env var", self.account_id
            )
            return False

        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Charset": "utf-8",
            },
            timeout=30.0,
        )

        try:
            # GET /v1/me to obtain user ID
            resp = await self._client.get("/me")
            resp.raise_for_status()
            data = resp.json().get("data", {})

            self._user_id = data.get("id")
            self._username = data.get("username")

            if not self._user_id:
                logger.error(
                    "Medium [%s]: Could not retrieve user ID", self.account_id
                )
                return False

            self._connected = True
            logger.info(
                "Medium [%s]: Connected as @%s (id=%s)",
                self.account_id,
                self._username,
                self._user_id,
            )
            return True

        except httpx.HTTPError as e:
            logger.error("Medium [%s]: Connection failed: %s", self.account_id, e)
            return False

    async def publish(self, item: QueueItem, content: dict) -> PublishResult:
        """
        Create a Medium post under the authenticated user.

        content keys:
            title (str): Article title (required)
            body (str): Article body -- HTML or Markdown
            content_format (str): "html" or "markdown" (default "html")
            tags (list[str]): Up to 5 tags
            canonical_url (str): Original article URL (for SEO)
            publish_status (str): "public" | "draft" | "unlisted" (default "draft")
            caption (str): Fallback for title/body
            notify_followers (bool): Whether to send email notifications (default False)
        """
        if not self._connected or not self._client or not self._user_id:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Medium not connected",
            )

        title = content.get("title", content.get("caption", "Untitled"))
        body = content.get("body", content.get("text", content.get("caption", "")))
        content_format = content.get("content_format", "html")
        tags = content.get("tags", [])[:5]  # Medium max 5 tags
        publish_status = content.get("publish_status", "draft")

        payload: dict[str, Any] = {
            "title": title,
            "contentFormat": content_format,
            "content": body,
            "tags": tags,
            "publishStatus": publish_status,
        }

        if content.get("canonical_url"):
            payload["canonicalUrl"] = content["canonical_url"]

        if content.get("notify_followers"):
            payload["notifyFollowers"] = True

        try:
            resp = await self._client.post(
                f"/users/{self._user_id}/posts", json=payload
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})

            post_url = data.get("url", "")
            post_id = data.get("id", "")

            logger.info(
                "Medium [%s]: Published '%s' -> %s (status=%s)",
                self.account_id,
                title[:50],
                post_url,
                publish_status,
            )
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=True,
                published_at=datetime.now(timezone.utc),
                post_url=post_url,
                post_id=post_id,
            )

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = e.response.text[:300]
            if status == 429:
                logger.warning("Medium [%s]: Rate limited", self.account_id)
            elif status == 401:
                logger.error("Medium [%s]: Token expired or invalid", self.account_id)
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

    async def get_user_publications(self) -> list[dict]:
        """List publications the authenticated user can write to."""
        if not self._connected or not self._client or not self._user_id:
            return []
        try:
            resp = await self._client.get(
                f"/users/{self._user_id}/publications"
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            logger.error(
                "Medium [%s]: Failed to list publications: %s",
                self.account_id, e,
            )
            return []

    async def publish_to_publication(
        self, publication_id: str, item: QueueItem, content: dict
    ) -> PublishResult:
        """
        Publish under a Medium Publication instead of user profile.
        Same content dict as publish().
        """
        if not self._connected or not self._client:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Medium not connected",
            )

        title = content.get("title", content.get("caption", "Untitled"))
        body = content.get("body", content.get("text", content.get("caption", "")))

        payload: dict[str, Any] = {
            "title": title,
            "contentFormat": content.get("content_format", "html"),
            "content": body,
            "tags": content.get("tags", [])[:5],
            "publishStatus": content.get("publish_status", "draft"),
        }
        if content.get("canonical_url"):
            payload["canonicalUrl"] = content["canonical_url"]

        try:
            resp = await self._client.post(
                f"/publications/{publication_id}/posts", json=payload
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=True,
                published_at=datetime.now(timezone.utc),
                post_url=data.get("url", ""),
                post_id=data.get("id", ""),
            )
        except Exception as e:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=str(e),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False