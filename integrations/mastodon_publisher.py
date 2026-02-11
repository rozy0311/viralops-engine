"""
ViralOps Engine — Mastodon Publisher (REAL Implementation)

Mastodon REST API — text + image + video posting.
Supports any Mastodon-compatible instance (mastodon.social, fosstodon.org, etc.)

API Docs: https://docs.joinmastodon.org/api/
Auth: OAuth2 Bearer token (create app → authorize → get token)
Rate limit: 300 requests/5min (varies by instance)

SETUP:
1. Go to your instance Preferences → Development → New Application
2. Set scopes: read write push
3. Copy your access token
4. Set env vars: MASTODON_MAIN_ACCESS_TOKEN, MASTODON_MAIN_INSTANCE_URL
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger("viralops.publisher.mastodon")


class MastodonPublisher:
    """Real Mastodon REST API publisher."""

    platform = "mastodon"
    MAX_CAPTION = 500  # Default Mastodon limit (varies by instance, some allow 5000)

    def __init__(self, account_id: str = "mastodon_main"):
        self.account_id = account_id
        self._access_token: str | None = None
        self._instance_url: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None
        self._max_chars: int = self.MAX_CAPTION

    async def connect(self) -> bool:
        """Load credentials and verify access."""
        prefix = self.account_id.upper().replace("-", "_")
        self._access_token = os.environ.get(
            f"{prefix}_ACCESS_TOKEN",
            os.environ.get("MASTODON_MAIN_ACCESS_TOKEN"),
        )
        self._instance_url = os.environ.get(
            f"{prefix}_INSTANCE_URL",
            os.environ.get("MASTODON_MAIN_INSTANCE_URL", "https://mastodon.social"),
        )

        if not self._access_token:
            logger.error(
                "Mastodon [%s]: Missing ACCESS_TOKEN env var", self.account_id
            )
            return False

        # Normalize instance URL
        self._instance_url = self._instance_url.rstrip("/")

        self._client = httpx.AsyncClient(
            base_url=f"{self._instance_url}/api",
            timeout=30.0,
            headers={"Authorization": f"Bearer {self._access_token}"},
        )

        # Verify credentials + get instance char limit
        try:
            resp = await self._client.get("/v1/accounts/verify_credentials")
            if resp.status_code == 200:
                data = resp.json()
                self._connected = True
                logger.info(
                    "Mastodon [%s]: Connected as @%s@%s",
                    self.account_id,
                    data.get("username", "?"),
                    self._instance_url.replace("https://", ""),
                )

                # Fetch instance char limit
                try:
                    inst_resp = await self._client.get("/v2/instance")
                    if inst_resp.status_code == 200:
                        inst_data = inst_resp.json()
                        config = inst_data.get("configuration", {})
                        statuses_config = config.get("statuses", {})
                        self._max_chars = statuses_config.get(
                            "max_characters", self.MAX_CAPTION
                        )
                        logger.info(
                            "Mastodon [%s]: Instance max_chars=%d",
                            self.account_id,
                            self._max_chars,
                        )
                except Exception:
                    pass  # Keep default 500

                return True
            else:
                logger.error(
                    "Mastodon [%s]: Auth failed: %s", self.account_id, resp.text
                )
                return False
        except Exception as e:
            logger.error("Mastodon [%s]: Connect error: %s", self.account_id, e)
            return False

    async def _upload_media(self, media_url: str, description: str = "") -> str | None:
        """Download media from URL and upload to Mastodon instance."""
        try:
            # Download from URL
            async with httpx.AsyncClient(timeout=60) as dl_client:
                dl_resp = await dl_client.get(media_url, follow_redirects=True)
                dl_resp.raise_for_status()

            content_type = dl_resp.headers.get("content-type", "image/jpeg")
            filename = "media.jpg"
            if "png" in content_type:
                filename = "media.png"
            elif "video" in content_type:
                filename = "media.mp4"
            elif "gif" in content_type:
                filename = "media.gif"

            # Upload to Mastodon
            resp = await self._client.post(
                "/v2/media",
                files={"file": (filename, dl_resp.content, content_type)},
                data={"description": description[:1500]} if description else {},
            )

            if resp.status_code in (200, 202):
                media_id = resp.json().get("id")
                logger.info(
                    "Mastodon [%s]: Media uploaded id=%s", self.account_id, media_id
                )
                return media_id
            else:
                logger.error(
                    "Mastodon [%s]: Media upload failed: %s",
                    self.account_id,
                    resp.text,
                )
                return None

        except Exception as e:
            logger.error(
                "Mastodon [%s]: Media upload error: %s", self.account_id, e
            )
            return None

    async def publish(self, content: dict) -> dict:
        """
        Publish a toot (status) to Mastodon.

        content keys:
          caption (str): Status text
          media_url (str, optional): Image/video URL
          media_type (str): "text", "image", "video"
          visibility (str): "public", "unlisted", "private", "direct"
          spoiler_text (str, optional): Content warning
          language (str, optional): ISO 639-1 language code
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {
                "success": False,
                "error": "Mastodon not connected",
                "platform": self.platform,
            }

        caption = content.get("caption", "")[:self._max_chars]
        media_url = content.get("media_url", "")
        visibility = content.get("visibility", "public")
        spoiler_text = content.get("spoiler_text", "")
        language = content.get("language", "")

        # Build status payload
        payload: dict[str, Any] = {
            "status": caption,
            "visibility": visibility,
        }
        if spoiler_text:
            payload["spoiler_text"] = spoiler_text
        if language:
            payload["language"] = language

        # Upload media if provided
        if media_url:
            media_id = await self._upload_media(
                media_url, description=caption[:200]
            )
            if media_id:
                payload["media_ids"] = [media_id]

        try:
            resp = await self._client.post("/v1/statuses", json=payload)

            if resp.status_code == 200:
                data = resp.json()
                post_id = data.get("id", "")
                post_url = data.get("url", "")

                logger.info(
                    "Mastodon [%s]: Published id=%s url=%s",
                    self.account_id,
                    post_id,
                    post_url,
                )
                return {
                    "success": True,
                    "post_id": post_id,
                    "post_url": post_url,
                    "platform": self.platform,
                }
            elif resp.status_code == 422:
                return {
                    "success": False,
                    "error": f"Validation error: {resp.text}",
                    "platform": self.platform,
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                    "platform": self.platform,
                }

        except Exception as e:
            logger.error("Mastodon [%s]: Publish error: %s", self.account_id, e)
            return {
                "success": False,
                "error": str(e),
                "platform": self.platform,
            }

    async def reply(self, parent_post_id: str, text: str) -> dict:
        """Reply to an existing toot."""
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        try:
            resp = await self._client.post(
                "/v1/statuses",
                json={
                    "status": text[:self._max_chars],
                    "in_reply_to_id": parent_post_id,
                    "visibility": "public",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return {"success": True, "post_id": data.get("id", "")}
            return {"success": False, "error": resp.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_metrics(self, post_id: str) -> dict | None:
        """Fetch toot engagement (favourites, reblogs, replies)."""
        if not self._connected or not self._client:
            return None
        try:
            resp = await self._client.get(f"/v1/statuses/{post_id}")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "favourites": data.get("favourites_count", 0),
                    "reblogs": data.get("reblogs_count", 0),
                    "replies": data.get("replies_count", 0),
                    "url": data.get("url", ""),
                }
            return None
        except Exception as e:
            logger.error("Mastodon [%s]: Metrics failed: %s", self.account_id, e)
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
