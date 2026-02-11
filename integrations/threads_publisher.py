"""
ViralOps Engine — Threads Publisher (REAL Implementation)

Meta Threads API (v1.0) — text + image + video posting.
Threads uses the same Graph API infrastructure as Instagram.

API Docs: https://developers.facebook.com/docs/threads
Auth: Instagram user access token (same as IG Graph API)
Rate limit: 250 posts/24h, 1000 replies/24h

SETUP:
1. You need a Meta App with threads_manage permissions
2. Get user access token via IG Login flow
3. Get your Threads user_id (same as IG user_id)
4. Set env vars: THREADS_MAIN_ACCESS_TOKEN, THREADS_MAIN_USER_ID
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger("viralops.publisher.threads")


class ThreadsPublisher:
    """Real Meta Threads API publisher."""

    platform = "threads"
    API_BASE = "https://graph.threads.net/v1.0"
    MAX_CAPTION = 500  # Threads caption limit

    def __init__(self, account_id: str = "threads_main"):
        self.account_id = account_id
        self._access_token: str | None = None
        self._user_id: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        """Load credentials and verify access."""
        prefix = self.account_id.upper().replace("-", "_")
        self._access_token = os.environ.get(
            f"{prefix}_ACCESS_TOKEN",
            os.environ.get("THREADS_MAIN_ACCESS_TOKEN"),
        )
        self._user_id = os.environ.get(
            f"{prefix}_USER_ID",
            os.environ.get("THREADS_MAIN_USER_ID"),
        )

        if not self._access_token or not self._user_id:
            logger.error(
                "Threads [%s]: Missing ACCESS_TOKEN or USER_ID env var",
                self.account_id,
            )
            return False

        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            timeout=30.0,
            headers={"Authorization": f"Bearer {self._access_token}"},
        )

        # Verify credentials
        try:
            resp = await self._client.get(
                f"/{self._user_id}",
                params={"fields": "id,username,threads_profile_picture_url"},
            )
            if resp.status_code == 200:
                data = resp.json()
                self._connected = True
                logger.info(
                    "Threads [%s]: Connected as @%s",
                    self.account_id,
                    data.get("username", "?"),
                )
                return True
            else:
                logger.error(
                    "Threads [%s]: Auth failed: %s", self.account_id, resp.text
                )
                return False
        except Exception as e:
            logger.error("Threads [%s]: Connect error: %s", self.account_id, e)
            return False

    async def publish(self, content: dict) -> dict:
        """
        Publish to Threads.

        Two-step process (like Instagram):
          1. Create media container
          2. Publish the container

        content keys:
          caption (str): Post text (max 500 chars)
          media_url (str, optional): Image or video URL
          media_type (str): "TEXT", "IMAGE", "VIDEO"
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {
                "success": False,
                "error": "Threads not connected",
                "platform": self.platform,
            }

        caption = content.get("caption", "")[:self.MAX_CAPTION]
        media_url = content.get("media_url", "")
        media_type = content.get("media_type", "TEXT").upper()

        # Determine post type
        if media_type == "VIDEO" and media_url:
            post_type = "VIDEO"
        elif media_type == "IMAGE" and media_url:
            post_type = "IMAGE"
        else:
            post_type = "TEXT"

        try:
            # Step 1: Create media container
            container_params = {
                "media_type": post_type,
                "text": caption,
            }
            if post_type == "IMAGE":
                container_params["image_url"] = media_url
            elif post_type == "VIDEO":
                container_params["video_url"] = media_url

            resp = await self._client.post(
                f"/{self._user_id}/threads",
                params=container_params,
            )

            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Container creation failed: {resp.text}",
                    "platform": self.platform,
                }

            container_id = resp.json().get("id")
            if not container_id:
                return {
                    "success": False,
                    "error": "No container ID returned",
                    "platform": self.platform,
                }

            # For VIDEO: poll until container is ready
            if post_type == "VIDEO":
                import asyncio
                for _ in range(30):  # Max 30 attempts (5 min)
                    status_resp = await self._client.get(
                        f"/{container_id}",
                        params={"fields": "status"},
                    )
                    status = status_resp.json().get("status", "")
                    if status == "FINISHED":
                        break
                    elif status == "ERROR":
                        return {
                            "success": False,
                            "error": "Video processing failed",
                            "platform": self.platform,
                        }
                    await asyncio.sleep(10)

            # Step 2: Publish the container
            pub_resp = await self._client.post(
                f"/{self._user_id}/threads_publish",
                params={"creation_id": container_id},
            )

            if pub_resp.status_code == 200:
                post_id = pub_resp.json().get("id", "")
                logger.info(
                    "Threads [%s]: Published %s post_id=%s",
                    self.account_id, post_type, post_id,
                )
                return {
                    "success": True,
                    "post_id": post_id,
                    "post_url": f"https://www.threads.net/post/{post_id}",
                    "platform": self.platform,
                }
            else:
                return {
                    "success": False,
                    "error": f"Publish failed: {pub_resp.text}",
                    "platform": self.platform,
                }

        except Exception as e:
            logger.error("Threads [%s]: Publish error: %s", self.account_id, e)
            return {
                "success": False,
                "error": str(e),
                "platform": self.platform,
            }

    async def reply(self, parent_post_id: str, text: str) -> dict:
        """Reply to a Threads post (for thread chains)."""
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        try:
            # Create reply container
            resp = await self._client.post(
                f"/{self._user_id}/threads",
                params={
                    "media_type": "TEXT",
                    "text": text[:self.MAX_CAPTION],
                    "reply_to_id": parent_post_id,
                },
            )
            container_id = resp.json().get("id")
            if not container_id:
                return {"success": False, "error": "Reply container failed"}

            # Publish reply
            pub_resp = await self._client.post(
                f"/{self._user_id}/threads_publish",
                params={"creation_id": container_id},
            )
            if pub_resp.status_code == 200:
                reply_id = pub_resp.json().get("id", "")
                return {"success": True, "post_id": reply_id}
            return {"success": False, "error": pub_resp.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_metrics(self, post_id: str) -> dict | None:
        """Fetch post insights (views, likes, replies, reposts, quotes)."""
        if not self._connected or not self._client:
            return None
        try:
            resp = await self._client.get(
                f"/{post_id}/insights",
                params={
                    "metric": "views,likes,replies,reposts,quotes",
                },
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                metrics = {}
                for m in data:
                    metrics[m["name"]] = m.get("values", [{}])[0].get("value", 0)
                return metrics
            return None
        except Exception as e:
            logger.error("Threads [%s]: Metrics failed: %s", self.account_id, e)
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
