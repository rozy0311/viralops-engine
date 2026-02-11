"""
ViralOps Engine — Bluesky Publisher (REAL Implementation)

AT Protocol (atproto) — text + image posting via Bluesky PDS.
Bluesky uses the decentralized AT Protocol for authentication and posting.

API Docs: https://docs.bsky.app/docs/api/
Auth: App password (handle + app-password) → creates session JWT
Rate limit: 1666 posts/day (hourly rolling), 300 actions/5min

SETUP:
1. Go to Settings → App Passwords on bsky.app
2. Generate a new app password
3. Set env vars: BLUESKY_MAIN_HANDLE, BLUESKY_MAIN_APP_PASSWORD
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("viralops.publisher.bluesky")


class BlueskyPublisher:
    """Real Bluesky AT Protocol publisher."""

    platform = "bluesky"
    PDS_URL = "https://bsky.social"
    MAX_CAPTION = 300  # Bluesky character limit (grapheme count)

    def __init__(self, account_id: str = "bluesky_main"):
        self.account_id = account_id
        self._handle: str | None = None
        self._app_password: str | None = None
        self._did: str | None = None
        self._access_jwt: str | None = None
        self._refresh_jwt: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        """Authenticate via createSession (handle + app password)."""
        prefix = self.account_id.upper().replace("-", "_")
        self._handle = os.environ.get(
            f"{prefix}_HANDLE",
            os.environ.get("BLUESKY_MAIN_HANDLE"),
        )
        self._app_password = os.environ.get(
            f"{prefix}_APP_PASSWORD",
            os.environ.get("BLUESKY_MAIN_APP_PASSWORD"),
        )

        if not self._handle or not self._app_password:
            logger.error(
                "Bluesky [%s]: Missing HANDLE or APP_PASSWORD env var",
                self.account_id,
            )
            return False

        self._client = httpx.AsyncClient(
            base_url=f"{self.PDS_URL}/xrpc",
            timeout=30.0,
        )

        try:
            resp = await self._client.post(
                "/com.atproto.server.createSession",
                json={
                    "identifier": self._handle,
                    "password": self._app_password,
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                self._did = data.get("did")
                self._access_jwt = data.get("accessJwt")
                self._refresh_jwt = data.get("refreshJwt")
                self._connected = True
                logger.info(
                    "Bluesky [%s]: Connected as %s (did=%s)",
                    self.account_id,
                    data.get("handle"),
                    self._did,
                )
                return True
            else:
                logger.error(
                    "Bluesky [%s]: Auth failed: %s", self.account_id, resp.text
                )
                return False
        except Exception as e:
            logger.error("Bluesky [%s]: Connect error: %s", self.account_id, e)
            return False

    async def _refresh_session(self) -> bool:
        """Refresh access token using refresh JWT."""
        if not self._client or not self._refresh_jwt:
            return False
        try:
            resp = await self._client.post(
                "/com.atproto.server.refreshSession",
                headers={"Authorization": f"Bearer {self._refresh_jwt}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                self._access_jwt = data.get("accessJwt")
                self._refresh_jwt = data.get("refreshJwt")
                return True
        except Exception as e:
            logger.error("Bluesky [%s]: Refresh failed: %s", self.account_id, e)
        return False

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_jwt}"}

    def _detect_facets(self, text: str) -> list[dict]:
        """
        Detect links, mentions, and hashtags in text for rich text facets.
        AT Protocol requires explicit byte-range annotations for rich text.
        """
        facets = []
        text_bytes = text.encode("utf-8")

        # Detect URLs
        url_pattern = re.compile(
            r'https?://[^\s\)\]\}>"\']+', re.IGNORECASE
        )
        for match in url_pattern.finditer(text):
            start = len(text[: match.start()].encode("utf-8"))
            end = len(text[: match.end()].encode("utf-8"))
            facets.append({
                "index": {"byteStart": start, "byteEnd": end},
                "features": [{"$type": "app.bsky.richtext.facet#link", "uri": match.group()}],
            })

        # Detect mentions (@handle.bsky.social)
        mention_pattern = re.compile(r'@([a-zA-Z0-9._-]+\.[a-zA-Z]+)')
        for match in mention_pattern.finditer(text):
            start = len(text[: match.start()].encode("utf-8"))
            end = len(text[: match.end()].encode("utf-8"))
            facets.append({
                "index": {"byteStart": start, "byteEnd": end},
                "features": [{"$type": "app.bsky.richtext.facet#mention", "did": match.group(1)}],
            })

        # Detect hashtags (#example)
        tag_pattern = re.compile(r'#([a-zA-Z0-9_]+)')
        for match in tag_pattern.finditer(text):
            start = len(text[: match.start()].encode("utf-8"))
            end = len(text[: match.end()].encode("utf-8"))
            facets.append({
                "index": {"byteStart": start, "byteEnd": end},
                "features": [{"$type": "app.bsky.richtext.facet#tag", "tag": match.group(1)}],
            })

        return facets

    async def _upload_image(self, image_url: str) -> dict | None:
        """Download image from URL and upload as blob to Bluesky PDS."""
        try:
            async with httpx.AsyncClient(timeout=60) as dl_client:
                img_resp = await dl_client.get(image_url, follow_redirects=True)
                img_resp.raise_for_status()

            content_type = img_resp.headers.get("content-type", "image/jpeg")
            resp = await self._client.post(
                "/com.atproto.repo.uploadBlob",
                content=img_resp.content,
                headers={
                    **self._auth_headers(),
                    "Content-Type": content_type,
                },
            )

            if resp.status_code == 200:
                return resp.json().get("blob")
            else:
                logger.error("Bluesky [%s]: Blob upload failed: %s", self.account_id, resp.text)
                return None
        except Exception as e:
            logger.error("Bluesky [%s]: Image upload error: %s", self.account_id, e)
            return None

    async def publish(self, content: dict) -> dict:
        """
        Publish to Bluesky.

        Uses com.atproto.repo.createRecord with app.bsky.feed.post lexicon.

        content keys:
          caption (str): Post text (max 300 graphemes)
          media_url (str, optional): Image URL to embed
          media_type (str): "text", "image"
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {
                "success": False,
                "error": "Bluesky not connected",
                "platform": self.platform,
            }

        caption = content.get("caption", "")[:self.MAX_CAPTION]
        media_url = content.get("media_url", "")
        now = datetime.now(timezone.utc).isoformat()

        # Build post record
        record = {
            "$type": "app.bsky.feed.post",
            "text": caption,
            "createdAt": now,
        }

        # Add rich text facets (links, mentions, hashtags)
        facets = self._detect_facets(caption)
        if facets:
            record["facets"] = facets

        # Embed image if provided
        if media_url and content.get("media_type") in ("image", "IMAGE"):
            blob = await self._upload_image(media_url)
            if blob:
                record["embed"] = {
                    "$type": "app.bsky.embed.images",
                    "images": [{"alt": caption[:100], "image": blob}],
                }

        try:
            resp = await self._client.post(
                "/com.atproto.repo.createRecord",
                json={
                    "repo": self._did,
                    "collection": "app.bsky.feed.post",
                    "record": record,
                },
                headers=self._auth_headers(),
            )

            if resp.status_code == 200:
                data = resp.json()
                uri = data.get("uri", "")
                cid = data.get("cid", "")
                # Extract rkey for web URL
                rkey = uri.split("/")[-1] if "/" in uri else ""
                handle = self._handle or ""
                post_url = f"https://bsky.app/profile/{handle}/post/{rkey}"

                logger.info(
                    "Bluesky [%s]: Published uri=%s cid=%s",
                    self.account_id, uri, cid,
                )
                return {
                    "success": True,
                    "post_id": uri,
                    "post_url": post_url,
                    "cid": cid,
                    "platform": self.platform,
                }

            # Handle expired token
            if resp.status_code == 401:
                if await self._refresh_session():
                    return await self.publish(content)  # Retry once

            return {
                "success": False,
                "error": f"Create record failed: {resp.text}",
                "platform": self.platform,
            }

        except Exception as e:
            logger.error("Bluesky [%s]: Publish error: %s", self.account_id, e)
            return {
                "success": False,
                "error": str(e),
                "platform": self.platform,
            }

    async def get_metrics(self, post_uri: str) -> dict | None:
        """Fetch post engagement (likes, reposts, replies)."""
        if not self._connected or not self._client:
            return None
        try:
            resp = await self._client.get(
                "/app.bsky.feed.getPostThread",
                params={"uri": post_uri, "depth": 0},
                headers=self._auth_headers(),
            )
            if resp.status_code == 200:
                post = resp.json().get("thread", {}).get("post", {})
                return {
                    "likes": post.get("likeCount", 0),
                    "reposts": post.get("repostCount", 0),
                    "replies": post.get("replyCount", 0),
                    "uri": post_uri,
                }
            return None
        except Exception as e:
            logger.error("Bluesky [%s]: Metrics failed: %s", self.account_id, e)
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
