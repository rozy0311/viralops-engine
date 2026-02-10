"""
ViralOps Engine  Reddit Publisher (REAL Implementation)

Reddit OAuth2 Script App => direct API posting.
Supports: text posts, link posts, image posts to subreddits.

API Docs: https://www.reddit.com/dev/api/
Auth: OAuth2 password grant (script app type)
Rate limit: 60 requests/min (with OAuth)

SETUP:
1. Go to https://www.reddit.com/prefs/apps
2. Create "script" type app
3. Note client_id (under app name) + client_secret
4. Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import httpx

from core.models import PublishResult, QueueItem

logger = logging.getLogger("viralops.publisher.reddit")


class RedditPublisher:
    """Real Reddit OAuth2 API publisher."""

    platform = "reddit"
    AUTH_URL = "https://www.reddit.com/api/v1/access_token"
    API_BASE = "https://oauth.reddit.com"
    USER_AGENT = "ViralOps/1.0 (by /u/{username})"

    def __init__(self, account_id: str = "reddit_main"):
        self.account_id = account_id
        self._client_id: str | None = None
        self._client_secret: str | None = None
        self._username: str | None = None
        self._access_token: str | None = None
        self._user_agent: str = self.USER_AGENT
        self._connected = False
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        """Authenticate via OAuth2 password grant (script app)."""
        prefix = self.account_id.upper().replace("-", "_")
        self._client_id = os.environ.get(
            f"{prefix}_CLIENT_ID", os.environ.get("REDDIT_CLIENT_ID")
        )
        self._client_secret = os.environ.get(
            f"{prefix}_CLIENT_SECRET", os.environ.get("REDDIT_CLIENT_SECRET")
        )
        self._username = os.environ.get(
            f"{prefix}_USERNAME", os.environ.get("REDDIT_USERNAME")
        )
        password = os.environ.get(
            f"{prefix}_PASSWORD", os.environ.get("REDDIT_PASSWORD")
        )

        if not all([self._client_id, self._client_secret, self._username, password]):
            logger.error(
                "Reddit [%s]: Missing credentials "
                "(CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD)",
                self.account_id,
            )
            return False

        self._user_agent = self.USER_AGENT.format(username=self._username)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.AUTH_URL,
                    auth=(self._client_id, self._client_secret),
                    data={
                        "grant_type": "password",
                        "username": self._username,
                        "password": password,
                    },
                    headers={"User-Agent": self._user_agent},
                )
                resp.raise_for_status()
                data = resp.json()

            if "access_token" not in data:
                logger.error("Reddit [%s]: Auth failed: %s", self.account_id, data)
                return False

            self._access_token = data["access_token"]
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "User-Agent": self._user_agent,
                },
                timeout=30.0,
            )
            self._connected = True
            logger.info(
                "Reddit [%s]: Connected as u/%s", self.account_id, self._username
            )
            return True

        except httpx.HTTPError as e:
            logger.error("Reddit [%s]: Connection failed: %s", self.account_id, e)
            return False

    async def publish(self, item: QueueItem, content: dict) -> PublishResult:
        """
        Submit a post to a subreddit.

        content keys:
            subreddit (str): Target subreddit name (without r/)
            title (str): Post title (required, max 300 chars)
            kind (str): "self" | "link" | "image" -- default "self"
            text (str): Body text for self posts (markdown)
            url (str): URL for link posts
            caption (str): Fallback for text/title
            flair_id (str): Optional subreddit flair ID
            nsfw (bool): Mark as NSFW (default False)
            spoiler (bool): Mark as spoiler (default False)
        """
        if not self._connected or not self._client:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Reddit not connected",
            )

        subreddit = content.get("subreddit", "")
        if not subreddit:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="No subreddit specified",
            )

        kind = content.get("kind", "self")
        title = content.get("title", content.get("caption", ""))[:300]

        post_data: dict[str, Any] = {
            "sr": subreddit,
            "kind": kind,
            "title": title,
            "resubmit": True,
            "nsfw": content.get("nsfw", False),
            "spoiler": content.get("spoiler", False),
        }

        if kind == "self":
            post_data["text"] = content.get("text", content.get("caption", ""))
        elif kind == "link":
            post_data["url"] = content.get("url", content.get("media_url", ""))
        elif kind == "image":
            post_data["url"] = content.get("media_url", "")

        if content.get("flair_id"):
            post_data["flair_id"] = content["flair_id"]

        try:
            resp = await self._client.post("/api/submit", data=post_data)
            resp.raise_for_status()
            result = resp.json()

            # Reddit: {"json": {"errors": [], "data": {"url", "id", "name": "t3_..."}}}
            errors = result.get("json", {}).get("errors", [])
            if errors:
                error_msg = "; ".join([str(e) for e in errors])
                logger.warning(
                    "Reddit [%s]: Post errors: %s", self.account_id, error_msg
                )
                return PublishResult(
                    queue_item_id=item.id,
                    platform=self.platform,
                    success=False,
                    error=error_msg,
                )

            post_result = result.get("json", {}).get("data", {})
            post_url = post_result.get("url", "")
            post_id = post_result.get("name", "")  # e.g. "t3_abc123"

            logger.info(
                "Reddit [%s]: Published to r/%s -> %s",
                self.account_id,
                subreddit,
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
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", "60")
                logger.warning(
                    "Reddit [%s]: Rate limited, retry after %ss",
                    self.account_id,
                    retry_after,
                )
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=str(e),
            )

    async def get_metrics(self, post_id: str) -> dict | None:
        """Fetch post metrics (score, upvote_ratio, comments)."""
        if not self._connected or not self._client:
            return None
        try:
            fullname = post_id if post_id.startswith("t3_") else f"t3_{post_id}"
            resp = await self._client.get("/api/info", params={"id": fullname})
            resp.raise_for_status()
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            if not children:
                return None
            post = children[0].get("data", {})
            return {
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
                "num_comments": post.get("num_comments", 0),
                "ups": post.get("ups", 0),
                "downs": post.get("downs", 0),
                "created_utc": post.get("created_utc"),
            }
        except Exception as e:
            logger.error("Reddit [%s]: Metrics failed: %s", self.account_id, e)
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False