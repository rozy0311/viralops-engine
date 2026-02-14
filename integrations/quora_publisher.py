"""
ViralOps Engine — Quora Publisher (REAL Implementation)

Quora has NO official public API for posting content.
This publisher uses a hybrid approach:
  1. Session-cookie auth (from browser dev-tools)
  2. GraphQL internal API for posting answers/spaces content
  3. Fallback: Webhook notification for manual posting via Quora app

IMPORTANT:
  - Quora may change internal APIs without notice
  - Session cookies expire (typically 30 days)
  - This is a best-effort integration, NOT guaranteed by Quora TOS
  - For production, consider Quora's official ads/partner API if available

SETUP:
1. Login to Quora in browser
2. Open DevTools → Application → Cookies
3. Copy 'm-b' cookie value → QUORA_MAIN_SESSION_COOKIE
4. Copy 'formkey' from page source → QUORA_MAIN_FORMKEY
5. (Optional) Set QUORA_MAIN_WEBHOOK for fallback notifications
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("viralops.publisher.quora")


class QuoraPublisher:
    """Quora publisher using session-based auth + GraphQL internal API."""

    platform = "quora"
    MAX_CAPTION = 50_000  # Quora allows long-form answers
    API_BASE = "https://www.quora.com"

    def __init__(self, account_id: str = "quora_main"):
        self.account_id = account_id
        self._session_cookie: str | None = None
        self._formkey: str | None = None
        self._webhook_url: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None
        self._username: str | None = None

    async def connect(self) -> bool:
        """Load session credentials and verify login."""
        prefix = self.account_id.upper().replace("-", "_")
        self._session_cookie = os.environ.get(
            f"{prefix}_SESSION_COOKIE",
            os.environ.get("QUORA_MAIN_SESSION_COOKIE"),
        )
        self._formkey = os.environ.get(
            f"{prefix}_FORMKEY",
            os.environ.get("QUORA_MAIN_FORMKEY"),
        )
        self._webhook_url = os.environ.get(
            f"{prefix}_WEBHOOK",
            os.environ.get("QUORA_MAIN_WEBHOOK"),
        )

        if not self._session_cookie or not self._formkey:
            logger.error(
                "Quora [%s]: Missing SESSION_COOKIE or FORMKEY env vars",
                self.account_id,
            )
            # If webhook is available, we can still do fallback
            if self._webhook_url:
                logger.info(
                    "Quora [%s]: Webhook fallback available", self.account_id
                )
                self._connected = True  # partial — webhook only
                return True
            return False

        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            timeout=30.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Quora-Formkey": self._formkey,
                "Content-Type": "application/json",
            },
            cookies={"m-b": self._session_cookie},
        )

        # Verify session by fetching profile
        try:
            resp = await self._client.post(
                "/graphql/gql_para_POST",
                json={
                    "queryName": "ViewerQuery",
                    "variables": {},
                    "query": (
                        "query ViewerQuery { viewer { user { uid firstName "
                        "profileUrl } } }"
                    ),
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                user_data = (
                    data.get("data", {})
                    .get("viewer", {})
                    .get("user", {})
                )
                if user_data:
                    self._username = user_data.get("firstName", "User")
                    self._connected = True
                    logger.info(
                        "Quora [%s]: Connected as %s",
                        self.account_id,
                        self._username,
                    )
                    return True
                else:
                    logger.error(
                        "Quora [%s]: Session expired or invalid", self.account_id
                    )
                    return False
            else:
                logger.error(
                    "Quora [%s]: Auth check failed: %s",
                    self.account_id,
                    resp.status_code,
                )
                return False
        except Exception as e:
            logger.error("Quora [%s]: Connect error: %s", self.account_id, e)
            return False

    async def publish(self, content: dict) -> dict:
        """
        Publish content to Quora.

        content keys:
          caption (str): The answer or post text (Markdown supported)
          question_url (str, optional): URL of question to answer
          space_id (str, optional): Space to post in
          title (str, optional): Title for space post
          media_url (str, optional): Image URL to include
          mode (str): "answer" | "space_post" | "webhook_fallback"
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {
                "success": False,
                "error": "Quora not connected",
                "platform": self.platform,
            }

        mode = content.get("mode", "answer")
        caption = content.get("caption", "")[:self.MAX_CAPTION]
        question_url = content.get("question_url", "")
        space_id = content.get("space_id", "")
        title = content.get("title", "")

        # Try webhook fallback if no GraphQL client
        if not self._client and self._webhook_url:
            return await self._webhook_fallback(content)

        if mode == "answer" and question_url:
            return await self._post_answer(question_url, caption)
        elif mode == "space_post" and space_id:
            return await self._post_to_space(space_id, title, caption)
        elif self._webhook_url:
            return await self._webhook_fallback(content)
        else:
            return {
                "success": False,
                "error": (
                    "Must provide question_url (mode=answer) or "
                    "space_id (mode=space_post) or webhook URL"
                ),
                "platform": self.platform,
            }

    async def _post_answer(self, question_url: str, text: str) -> dict:
        """Post an answer to a specific Quora question via GraphQL."""
        # Extract question ID from URL
        qid = self._extract_question_id(question_url)
        if not qid:
            return {
                "success": False,
                "error": f"Cannot extract question ID from: {question_url}",
                "platform": self.platform,
            }

        try:
            # Convert markdown to Quora's JSON format
            content_json = self._markdown_to_quora_json(text)

            resp = await self._client.post(
                "/graphql/gql_para_POST",
                json={
                    "queryName": "CreateAnswerMutation",
                    "variables": {
                        "questionQid": qid,
                        "content": content_json,
                        "attachAnonymously": False,
                    },
                    "query": (
                        "mutation CreateAnswerMutation("
                        "$questionQid: String!, "
                        "$content: String!, "
                        "$attachAnonymously: Boolean"
                        ") { answerCreate("
                        "questionQid: $questionQid, "
                        "content: $content, "
                        "attachAnonymously: $attachAnonymously"
                        ") { answer { aid url viewerHasUpvoted } } }"
                    ),
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                answer_data = (
                    data.get("data", {})
                    .get("answerCreate", {})
                    .get("answer", {})
                )
                if answer_data:
                    answer_id = answer_data.get("aid", "")
                    answer_url = answer_data.get("url", question_url)
                    logger.info(
                        "Quora [%s]: Answer posted aid=%s",
                        self.account_id,
                        answer_id,
                    )
                    return {
                        "success": True,
                        "post_id": answer_id,
                        "post_url": answer_url,
                        "mode": "answer",
                        "platform": self.platform,
                    }
                else:
                    errors = data.get("errors", [])
                    error_msg = errors[0].get("message", "Unknown") if errors else "No answer data returned"
                    return {
                        "success": False,
                        "error": error_msg,
                        "platform": self.platform,
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "platform": self.platform,
                }

        except Exception as e:
            logger.error(
                "Quora [%s]: Post answer error: %s", self.account_id, e
            )
            return {"success": False, "error": str(e), "platform": self.platform}

    async def _post_to_space(self, space_id: str, title: str, text: str) -> dict:
        """Post content to a Quora Space."""
        try:
            content_json = self._markdown_to_quora_json(text)

            resp = await self._client.post(
                "/graphql/gql_para_POST",
                json={
                    "queryName": "SpacePostCreateMutation",
                    "variables": {
                        "spaceId": space_id,
                        "title": title[:250],
                        "content": content_json,
                    },
                    "query": (
                        "mutation SpacePostCreateMutation("
                        "$spaceId: String!, "
                        "$title: String!, "
                        "$content: String!"
                        ") { spacePostCreate("
                        "spaceId: $spaceId, "
                        "title: $title, "
                        "content: $content"
                        ") { post { pid url title } } }"
                    ),
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                post_data = (
                    data.get("data", {})
                    .get("spacePostCreate", {})
                    .get("post", {})
                )
                if post_data:
                    pid = post_data.get("pid", "")
                    purl = post_data.get("url", "")
                    logger.info(
                        "Quora [%s]: Space post created pid=%s",
                        self.account_id,
                        pid,
                    )
                    return {
                        "success": True,
                        "post_id": pid,
                        "post_url": purl,
                        "mode": "space_post",
                        "platform": self.platform,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Space post creation failed",
                        "platform": self.platform,
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}",
                    "platform": self.platform,
                }

        except Exception as e:
            logger.error(
                "Quora [%s]: Space post error: %s", self.account_id, e
            )
            return {"success": False, "error": str(e), "platform": self.platform}

    async def _webhook_fallback(self, content: dict) -> dict:
        """Send content to webhook for manual posting (Telegram, n8n, etc.)."""
        if not self._webhook_url:
            return {
                "success": False,
                "error": "No webhook URL configured",
                "platform": self.platform,
            }

        payload = {
            "platform": "quora",
            "account": self.account_id,
            "title": content.get("title", ""),
            "caption": content.get("caption", "")[:500],  # Preview only
            "question_url": content.get("question_url", ""),
            "mode": content.get("mode", "answer"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_needed": "manual_post",
        }

        try:
            async with httpx.AsyncClient(timeout=15) as wh_client:
                resp = await wh_client.post(self._webhook_url, json=payload)
            if resp.status_code in (200, 201, 202, 204):
                logger.info(
                    "Quora [%s]: Webhook fallback sent", self.account_id
                )
                return {
                    "success": True,
                    "post_id": "webhook_fallback",
                    "post_url": "",
                    "mode": "webhook_fallback",
                    "platform": self.platform,
                }
            else:
                return {
                    "success": False,
                    "error": f"Webhook HTTP {resp.status_code}",
                    "platform": self.platform,
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Webhook error: {e}",
                "platform": self.platform,
            }

    @staticmethod
    def _extract_question_id(url: str) -> str | None:
        """Extract Quora question ID (qid) from URL."""
        # Quora question URLs:
        #   https://www.quora.com/What-is-Python/answer/SomeUser
        #   https://www.quora.com/unanswered/What-is-Python
        #   Direct qid if already numeric
        if url.isdigit():
            return url
        # Try extracting the question slug from URL
        match = re.search(r"quora\.com/(?:unanswered/)?([^/?\s]+)", url)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _markdown_to_quora_json(text: str) -> str:
        """
        Convert markdown text to Quora's internal JSON content format.
        Quora uses a custom document format for rich text.
        """
        # Simplified: wrap each paragraph as a section
        paragraphs = text.split("\n\n")
        sections = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if para.startswith("# "):
                sections.append(
                    {"type": "heading", "text": para[2:].strip(), "level": 1}
                )
            elif para.startswith("## "):
                sections.append(
                    {"type": "heading", "text": para[3:].strip(), "level": 2}
                )
            elif para.startswith("- ") or para.startswith("* "):
                items = [
                    line.lstrip("-* ").strip()
                    for line in para.split("\n")
                    if line.strip()
                ]
                sections.append({"type": "list", "items": items})
            else:
                sections.append({"type": "paragraph", "text": para})
        return json.dumps({"sections": sections})

    async def get_metrics(self, post_id: str) -> dict | None:
        """Fetch answer/post engagement metrics."""
        if not self._connected or not self._client:
            return None
        try:
            resp = await self._client.post(
                "/graphql/gql_para_POST",
                json={
                    "queryName": "AnswerMetricsQuery",
                    "variables": {"aid": post_id},
                    "query": (
                        "query AnswerMetricsQuery($aid: String!) { "
                        "answer(aid: $aid) { "
                        "numUpvotes numViews numShares numComments url } }"
                    ),
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("data", {}).get("answer", {})
                if answer:
                    return {
                        "upvotes": answer.get("numUpvotes", 0),
                        "views": answer.get("numViews", 0),
                        "shares": answer.get("numShares", 0),
                        "comments": answer.get("numComments", 0),
                        "url": answer.get("url", ""),
                    }
            return None
        except Exception as e:
            logger.error(
                "Quora [%s]: Metrics failed: %s", self.account_id, e
            )
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
