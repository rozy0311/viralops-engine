"""
ViralOps Engine — Sendible API Publisher (REAL Implementation)

Official Sendible REST API — posts, schedules, media upload, analytics.
Python port of the Ruby SDK (radiatemedia/sendible).

API Base: https://api.sendible.com/api (v1/v2 endpoints)
Auth: application_id + username + api_key → AES-256-CBC encrypted access_key
      → /v1/auth → access_token (bearer)

This replaces Playwright/browser automation — ZERO bot detection risk.

SETUP:
1. Create a Developer Application at Sendible:
   https://snd-store.s3.amazonaws.com/developers/Login%20for%20Developer%20Apps.pdf
2. Get: application_id, shared_key, shared_iv
3. Get: your username + api_key from Sendible account
4. Set env vars (see .env.example)

Capabilities:
- Post to ALL platforms connected in Sendible (TikTok, IG, FB, Twitter, LinkedIn, etc.)
- Schedule posts with send_date_client
- Upload media (images/videos)
- List connected services/profiles
- Message reporting/analytics
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

# Optional: PyCryptodome for AES-256-CBC encryption
# Falls back to manual access_token if crypto unavailable
try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

logger = logging.getLogger("viralops.publisher.sendible")


class SendibleAuth:
    """
    Sendible authentication handler.

    Two auth modes:
    1. Full OAuth (application_id + shared_key + shared_iv + username + api_key)
       → encrypted access_key → /v1/auth → access_token
    2. Direct token (SENDIBLE_ACCESS_TOKEN) — if user already has a token
    """

    API_URL = "https://api.sendible.com/api"

    def __init__(self):
        self.application_id: str = os.environ.get("SENDIBLE_APPLICATION_ID", "")
        self.shared_key: str = os.environ.get("SENDIBLE_SHARED_KEY", "")
        self.shared_iv: str = os.environ.get("SENDIBLE_SHARED_IV", "")
        self.username: str = os.environ.get("SENDIBLE_USERNAME", "")
        self.api_key: str = os.environ.get("SENDIBLE_API_KEY", "")
        self._direct_token: str = os.environ.get("SENDIBLE_ACCESS_TOKEN", "")
        self._access_token: str | None = None
        self._token_time: float = 0

    @property
    def is_configured(self) -> bool:
        """Check if any auth method is configured."""
        if self._direct_token:
            return True
        return bool(
            self.application_id
            and self.shared_key
            and self.shared_iv
            and self.username
            and self.api_key
        )

    def _build_access_key(self) -> str:
        """
        Build encrypted access_key (AES-256-CBC).
        Matches Ruby: OpenSSL::Cipher::AES256.new(:CBC)
        """
        if not HAS_CRYPTO:
            raise RuntimeError(
                "pycryptodome required for Sendible OAuth. "
                "Install: pip install pycryptodome  "
                "OR set SENDIBLE_ACCESS_TOKEN directly."
            )

        payload = json.dumps(
            {
                "user_login": self.username,
                "user_api_key": self.api_key,
                "timestamp": int(time.time()),
            }
        )

        key = base64.b64decode(self.shared_key)
        iv = base64.b64decode(self.shared_iv)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded = pad(payload.encode("utf-8"), AES.block_size)
        encrypted = cipher.encrypt(padded)

        return base64.b64encode(encrypted).decode("utf-8")

    async def get_access_token(self, client: httpx.AsyncClient) -> str:
        """
        Get access token. Uses cache if < 55 minutes old.
        """
        # Mode 1: Direct token
        if self._direct_token:
            return self._direct_token

        # Mode 2: Cached token (valid ~1 hour)
        if self._access_token and (time.time() - self._token_time) < 3300:
            return self._access_token

        # Mode 3: Fetch new token via OAuth
        access_key = self._build_access_key()
        url = f"{self.API_URL}/v1/auth"
        params = {
            "app_id": self.application_id,
            "access_key": access_key,
        }

        resp = await client.get(url, params=params)

        if resp.status_code == 200:
            # Response is plain text token (not JSON)
            token = resp.text.strip()

            # Check for XML error
            if token.startswith("<error"):
                raise RuntimeError(f"Sendible auth error: {token}")

            self._access_token = token
            self._token_time = time.time()
            logger.info("Sendible: Auth token acquired")
            return token

        raise RuntimeError(
            f"Sendible auth failed: {resp.status_code} — {resp.text}"
        )


class SendiblePublisher:
    """
    Real Sendible REST API publisher for ViralOps.

    Posts to ANY platform connected in user's Sendible account:
    TikTok, Instagram, Facebook, Twitter/X, LinkedIn, YouTube, Pinterest, etc.
    """

    platform = "sendible"
    API_URL = "https://api.sendible.com/api"
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, account_id: str = "sendible_main"):
        self.account_id = account_id
        self.auth = SendibleAuth()
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._services: list[dict] = []  # Cached connected services

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        version: int = 1,
        params: dict | None = None,
        data: dict | None = None,
        files: dict | None = None,
    ) -> httpx.Response:
        """
        Make authenticated Sendible API request with retry logic.
        """
        client = await self._get_client()
        token = await self.auth.get_access_token(client)
        app_id = self.auth.application_id or ""

        url = f"{self.API_URL}/v{version}/{endpoint}.json"
        base_params = {
            "application_id": app_id,
            "access_token": token,
        }

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if method.upper() in ("GET", "DELETE"):
                    merged_params = {**base_params, **(params or {})}
                    resp = await client.request(
                        method, url, params=merged_params
                    )
                else:
                    # POST/PUT — params in body, auth in query
                    merged_data = {**base_params, **(data or {})}
                    if files:
                        resp = await client.request(
                            method, url, data=merged_data, files=files
                        )
                    else:
                        resp = await client.request(
                            method, url, data=merged_data
                        )

                if resp.status_code == 429:
                    retry_after = int(
                        resp.headers.get(
                            "Retry-After", self.RETRY_DELAY * (attempt + 1)
                        )
                    )
                    logger.warning(
                        "Sendible rate limited, retry in %ds", retry_after
                    )
                    import asyncio

                    await asyncio.sleep(retry_after)
                    continue

                return resp

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "Sendible timeout attempt %d/%d",
                    attempt + 1,
                    self.MAX_RETRIES,
                )
                import asyncio

                await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
            except Exception as e:
                last_error = e
                logger.error("Sendible request error: %s", e)
                import asyncio

                await asyncio.sleep(self.RETRY_DELAY)

        raise last_error or Exception("Sendible: Max retries exceeded")

    # ──────────────────────────────────────────────
    # Connection / Services
    # ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Test connection and cache connected services."""
        if not self.auth.is_configured:
            logger.error(
                "Sendible [%s]: Not configured — set SENDIBLE_* env vars",
                self.account_id,
            )
            return False

        try:
            client = await self._get_client()
            token = await self.auth.get_access_token(client)
            self._connected = bool(token)

            if self._connected:
                # Cache connected services
                await self.get_services()
                logger.info(
                    "Sendible [%s]: Connected, %d services available",
                    self.account_id,
                    len(self._services),
                )

            return self._connected
        except Exception as e:
            logger.error("Sendible [%s]: Connect failed: %s", self.account_id, e)
            return False

    async def get_services(self, force: bool = False) -> list[dict]:
        """
        List all connected social services in user's Sendible account.
        Endpoint: GET /v1/services.json
        Returns: [{service_id, service_type, name, ...}, ...]
        """
        if self._services and not force:
            return self._services

        try:
            resp = await self._request("GET", "services")
            if resp.status_code == 200:
                data = resp.json()
                # Sendible returns list of services
                if isinstance(data, list):
                    self._services = data
                elif isinstance(data, dict) and "result" in data:
                    result = data["result"]
                    self._services = result if isinstance(result, list) else []
                else:
                    self._services = []
                return self._services
            else:
                logger.error("Sendible services error: %s", resp.text)
                return []
        except Exception as e:
            logger.error("Sendible get_services error: %s", e)
            return []

    async def get_service_ids(self, platform_filter: str = "") -> list[str]:
        """
        Get service IDs for connected platforms.
        If platform_filter provided, filter by service type name.
        E.g., platform_filter='instagram' → only IG service IDs.
        """
        services = await self.get_services()
        ids = []
        for svc in services:
            svc_type = str(svc.get("service_type", "")).lower()
            svc_name = str(svc.get("name", "")).lower()
            svc_id = str(svc.get("id", svc.get("service_id", "")))

            if platform_filter:
                if (
                    platform_filter.lower() in svc_type
                    or platform_filter.lower() in svc_name
                ):
                    ids.append(svc_id)
            else:
                ids.append(svc_id)
        return ids

    # ──────────────────────────────────────────────
    # Publish / Schedule
    # ──────────────────────────────────────────────

    async def publish(self, content: dict) -> dict:
        """
        Publish content via Sendible to connected platforms.

        Endpoint: POST /v1/message.json
        Params:
          send_to: comma-separated service IDs (or "all")
          message_text: post body
          subject: title (for blog-type platforms)
          media: media file(s) to attach
          send_date_client: ISO datetime for scheduling (omit = post now)
          tags: comma-separated tags
          status: 1 = published, 0 = draft

        content dict keys:
          caption (str): Post text
          title (str, optional): Title for blog platforms
          media_url (str, optional): Image/video URL
          platforms (list[str], optional): Filter to specific platforms
          service_ids (list[str], optional): Direct service IDs to post to
          schedule_at (str, optional): ISO datetime for scheduled posting
          hashtags (list[str], optional): Tags to append
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {
                "success": False,
                "error": "Sendible not connected — check SENDIBLE_* env vars",
                "platform": self.platform,
            }

        # Determine recipients
        service_ids = content.get("service_ids", [])
        if not service_ids:
            platforms = content.get("platforms", [])
            if platforms:
                for pf in platforms:
                    ids = await self.get_service_ids(pf)
                    service_ids.extend(ids)
            else:
                # Post to all connected services
                service_ids = await self.get_service_ids()

        if not service_ids:
            return {
                "success": False,
                "error": "No Sendible services found for target platforms",
                "platform": self.platform,
            }

        # Build message
        caption = content.get("caption", "")
        hashtags = content.get("hashtags", [])
        if hashtags:
            tag_str = " ".join(
                f"#{t}" if not t.startswith("#") else t for t in hashtags
            )
            caption = f"{caption}\n\n{tag_str}".strip()

        post_data: dict[str, Any] = {
            "send_to": ",".join(service_ids),
            "message_text": caption,
            "status": "1",  # Published
        }

        # Optional: title for blog platforms
        title = content.get("title", "")
        if title:
            post_data["subject"] = title

        # Optional: scheduling
        schedule_at = content.get("schedule_at", "")
        if schedule_at:
            post_data["send_date_client"] = schedule_at
            post_data["status"] = "0"  # Scheduled = draft until send time

        # Optional: tags
        tags = content.get("tags", content.get("hashtags", []))
        if tags:
            post_data["tags"] = ",".join(tags)

        # Optional: media upload
        files = None
        media_url = content.get("media_url", "")
        if media_url:
            try:
                media_data = await self._download_media(media_url)
                if media_data:
                    files = {"media": media_data}
            except Exception as e:
                logger.warning("Sendible media download failed: %s", e)

        try:
            resp = await self._request(
                "POST", "message", data=post_data, files=files
            )

            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", data)

                status = result.get("status", "")
                message_id = str(
                    result.get("message_id", result.get("id", ""))
                )

                if status == "success" or message_id:
                    logger.info(
                        "Sendible [%s]: Published message_id=%s to %d services",
                        self.account_id,
                        message_id,
                        len(service_ids),
                    )
                    return {
                        "success": True,
                        "post_id": message_id,
                        "post_url": "",  # Sendible doesn't return direct URLs
                        "platform": self.platform,
                        "services_count": len(service_ids),
                        "service_ids": service_ids,
                        "scheduled": bool(schedule_at),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Sendible post failed: {json.dumps(result)}",
                        "platform": self.platform,
                    }

            return {
                "success": False,
                "error": f"Sendible API {resp.status_code}: {resp.text}",
                "platform": self.platform,
            }

        except Exception as e:
            logger.error("Sendible [%s]: Publish error: %s", self.account_id, e)
            return {
                "success": False,
                "error": str(e),
                "platform": self.platform,
            }

    async def schedule(self, content: dict) -> dict:
        """
        Schedule content for future posting.
        Convenience wrapper — sets schedule_at and calls publish().
        """
        if "schedule_at" not in content:
            return {
                "success": False,
                "error": "schedule_at is required for scheduling",
                "platform": self.platform,
            }
        return await self.publish(content)

    # ──────────────────────────────────────────────
    # Messages / Reports
    # ──────────────────────────────────────────────

    async def get_messages(
        self,
        status: str = "",
        per_page: int = 20,
        page: int = 1,
    ) -> list[dict]:
        """
        List messages.
        Endpoint: GET /v1/messages.json
        status: "" = all, "scheduled", "sent", "draft"
        """
        params: dict[str, Any] = {"per_page": per_page, "page": page}
        if status:
            params["status"] = status

        try:
            resp = await self._request("GET", "messages", params=params)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                return data.get("result", [])
            return []
        except Exception as e:
            logger.error("Sendible get_messages error: %s", e)
            return []

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message by ID."""
        try:
            resp = await self._request(
                "DELETE", "message", params={"message_id": message_id}
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error("Sendible delete_message error: %s", e)
            return False

    async def get_reports(
        self, per_page: int = 20, page: int = 1
    ) -> list[dict]:
        """
        Get message reports / analytics.
        Endpoint: GET /v1/message-reports.json
        """
        try:
            resp = await self._request(
                "GET",
                "message-reports",
                params={"per_page": per_page, "page": page},
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                return data.get("result", [])
            return []
        except Exception as e:
            logger.error("Sendible get_reports error: %s", e)
            return []

    async def get_account_details(self) -> dict:
        """
        Get account info (plan, usage, etc.).
        Endpoint: GET /v1/account-details.json
        """
        try:
            resp = await self._request("GET", "account-details")
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception as e:
            logger.error("Sendible account_details error: %s", e)
            return {}

    # ──────────────────────────────────────────────
    # URL Shortening
    # ──────────────────────────────────────────────

    async def shorten_url(self, url: str) -> str:
        """
        Shorten a URL via Sendible's built-in shortener.
        Endpoint: GET /v1/shorten.json?url=...
        """
        try:
            resp = await self._request(
                "GET", "shorten", params={"url": url}
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", {}).get("url", url)
            return url
        except Exception as e:
            logger.warning("Sendible shorten error: %s", e)
            return url

    # ──────────────────────────────────────────────
    # Media helpers
    # ──────────────────────────────────────────────

    async def _download_media(
        self, url: str
    ) -> tuple[str, bytes, str] | None:
        """Download media from URL, return (filename, bytes, content_type)."""
        try:
            async with httpx.AsyncClient(timeout=60) as dl:
                resp = await dl.get(url, follow_redirects=True)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "image/jpeg")
            ext = content_type.split("/")[-1].split(";")[0]
            filename = f"media.{ext}"
            return (filename, resp.content, content_type)
        except Exception as e:
            logger.error("Sendible media download failed: %s", e)
            return None

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False

    async def test_connection(self) -> dict:
        """Test if Sendible credentials work. Returns status dict."""
        try:
            connected = await self.connect()
            if connected:
                services = await self.get_services()
                service_names = [
                    s.get("name", s.get("service_type", "unknown"))
                    for s in services
                ]
                return {
                    "connected": True,
                    "services_count": len(services),
                    "service_names": service_names,
                    "account": self.auth.username or "(direct token)",
                }
            return {"connected": False, "error": "Auth failed"}
        except Exception as e:
            return {"connected": False, "error": str(e)}
