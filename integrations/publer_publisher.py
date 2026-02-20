"""
ViralOps Engine — Publer API Publisher (REST API)

Official Publer REST API — posts, schedules, media upload, analytics.
Bridge publisher for ViralOps Engine.

API Base: https://app.publer.com/api/v1/
Auth: Bearer-API token + Workspace ID header
Docs: https://publer.com/docs

SETUP:
1. Sign up at https://publer.com (Business plan required for API)
2. Go to Settings → Access & Login → API Keys
3. Create API key with scopes: workspaces, accounts, posts, media, analytics
4. Set env vars: PUBLER_API_KEY, PUBLER_WORKSPACE_ID
5. Connect your social accounts in Publer dashboard

Capabilities:
- Post to 13+ platforms (TikTok, IG, FB, Twitter/X, LinkedIn, YT, Pinterest, etc.)
- Schedule posts with per-account scheduling
- Bulk scheduling (up to 500 posts/request)
- Upload media (images/videos)
- List connected accounts/workspaces
- Analytics & insights
- Auto-scheduling to optimal time slots
- Async workflow: Submit → job_id → poll status

Pricing: ~$10/mo per social account (Business plan)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

# Auto-load .env (best-effort) so CLI utilities and one-off scripts
# don't fail when they forget to call load_dotenv().
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass

logger = logging.getLogger("viralops.publisher.publer")

# ── Network provider mapping ──
# Maps ViralOps platform names to Publer network provider keys
PLATFORM_TO_PUBLER_NETWORK = {
    "facebook": "facebook",
    "instagram": "instagram",
    "twitter": "twitter",
    "linkedin": "linkedin",
    "pinterest": "pinterest",
    "youtube": "youtube",
    "tiktok": "tiktok",
    "google_business": "google",
    "telegram": "telegram",
    "mastodon": "mastodon",
    "threads": "threads",
    "bluesky": "bluesky",
    "wordpress": "wordpress_basic",
}

# ── Content type mapping ──
# Default content types per platform
PLATFORM_DEFAULT_CONTENT_TYPE = {
    "facebook": "status",
    "instagram": "photo",
    "twitter": "status",
    "linkedin": "status",
    "pinterest": "pin",
    "youtube": "video",
    "tiktok": "video",
    "google": "status",
    "telegram": "status",
    "mastodon": "status",
    "threads": "status",
    "bluesky": "status",
    "wordpress_basic": "article",
}


class PublerPublisher:
    """
    Real Publer REST API publisher for ViralOps.

    Posts to ANY platform connected in user's Publer account:
    TikTok, Instagram, Facebook, Twitter/X, LinkedIn, YouTube, Pinterest,
    Google Business, Telegram, Mastodon, Threads, Bluesky.

    Async workflow:
    1. Submit post → receive job_id
    2. Poll /job_status/{job_id} until complete
    3. Return result
    """

    platform = "publer"
    API_URL = "https://app.publer.com/api/v1"
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    JOB_POLL_INTERVAL = 2.0  # seconds between polls
    JOB_POLL_TIMEOUT = 60.0  # max seconds to wait for job

    def __init__(self, account_id: str = "publer_main"):
        self.account_id = account_id
        self.api_key: str = os.environ.get("PUBLER_API_KEY", "")
        self.workspace_id: str = os.environ.get("PUBLER_WORKSPACE_ID", "")
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._accounts: list[dict] = []  # Cached connected accounts
        self._workspaces: list[dict] = []

    @property
    def is_configured(self) -> bool:
        """Check if Publer credentials are set."""
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth."""
        h = {
            "Authorization": f"Bearer-API {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.workspace_id:
            h["Publer-Workspace-Id"] = self.workspace_id
        return h

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=self._headers(),
            )
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> httpx.Response:
        """
        Make authenticated Publer API request with retry logic.
        Handles 429 rate limiting with retry-after.
        """
        client = await self._get_client()
        url = f"{self.API_URL}/{endpoint}"

        transient_statuses = {500, 502, 503, 504, 520, 521, 522, 523, 524}
        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = await client.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                    headers=self._headers(),
                )

                if resp.status_code == 429:
                    retry_after = int(
                        resp.headers.get(
                            "Retry-After", self.RETRY_DELAY * (attempt + 1)
                        )
                    )
                    logger.warning(
                        "Publer rate limited, retry in %ds", retry_after
                    )
                    await asyncio.sleep(retry_after)
                    continue

                # Retry transient upstream/Cloudflare failures (e.g., HTTP 520).
                if resp.status_code in transient_statuses:
                    delay = min(30.0, float(self.RETRY_DELAY) * (attempt + 1) * 2)
                    delay += random.uniform(0.0, 0.6)
                    logger.warning(
                        "Publer transient HTTP %d on %s %s (attempt %d/%d) — retry in %.1fs",
                        resp.status_code,
                        method,
                        endpoint,
                        attempt + 1,
                        self.MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                return resp

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "Publer timeout attempt %d/%d",
                    attempt + 1,
                    self.MAX_RETRIES,
                )
                await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
            except Exception as e:
                last_error = e
                logger.error("Publer request error: %s", e)
                await asyncio.sleep(self.RETRY_DELAY)

        raise last_error or Exception("Publer: Max retries exceeded")

    # ──────────────────────────────────────────────
    # Job Status Polling (Publer async workflow)
    # ──────────────────────────────────────────────

    async def _poll_job(self, job_id: str) -> dict:
        """
        Poll job status until complete or timeout.
        Publer uses async workflow: submit → job_id → poll until done.
        """
        start = time.time()
        while (time.time() - start) < self.JOB_POLL_TIMEOUT:
            try:
                resp = await self._request("GET", f"job_status/{job_id}")
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("data", data)
                    status = result.get("status", "")

                    if status in ("complete", "completed"):
                        return {
                            "success": True,
                            "status": status,
                            "payload": result.get("result", result.get("payload", {})),
                        }
                    elif status == "failed":
                        payload = result.get("result", result.get("payload", {}))
                        error_msg = ""
                        if isinstance(payload, dict):
                            failures = payload.get("failures", {})
                            error_msg = failures.get("error", json.dumps(payload))
                        return {
                            "success": False,
                            "status": "failed",
                            "error": error_msg or "Job failed",
                        }
                    # Still working — wait and retry
                    await asyncio.sleep(self.JOB_POLL_INTERVAL)
                else:
                    logger.warning("Publer job_status %d: %s", resp.status_code, resp.text)
                    await asyncio.sleep(self.JOB_POLL_INTERVAL)
            except Exception as e:
                logger.warning("Publer poll error: %s", e)
                await asyncio.sleep(self.JOB_POLL_INTERVAL)

        return {"success": False, "status": "timeout", "error": "Job polling timed out"}

    # ──────────────────────────────────────────────
    # Connection / Accounts
    # ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Test connection and cache user info + accounts."""
        if not self.is_configured:
            logger.error(
                "Publer [%s]: Not configured — set PUBLER_API_KEY env var",
                self.account_id,
            )
            return False

        try:
            # Validate credentials
            resp = await self._request("GET", "users/me")
            if resp.status_code != 200:
                logger.error("Publer [%s]: Auth failed: %s", self.account_id, resp.text)
                return False

            self._connected = True

            # Auto-discover workspace if not set
            if not self.workspace_id:
                workspaces = await self.get_workspaces()
                if workspaces:
                    self.workspace_id = str(workspaces[0].get("id", ""))
                    logger.info(
                        "Publer [%s]: Auto-selected workspace: %s",
                        self.account_id,
                        self.workspace_id,
                    )

            # Cache connected accounts
            await self.get_accounts()
            logger.info(
                "Publer [%s]: Connected, %d accounts available",
                self.account_id,
                len(self._accounts),
            )
            return True

        except Exception as e:
            logger.error("Publer [%s]: Connect failed: %s", self.account_id, e)
            return False

    async def get_workspaces(self, force: bool = False) -> list[dict]:
        """
        List all workspaces.
        Endpoint: GET /workspaces
        """
        if self._workspaces and not force:
            return self._workspaces

        try:
            resp = await self._request("GET", "workspaces")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data:
                    self._workspaces = data["data"]
                elif isinstance(data, list):
                    self._workspaces = data
                else:
                    self._workspaces = [data] if data else []
                return self._workspaces
            else:
                logger.error("Publer workspaces error: %s", resp.text)
                return []
        except Exception as e:
            logger.error("Publer get_workspaces error: %s", e)
            return []

    async def get_accounts(self, force: bool = False) -> list[dict]:
        """
        List all connected social accounts in workspace.
        Endpoint: GET /accounts
        """
        if self._accounts and not force:
            return self._accounts

        try:
            resp = await self._request("GET", "accounts")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data:
                    self._accounts = data["data"]
                elif isinstance(data, list):
                    self._accounts = data
                else:
                    self._accounts = []
                return self._accounts
            else:
                logger.error("Publer accounts error: %s", resp.text)
                return []
        except Exception as e:
            logger.error("Publer get_accounts error: %s", e)
            return []

    async def get_account_ids(self, platform_filter: str = "") -> list[str]:
        """
        Get account IDs for connected platforms.
        If platform_filter provided, filter by platform type.
        E.g., platform_filter='tiktok' → only TikTok account IDs.
        """
        accounts = await self.get_accounts()
        ids = []
        for acc in accounts:
            acc_type = str(acc.get("type", acc.get("provider", ""))).lower()
            acc_name = str(acc.get("name", "")).lower()
            acc_id = str(acc.get("id", acc.get("_id", "")))

            if platform_filter:
                pf = platform_filter.lower()
                # Map ViralOps names to Publer provider names
                publer_key = PLATFORM_TO_PUBLER_NETWORK.get(pf, pf)
                # Publer uses provider aliases like fb_page / pin_business.
                alias_ok = False
                if pf == "facebook" and (acc_type.startswith("fb") or "facebook" in acc_type):
                    alias_ok = True
                if pf == "pinterest" and (acc_type.startswith("pin") or "pinterest" in acc_type):
                    alias_ok = True

                if publer_key in acc_type or pf in acc_type or pf in acc_name or alias_ok:
                    ids.append(acc_id)
            else:
                ids.append(acc_id)
        return ids

    # ──────────────────────────────────────────────
    # Media Upload
    # ──────────────────────────────────────────────

    async def upload_media(self, url: str = "", file_path: str = "") -> dict | None:
        """
        Upload media to Publer Media Library.
        Returns media dict with 'id' and 'path' for use in posts.

        Supports URL upload (async via /media/from-url) or
        direct file upload (sync via POST /media multipart).
        """
        transient_statuses = {500, 502, 503, 504, 520, 521, 522, 523, 524}
        try:
            if url:
                # URL-based upload — async, returns job_id
                # Publer API: POST /media/from-url
                resp = await self._request(
                    "POST",
                    "media/from-url",
                    json_data={
                        "media": [{"url": url}],
                        "type": "single",
                        "direct_upload": False,
                        "in_library": False,
                    },
                )
            elif file_path:
                # Direct file upload — multipart/form-data
                # Publer API: POST /media
                # IMPORTANT: Use a fresh client without Content-Type default.
                # The shared client has Content-Type: application/json which
                # conflicts with httpx's auto-set multipart/form-data boundary.
                upload_headers = {
                    "Authorization": f"Bearer-API {self.api_key}",
                    "Accept": "application/json",
                }
                if self.workspace_id:
                    upload_headers["Publer-Workspace-Id"] = self.workspace_id

                resp: httpx.Response | None = None
                max_attempts = max(2, int(self.MAX_RETRIES))
                for attempt in range(max_attempts):
                    async with httpx.AsyncClient(timeout=60.0) as upload_client:
                        with open(file_path, "rb") as f:
                            files = {"file": f}
                            resp = await upload_client.post(
                                f"{self.API_URL}/media",
                                files=files,
                                headers=upload_headers,
                            )

                    if resp.status_code in transient_statuses and attempt < (max_attempts - 1):
                        delay = min(30.0, float(self.RETRY_DELAY) * (attempt + 1) * 2)
                        delay += random.uniform(0.0, 0.6)
                        logger.warning(
                            "Publer media upload transient HTTP %d (attempt %d/%d) — retry in %.1fs",
                            resp.status_code,
                            attempt + 1,
                            max_attempts,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
            else:
                return None

            if resp.status_code in (200, 201, 202):
                data = resp.json()
                # URL uploads are async — poll job_id
                job_id = data.get("job_id") or data.get("data", {}).get("job_id")
                if job_id:
                    result = await self._poll_job(job_id)
                    if result["success"]:
                        payload = result.get("payload", {})
                        # Payload is a list for URL uploads
                        if isinstance(payload, list) and len(payload) > 0:
                            return payload[0]
                        return payload
                    return None
                # Direct upload returns media object immediately
                return data.get("data", data)
            else:
                body = (resp.text or "")
                snippet = body[:600].replace("\r", " ").replace("\n", " ")
                logger.error("Publer media upload error (HTTP %d): %s", resp.status_code, snippet)
                return None

        except Exception as e:
            logger.error("Publer media upload failed: %s", e)
            return None

    # ──────────────────────────────────────────────
    # Publish / Schedule
    # ──────────────────────────────────────────────

    async def publish(self, content: dict) -> dict:
        """
        Publish content via Publer to connected platforms.

        Publer API uses async workflow:
        1. POST /posts/schedule or /posts/schedule/publish
        2. Receive job_id
        3. Poll /job_status/{job_id}

        content dict keys:
          caption (str): Post text
          title (str, optional): Title for blog/article platforms
          media_url (str, optional): Image/video URL to attach
          media_ids (list[str], optional): Pre-uploaded media IDs
          platforms (list[str], optional): Filter to specific platforms
          account_ids (list[str], optional): Direct account IDs to post to
          schedule_at (str, optional): ISO datetime for scheduled posting
          hashtags (list[str], optional): Tags to append
          content_type (str, optional): Override content type (status/photo/video/pin/article)
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {
                "success": False,
                "error": "Publer not connected — check PUBLER_API_KEY env var",
                "platform": self.platform,
            }

        # ── Determine target accounts ──
        account_ids = content.get("account_ids", [])
        if not account_ids:
            platforms = content.get("platforms", [])
            if platforms:
                for pf in platforms:
                    ids = await self.get_account_ids(pf)
                    account_ids.extend(ids)
            else:
                account_ids = await self.get_account_ids()

        if not account_ids:
            return {
                "success": False,
                "error": "No Publer accounts found for target platforms",
                "platform": self.platform,
            }

        # ── Build caption with hashtags ──
        caption = content.get("caption", content.get("body", ""))
        hashtags = content.get("hashtags", [])
        if hashtags:
            tag_str = " ".join(
                f"#{t.lstrip('#')}" for t in hashtags
            )
            caption = f"{caption}\n\n{tag_str}".strip()

        # ── Build network content ──
        # Determine which networks these accounts belong to
        accounts_info = await self.get_accounts()
        networks_content = {}
        target_accounts = []

        for acc_id in account_ids:
            acc_info = next(
                (a for a in accounts_info if str(a.get("id", a.get("_id", ""))) == acc_id),
                None,
            )
            if not acc_info:
                continue

            provider = str(acc_info.get("type", acc_info.get("provider", ""))).lower()

            # Normalize provider aliases returned by Publer accounts.
            # Examples: fb_page, pin_business. Publer networks expect canonical keys.
            if provider.startswith("fb"):
                provider = "facebook"
            elif provider.startswith("pin"):
                provider = "pinterest"
            publer_network = None
            for viralops_name, publer_key in PLATFORM_TO_PUBLER_NETWORK.items():
                if viralops_name in provider or publer_key in provider:
                    publer_network = publer_key
                    break

            if not publer_network:
                publer_network = provider

            # Build network content if not already added
            if publer_network not in networks_content:
                content_type = content.get(
                    "content_type",
                    PLATFORM_DEFAULT_CONTENT_TYPE.get(publer_network, "status"),
                )
                network_data: dict[str, Any] = {
                    "type": content_type,
                    "text": caption,
                }

                # Destination link (Pinterest Pin URL). Publer stores this as post.url.
                if publer_network == "pinterest":
                    dest_url = str(content.get("url", "") or "").strip()
                    if dest_url:
                        network_data["url"] = dest_url
                # Add title for article/blog types
                title = content.get("title", "")
                if title and content_type in ("article", "pin"):
                    network_data["title"] = title

                # Add media references (pre-uploaded)
                media_ids = content.get("media_ids", [])
                media_paths = content.get("media_paths", [])  # optional CDN paths
                if media_ids:
                    media_list = []
                    for idx, mid in enumerate(media_ids):
                        media_entry: dict[str, Any] = {"id": mid}
                        if idx < len(media_paths):
                            media_entry["path"] = media_paths[idx]
                        media_list.append(media_entry)
                    network_data["media"] = media_list

                networks_content[publer_network] = network_data

            # Build account entry
            acc_entry: dict[str, Any] = {"id": acc_id}
            schedule_at = content.get("schedule_at", "")
            if schedule_at:
                acc_entry["scheduled_at"] = schedule_at

            # Pinterest requires an album/board selection (Publer calls it album).
            # If not provided, Publer can reject with: "Album can't be blank".
            if publer_network == "pinterest":
                album_id = str(content.get("album_id", "") or "").strip()
                if album_id:
                    acc_entry["album_id"] = album_id

            target_accounts.append(acc_entry)

        if not target_accounts:
            return {
                "success": False,
                "error": "Could not resolve any target accounts",
                "platform": self.platform,
            }

        # ── Handle media upload from URL ──
        media_url = content.get("media_url", "")
        if media_url and not content.get("media_ids"):
            media_result = await self.upload_media(url=media_url)
            if media_result and "id" in media_result:
                media_ref = [{
                    "id": media_result["id"],
                    "path": media_result.get("path", ""),
                }]
                # Detect actual media type (photo vs video) from upload result
                actual_media_type = media_result.get("type", "")  # "photo" or "video"
                for net_key in networks_content:
                    networks_content[net_key]["media"] = media_ref
                    # Update content type if media is photo but default was video
                    if actual_media_type == "photo" and networks_content[net_key].get("type") == "video":
                        networks_content[net_key]["type"] = "photo"

        # ── Handle local file upload ──
        media_local_path = content.get("media_local_path", "")
        if media_local_path and not content.get("media_ids") and not media_url:
            if os.path.isfile(media_local_path):
                logger.info("Publer: uploading local file %s", media_local_path)
                media_result = await self.upload_media(file_path=media_local_path)
                if media_result and "id" in media_result:
                    media_ref = [{
                        "id": media_result["id"],
                        "path": media_result.get("path", ""),
                    }]
                    actual_media_type = media_result.get("type", "photo")
                    for net_key in networks_content:
                        networks_content[net_key]["media"] = media_ref
                        if actual_media_type == "photo" and networks_content[net_key].get("type") == "video":
                            networks_content[net_key]["type"] = "photo"
                else:
                    logger.warning("Publer: local file upload failed for %s", media_local_path)
            else:
                logger.warning("Publer: local file not found: %s", media_local_path)

        # ── Build request body ──
        # Publer supports many post states (draft/scheduled/published_*).
        # If we're hitting the publish endpoint, we must NOT send draft state,
        # otherwise the post may be created as a draft and never posted.
        schedule_at = content.get("schedule_at", "")
        if schedule_at:
            state = "scheduled"
            endpoint = "posts/schedule"
        else:
            state = "published"
            endpoint = "posts/schedule/publish"

        post_body = {
            "bulk": {
                "state": state,
                "posts": [
                    {
                        "networks": networks_content,
                        "accounts": target_accounts,
                    }
                ],
            }
        }

        # ── Submit post ──
        try:
            resp = await self._request("POST", endpoint, json_data=post_body)

            if resp.status_code in (200, 201):
                data = resp.json()
                job_id = data.get("data", data).get("job_id", "")

                if job_id:
                    # Poll for completion
                    job_result = await self._poll_job(job_id)

                    if job_result["success"]:
                        payload = job_result.get("payload", {})

                        # Publer payload can be a dict or a list of job items.
                        # We must detect partial failures deterministically.
                        failures_out: list[dict[str, Any]] = []
                        if isinstance(payload, dict):
                            failures = payload.get("failures")
                            if failures:
                                # keep shape but also normalize to list
                                failures_out.append({"failures": failures})
                        elif isinstance(payload, list):
                            for item in payload:
                                if not isinstance(item, dict):
                                    continue
                                if item.get("failure"):
                                    failures_out.append(item.get("failure"))
                                    continue
                                post = item.get("post") if isinstance(item.get("post"), dict) else {}
                                state = str(post.get("state", "") or "").lower()
                                if state == "failed" or post.get("error") or post.get("details", {}).get("error"):
                                    failures_out.append(
                                        {
                                            "account_id": post.get("account_id"),
                                            "provider": post.get("provider"),
                                            "message": post.get("error")
                                            or (post.get("details", {}) or {}).get("error")
                                            or "Post failed",
                                            "post_id": post.get("id"),
                                        }
                                    )

                        ok = len(failures_out) == 0

                        error_summary = ""
                        if not ok:
                            # Best-effort: extract a single readable message.
                            first = failures_out[0] if failures_out else {}
                            if isinstance(first, dict):
                                error_summary = str(first.get("message") or first.get("error") or "Publish failed")
                            else:
                                error_summary = "Publish failed"

                        logger.info(
                            "Publer [%s]: Job complete (job=%s, accounts=%d, ok=%s)",
                            self.account_id,
                            job_id,
                            len(target_accounts),
                            ok,
                        )

                        created_post_ids: list[str] = []
                        if isinstance(payload, list):
                            for item in payload:
                                if not isinstance(item, dict):
                                    continue
                                post = item.get("post") if isinstance(item.get("post"), dict) else None
                                if post and post.get("id"):
                                    created_post_ids.append(str(post.get("id")))
                        return {
                            "success": bool(ok),
                            "post_id": job_id,
                            "post_url": "",
                            "platform": self.platform,
                            "accounts_count": len(target_accounts),
                            "account_ids": account_ids,
                            "scheduled": bool(schedule_at),
                            "job_result": payload,
                            "failures": failures_out,
                            "error": error_summary,
                            "created_post_ids": created_post_ids,
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Publer job failed: {job_result.get('error', 'unknown')}",
                            "platform": self.platform,
                            "job_id": job_id,
                            "job_result": job_result.get("payload", {}),
                        }

                # No job_id — might be direct success
                # In practice we REQUIRE a job_id to verify per-account outcomes.
                return {
                    "success": False,
                    "error": "Publer response missing job_id (cannot verify publish result)",
                    "platform": self.platform,
                    "accounts_count": len(target_accounts),
                    "scheduled": bool(schedule_at),
                    "raw": data,
                }

            return {
                "success": False,
                "error": f"Publer API {resp.status_code}: {resp.text}",
                "platform": self.platform,
            }

        except Exception as e:
            logger.error("Publer [%s]: Publish error: %s", self.account_id, e)
            return {
                "success": False,
                "error": str(e),
                "platform": self.platform,
            }

    async def schedule(self, content: dict) -> dict:
        """
        Schedule content for future posting.
        Convenience wrapper — ensures schedule_at is set.
        """
        if "schedule_at" not in content:
            return {
                "success": False,
                "error": "schedule_at is required for scheduling",
                "platform": self.platform,
            }
        return await self.publish(content)

    async def bulk_schedule(self, posts: list[dict]) -> dict:
        """
        Bulk schedule up to 500 posts in a single request.
        Each post dict follows the same format as publish().
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        bulk_posts = []
        for post_content in posts:
            # Build per-post structure
            caption = post_content.get("caption", "")
            hashtags = post_content.get("hashtags", [])
            if hashtags:
                tag_str = " ".join(f"#{t.lstrip('#')}" for t in hashtags)
                caption = f"{caption}\n\n{tag_str}".strip()

            account_ids = post_content.get("account_ids", [])
            if not account_ids:
                platforms = post_content.get("platforms", [])
                for pf in platforms:
                    ids = await self.get_account_ids(pf)
                    account_ids.extend(ids)

            accounts = []
            for aid in account_ids:
                entry: dict[str, Any] = {"id": aid}
                if post_content.get("schedule_at"):
                    entry["scheduled_at"] = post_content["schedule_at"]
                accounts.append(entry)

            # Default network content
            networks = {}
            for pf in post_content.get("platforms", ["facebook"]):
                net_key = PLATFORM_TO_PUBLER_NETWORK.get(pf, pf)
                ct = PLATFORM_DEFAULT_CONTENT_TYPE.get(net_key, "status")
                networks[net_key] = {"type": ct, "text": caption}

            bulk_posts.append({"networks": networks, "accounts": accounts})

        body = {
            "bulk": {
                "state": "scheduled",
                "posts": bulk_posts[:500],  # Publer limit
            }
        }

        try:
            resp = await self._request("POST", "posts/schedule", json_data=body)
            if resp.status_code in (200, 201):
                data = resp.json()
                job_id = data.get("data", data).get("job_id", "")
                if job_id:
                    job_result = await self._poll_job(job_id)
                    return {
                        "success": job_result["success"],
                        "job_id": job_id,
                        "posts_count": len(bulk_posts),
                        "result": job_result,
                    }
                return {"success": True, "posts_count": len(bulk_posts)}
            return {"success": False, "error": f"API {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────
    # Posts / Analytics
    # ──────────────────────────────────────────────

    async def get_posts(
        self,
        status: str = "",
        limit: int = 20,
        page: int = 1,
    ) -> list[dict]:
        """
        List posts.
        Endpoint: GET /posts
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return []

        params: dict[str, Any] = {"limit": limit, "page": page}
        if status:
            # Publer payloads use the term `state`. Some API variants accept `status`.
            # Send both for compatibility.
            params["status"] = status
            params["state"] = status

        try:
            resp = await self._request("GET", "posts", params=params)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        return data["data"]
                    # Publer sometimes returns {"posts": [...]} 
                    if "posts" in data and isinstance(data["posts"], list):
                        return data["posts"]
                if isinstance(data, list):
                    return data
                return []
            logger.warning("Publer get_posts failed: %d %s", resp.status_code, resp.text[:300])
            return []
        except Exception as e:
            logger.error("Publer get_posts error: %s", e)
            return []

    async def get_post(self, post_id: str) -> dict | None:
        """Get a single post by ID.

        Endpoint: GET /posts/{id}
        Returns the post dict (commonly under key `data`) or None.
        """
        pid = str(post_id or "").strip()
        if not pid:
            return None
        if not self._connected:
            await self.connect()
        if not self._connected:
            return None

        try:
            resp = await self._request("GET", f"posts/{pid}")
            if resp.status_code != 200:
                return None
            data = resp.json()
            if isinstance(data, dict) and isinstance(data.get("data"), dict):
                return data["data"]
            if isinstance(data, dict) and isinstance(data.get("post"), dict):
                return data["post"]
            if isinstance(data, dict):
                return data
            return None
        except Exception as e:
            logger.error("Publer get_post error: %s", e)
            return None

    async def delete_post(self, post_id: str) -> bool:
        """Delete a post by ID."""
        try:
            if not self._connected:
                await self.connect()
            if not self._connected:
                return False
            resp = await self._request("DELETE", f"posts/{post_id}")
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.error("Publer delete_post error: %s", e)
            return False

    async def get_analytics(self, account_id: str = "", chart: str = "") -> dict:
        """
        Get analytics data.
        Endpoint: GET /analytics/charts or /analytics/chart_data
        """
        try:
            params: dict[str, Any] = {}
            if account_id:
                params["account_id"] = account_id
            if chart:
                params["chart"] = chart

            resp = await self._request("GET", "analytics/charts", params=params)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception as e:
            logger.error("Publer analytics error: %s", e)
            return {}

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
        """Test if Publer credentials work. Returns status dict."""
        try:
            connected = await self.connect()
            if connected:
                accounts = await self.get_accounts()
                account_names = [
                    f"{a.get('name', 'unknown')} ({a.get('type', a.get('provider', '?'))})"
                    for a in accounts
                ]
                return {
                    "connected": True,
                    "accounts_count": len(accounts),
                    "account_names": account_names,
                    "workspace_id": self.workspace_id,
                }
            return {"connected": False, "error": "Auth failed"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"PublerPublisher(account={self.account_id}, status={status}, accounts={len(self._accounts)})"
