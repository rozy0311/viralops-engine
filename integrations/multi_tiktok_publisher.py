"""
Multi-Account TikTok Publisher — Post to multiple TikTok accounts.

Supports N TikTok accounts via env vars:
  TIKTOK_1_ACCESS_TOKEN, TIKTOK_1_OPEN_ID, TIKTOK_1_LABEL
  TIKTOK_2_ACCESS_TOKEN, TIKTOK_2_OPEN_ID, TIKTOK_2_LABEL
  ... up to TIKTOK_99_*

Posting modes:
  - broadcast: Post to ALL accounts simultaneously
  - round_robin: Rotate through accounts (1→2→3→1→...)
  - specific: Post to a named account by label or index

TikTok Content Posting API v2:
  - Video: POST /v2/post/publish/video/init/ (URL-pull)
  - Photo: POST /v2/post/publish/content/init/ (photo_images)

API Docs: https://developers.tiktok.com/doc/content-posting-api-get-started
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("viralops.multi_tiktok")

TIKTOK_API_BASE = "https://open.tiktokapis.com/v2"


@dataclass
class TikTokAccount:
    """Single TikTok account credentials + metadata."""
    index: int
    label: str
    access_token: str
    open_id: str
    posts_count: int = 0
    last_post_at: str = ""
    enabled: bool = True


class MultiTikTokPublisher:
    """
    Post to multiple TikTok accounts.

    Usage:
        publisher = MultiTikTokPublisher()
        # Broadcast to all accounts
        results = await publisher.publish_to_all(content)
        # Round-robin
        result = await publisher.publish_round_robin(content)
        # Specific account
        result = await publisher.publish_to_account("main", content)
    """

    def __init__(self):
        self.accounts: list[TikTokAccount] = []
        self._round_robin_index: int = 0
        self._discover_accounts()

    def _discover_accounts(self):
        """Discover TikTok accounts from environment variables."""
        self.accounts = []

        # Pattern: TIKTOK_{N}_ACCESS_TOKEN, TIKTOK_{N}_OPEN_ID
        for i in range(1, 100):
            token = os.environ.get(f"TIKTOK_{i}_ACCESS_TOKEN", "")
            open_id = os.environ.get(f"TIKTOK_{i}_OPEN_ID", "")
            if not token:
                # Also try without number for single account compat
                if i == 1:
                    token = os.environ.get("TIKTOK_ACCESS_TOKEN", "")
                    open_id = os.environ.get("TIKTOK_OPEN_ID", "")
                if not token:
                    break

            label = os.environ.get(f"TIKTOK_{i}_LABEL", f"account_{i}")
            enabled = os.environ.get(
                f"TIKTOK_{i}_ENABLED", "true"
            ).lower() != "false"

            account = TikTokAccount(
                index=i,
                label=label,
                access_token=token,
                open_id=open_id,
                enabled=enabled,
            )
            self.accounts.append(account)
            logger.info(
                "MultiTikTok: Discovered account #%d '%s' (enabled=%s)",
                i, label, enabled,
            )

        if not self.accounts:
            logger.warning("MultiTikTok: No TikTok accounts configured")
        else:
            logger.info(
                "MultiTikTok: %d account(s) loaded", len(self.accounts)
            )

    def get_active_accounts(self) -> list[TikTokAccount]:
        """Return only enabled accounts."""
        return [a for a in self.accounts if a.enabled]

    def get_account_by_label(self, label: str) -> TikTokAccount | None:
        """Find account by label (case-insensitive)."""
        label_lower = label.lower()
        for a in self.accounts:
            if a.label.lower() == label_lower:
                return a
        return None

    def get_account_by_index(self, index: int) -> TikTokAccount | None:
        """Find account by index (1-based)."""
        for a in self.accounts:
            if a.index == index:
                return a
        return None

    # ════════════════════════════════════════════
    # Publishing — Video Posts
    # ════════════════════════════════════════════

    async def publish_video(
        self,
        account: TikTokAccount,
        video_url: str,
        caption: str = "",
        hashtags: list[str] | None = None,
        privacy: str = "PUBLIC_TO_EVERYONE",
    ) -> dict:
        """
        Post a video to a specific TikTok account via URL-pull.

        Args:
            account: Target TikTok account
            video_url: Public URL of the video file
            caption: Post caption (max 2200 chars)
            hashtags: List of hashtags to append
            privacy: PUBLIC_TO_EVERYONE, MUTUAL_FOLLOW_FRIENDS, FOLLOWER_OF_CREATOR, SELF_ONLY
        """
        if not account.access_token:
            return {
                "success": False,
                "error": f"Account '{account.label}' has no access token",
                "account": account.label,
            }

        full_caption = self._build_caption(caption, hashtags)

        payload = {
            "post_info": {
                "title": full_caption[:2200],
                "privacy_level": privacy,
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "PULL_FROM_URL",
                "video_url": video_url,
            },
        }

        return await self._post_to_tiktok(
            account, "/post/publish/video/init/", payload
        )

    # ════════════════════════════════════════════
    # Publishing — Photo Posts
    # ════════════════════════════════════════════

    async def publish_photo(
        self,
        account: TikTokAccount,
        image_urls: list[str],
        caption: str = "",
        hashtags: list[str] | None = None,
        privacy: str = "PUBLIC_TO_EVERYONE",
    ) -> dict:
        """
        Post photo(s) to a specific TikTok account.
        TikTok supports up to 35 images per photo post.

        Args:
            account: Target TikTok account
            image_urls: List of public image URLs (1-35)
            caption: Post caption
            hashtags: Hashtags to append
            privacy: Privacy level
        """
        if not account.access_token:
            return {
                "success": False,
                "error": f"Account '{account.label}' has no access token",
                "account": account.label,
            }

        if not image_urls:
            return {
                "success": False,
                "error": "At least one image URL is required",
                "account": account.label,
            }

        full_caption = self._build_caption(caption, hashtags)

        payload = {
            "post_info": {
                "title": full_caption[:2200],
                "privacy_level": privacy,
                "disable_comment": False,
            },
            "source_info": {
                "source": "PULL_FROM_URL",
                "photo_cover_index": 0,
                "photo_images": image_urls[:35],
            },
            "post_mode": "DIRECT_POST",
            "media_type": "PHOTO",
        }

        return await self._post_to_tiktok(
            account, "/post/publish/content/init/", payload
        )

    # ════════════════════════════════════════════
    # Posting Modes
    # ════════════════════════════════════════════

    async def publish_to_all(
        self,
        content: dict,
        mode: str = "photo",
    ) -> list[dict]:
        """
        Broadcast — Post to ALL active accounts.

        content: {
            "video_url": "..." or "image_urls": ["..."],
            "caption": "...",
            "hashtags": ["...", ...],
            "privacy": "PUBLIC_TO_EVERYONE"
        }
        """
        active = self.get_active_accounts()
        if not active:
            return [{"success": False, "error": "No active TikTok accounts"}]

        results = []
        for account in active:
            result = await self._publish_content(account, content, mode)
            results.append(result)
        return results

    async def publish_round_robin(
        self,
        content: dict,
        mode: str = "photo",
    ) -> dict:
        """Round-robin — Post to next account in rotation."""
        active = self.get_active_accounts()
        if not active:
            return {"success": False, "error": "No active TikTok accounts"}

        account = active[self._round_robin_index % len(active)]
        self._round_robin_index = (self._round_robin_index + 1) % len(active)

        return await self._publish_content(account, content, mode)

    async def publish_to_account(
        self,
        label_or_index: str | int,
        content: dict,
        mode: str = "photo",
    ) -> dict:
        """Post to a specific account by label or index."""
        account = None
        if isinstance(label_or_index, int):
            account = self.get_account_by_index(label_or_index)
        else:
            account = self.get_account_by_label(label_or_index)
            if not account:
                # Try parsing as integer
                try:
                    account = self.get_account_by_index(int(label_or_index))
                except ValueError:
                    pass

        if not account:
            return {
                "success": False,
                "error": f"TikTok account '{label_or_index}' not found",
            }

        return await self._publish_content(account, content, mode)

    async def _publish_content(
        self, account: TikTokAccount, content: dict, mode: str
    ) -> dict:
        """Dispatch to photo or video publish based on mode."""
        caption = content.get("caption", "")
        hashtags = content.get("hashtags", [])
        privacy = content.get("privacy", "PUBLIC_TO_EVERYONE")

        if mode == "video":
            video_url = content.get("video_url", "")
            if not video_url:
                return {
                    "success": False,
                    "error": "video_url required for video mode",
                    "account": account.label,
                }
            return await self.publish_video(
                account, video_url, caption, hashtags, privacy
            )
        else:
            # Photo mode (default for blog sharing)
            image_urls = content.get("image_urls", [])
            if not image_urls:
                # Single image fallback
                image_url = content.get("image_url", "")
                if image_url:
                    image_urls = [image_url]
            if not image_urls:
                return {
                    "success": False,
                    "error": "image_urls or image_url required for photo mode",
                    "account": account.label,
                }
            return await self.publish_photo(
                account, image_urls, caption, hashtags, privacy
            )

    # ════════════════════════════════════════════
    # Test Connection
    # ════════════════════════════════════════════

    async def test_account(self, account: TikTokAccount) -> dict:
        """Test connection for a specific account."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{TIKTOK_API_BASE}/user/info/",
                    headers={"Authorization": f"Bearer {account.access_token}"},
                    params={"fields": "open_id,display_name,avatar_url"},
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", {}).get("user", {})
                    return {
                        "success": True,
                        "account": account.label,
                        "display_name": data.get("display_name", ""),
                        "open_id": data.get("open_id", ""),
                    }
                return {
                    "success": False,
                    "account": account.label,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                }
        except Exception as e:
            return {
                "success": False,
                "account": account.label,
                "error": str(e),
            }

    async def test_all_accounts(self) -> list[dict]:
        """Test connection for all accounts."""
        results = []
        for account in self.accounts:
            result = await self.test_account(account)
            results.append(result)
        return results

    # ════════════════════════════════════════════
    # Internal — API Call
    # ════════════════════════════════════════════

    async def _post_to_tiktok(
        self, account: TikTokAccount, endpoint: str, payload: dict
    ) -> dict:
        """Execute TikTok API request with retry."""
        url = f"{TIKTOK_API_BASE}{endpoint}"
        headers = {
            "Authorization": f"Bearer {account.access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)

                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        publish_id = data.get("publish_id", "")
                        account.posts_count += 1
                        from datetime import datetime, timezone
                        account.last_post_at = datetime.now(timezone.utc).isoformat()

                        logger.info(
                            "MultiTikTok: Posted to '%s' — publish_id=%s",
                            account.label, publish_id,
                        )
                        return {
                            "success": True,
                            "publish_id": publish_id,
                            "account": account.label,
                            "account_index": account.index,
                            "note": "Content is processing — URL available after TikTok processes it",
                        }

                    elif resp.status_code == 429:
                        retry_after = int(
                            resp.headers.get("Retry-After", "5")
                        )
                        logger.warning(
                            "MultiTikTok: Rate limited on '%s', retry in %ds",
                            account.label, retry_after,
                        )
                        import asyncio
                        await asyncio.sleep(retry_after)
                        continue

                    else:
                        error_msg = resp.text[:300]
                        logger.error(
                            "MultiTikTok: Error on '%s': HTTP %d — %s",
                            account.label, resp.status_code, error_msg,
                        )
                        return {
                            "success": False,
                            "error": f"HTTP {resp.status_code}: {error_msg}",
                            "account": account.label,
                        }

            except Exception as e:
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "success": False,
                    "error": str(e),
                    "account": account.label,
                }

        return {
            "success": False,
            "error": "Max retries exceeded",
            "account": account.label,
        }

    # ════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════

    @staticmethod
    def _build_caption(caption: str, hashtags: list[str] | None) -> str:
        """Build caption with hashtags appended."""
        if hashtags:
            tags = " ".join(
                f"#{t.lstrip('#')}" for t in hashtags[:10] if t
            )
            caption = f"{caption}\n\n{tags}" if caption else tags
        return caption.strip()

    def status(self) -> dict:
        """Return status summary of all accounts."""
        return {
            "total_accounts": len(self.accounts),
            "active_accounts": len(self.get_active_accounts()),
            "round_robin_index": self._round_robin_index,
            "accounts": [
                {
                    "index": a.index,
                    "label": a.label,
                    "enabled": a.enabled,
                    "has_token": bool(a.access_token),
                    "posts_count": a.posts_count,
                    "last_post_at": a.last_post_at,
                }
                for a in self.accounts
            ],
        }
