"""
ViralOps Engine -- Lemon8 Publisher (Mobile-Only Platform)

Lemon8 (by ByteDance) is MOBILE-ONLY -- NO desktop posting, NO public API.
Content can ONLY be published through the Lemon8 mobile app.

Strategy: Prepare content + push to phone for manual posting
  1. Format content into mobile-ready draft (title, body, images, tags)
  2. Save draft JSON locally (staging dir)
  3. Send formatted content to Telegram bot (user's phone)
     - Title, body text, hashtags ready to copy-paste
     - Image URLs / downloaded images ready to save
     - Step-by-step posting instructions
  4. Track status: pending -> posted (user confirms via Telegram command)

This is the ONLY honest approach for Lemon8. Any "API" claim is fake.

SETUP:
1. LEMON8_STAGING_DIR -- local folder for draft JSONs
2. LEMON8_TELEGRAM_BOT_TOKEN -- Telegram bot that sends to your phone
3. LEMON8_TELEGRAM_CHAT_ID -- your Telegram user/group chat ID
   (Alternative: LEMON8_WEBHOOK_URL for n8n/Slack/other webhook)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from core.models import PublishResult, QueueItem

logger = logging.getLogger("viralops.publisher.lemon8")


class Lemon8Publisher:
    """
    Lemon8 mobile-only publisher.

    Since Lemon8 has NO API and NO desktop posting, this publisher:
    - Formats content into copy-paste-ready blocks
    - Sends to user's phone via Telegram bot
    - Saves local draft for tracking
    - Tracks pending/posted status

    This is human-in-the-loop by design -- not a limitation, it's the
    ONLY way to post on Lemon8.
    """

    platform = "lemon8"

    # Lemon8 content limits (as of 2025)
    MAX_TITLE_CHARS = 55
    MAX_BODY_CHARS = 2200
    MAX_HASHTAGS = 10
    MAX_IMAGES = 20
    SUPPORTED_CATEGORIES = [
        "Food", "Beauty", "Fashion", "Travel", "Lifestyle",
        "Wellness", "Home", "DIY", "Pets", "Entertainment",
    ]

    def __init__(self, account_id: str = "lemon8_main"):
        self.account_id = account_id
        self._staging_dir: Path | None = None
        self._telegram_bot_token: str | None = None
        self._telegram_chat_id: str | None = None
        self._webhook_url: str | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Initialize Lemon8 publisher -- set up staging + notification channel."""
        prefix = self.account_id.upper().replace("-", "_")

        staging = os.environ.get(
            f"{prefix}_STAGING_DIR",
            os.environ.get("LEMON8_STAGING_DIR", "./lemon8_drafts"),
        )
        self._staging_dir = Path(staging)

        # Telegram bot (preferred -- sends directly to phone)
        self._telegram_bot_token = os.environ.get(
            f"{prefix}_TELEGRAM_BOT_TOKEN",
            os.environ.get("LEMON8_TELEGRAM_BOT_TOKEN"),
        )
        self._telegram_chat_id = os.environ.get(
            f"{prefix}_TELEGRAM_CHAT_ID",
            os.environ.get("LEMON8_TELEGRAM_CHAT_ID"),
        )

        # Fallback webhook (n8n / Slack / etc.)
        self._webhook_url = os.environ.get(
            f"{prefix}_WEBHOOK_URL",
            os.environ.get("LEMON8_WEBHOOK_URL"),
        )

        # Ensure staging directory exists
        self._staging_dir.mkdir(parents=True, exist_ok=True)

        has_telegram = bool(self._telegram_bot_token and self._telegram_chat_id)
        has_webhook = bool(self._webhook_url)

        if not has_telegram and not has_webhook:
            logger.warning(
                "Lemon8 [%s]: No notification channel configured. "
                "Drafts will be saved locally only. Set LEMON8_TELEGRAM_BOT_TOKEN "
                "+ LEMON8_TELEGRAM_CHAT_ID for phone notifications.",
                self.account_id,
            )

        notify_via = "Telegram" if has_telegram else ("webhook" if has_webhook else "local-only")
        logger.info(
            "Lemon8 [%s]: Ready (mobile-only workflow, notify via: %s, staging: %s)",
            self.account_id,
            notify_via,
            self._staging_dir,
        )
        self._connected = True
        return True

    async def publish(self, item: QueueItem, content: dict) -> PublishResult:
        """
        Prepare Lemon8 content and push to phone for manual posting.

        content keys:
            title (str): Post title (max 55 chars, required)
            body (str): Post body/description (max 2200 chars)
            caption (str): Fallback for body
            media_urls (list[str]): Image URLs (Lemon8 is image-first, max 20)
            media_url (str): Single image URL fallback
            tags (list[str]): Hashtags (max 10)
            category (str): Lemon8 category (Food, Beauty, Fashion, etc.)
            cover_index (int): Which image to use as cover (0-based)
        """
        if not self._connected:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Lemon8 publisher not initialized",
            )

        # --- Validate & format content ---
        title = content.get("title", content.get("caption", ""))[:self.MAX_TITLE_CHARS]
        body = content.get("body", content.get("text", content.get("caption", "")))[:self.MAX_BODY_CHARS]
        media_urls = content.get("media_urls", [])
        if not media_urls and content.get("media_url"):
            media_urls = [content["media_url"]]
        media_urls = media_urls[:self.MAX_IMAGES]
        tags = content.get("tags", [])[:self.MAX_HASHTAGS]
        category = content.get("category", "Lifestyle")

        if not title:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Lemon8 requires a title",
            )

        if not media_urls:
            logger.warning(
                "Lemon8 [%s]: No images provided. Lemon8 is image-first, "
                "posts without images get very low reach.",
                self.account_id,
            )

        # --- Create draft ---
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        draft_id = f"lemon8_{timestamp}_{item.id[:8] if item.id else 'manual'}"

        draft_data = {
            "draft_id": draft_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending_manual",
            "queue_item_id": item.id,
            "platform_note": "MOBILE-ONLY: Must post from Lemon8 app on phone",
            "content": {
                "title": title,
                "body": body,
                "media_urls": media_urls,
                "tags": tags,
                "category": category,
                "cover_index": content.get("cover_index", 0),
            },
        }

        # Save draft file
        draft_path = self._staging_dir / f"{draft_id}.json"
        try:
            draft_path.write_text(
                json.dumps(draft_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Lemon8 [%s]: Draft saved -> %s", self.account_id, draft_path)
        except Exception as e:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=f"Failed to save draft: {e}",
            )

        # --- Send to phone ---
        notified_via = "none"

        # Priority 1: Telegram bot (sends directly to phone)
        if self._telegram_bot_token and self._telegram_chat_id:
            tg_ok = await self._send_telegram(title, body, media_urls, tags, category)
            if tg_ok:
                notified_via = "telegram"

        # Priority 2: Generic webhook (n8n / Slack)
        if notified_via == "none" and self._webhook_url:
            wh_ok = await self._send_webhook(draft_data)
            if wh_ok:
                notified_via = "webhook"

        if notified_via == "none":
            logger.warning(
                "Lemon8 [%s]: Draft saved but NO notification sent. "
                "Check staging dir manually: %s",
                self.account_id,
                draft_path,
            )

        return PublishResult(
            queue_item_id=item.id,
            platform=self.platform,
            success=True,
            published_at=datetime.utcnow(),
            post_url=str(draft_path),
            post_id=draft_id,
            metadata={
                "mode": "mobile_manual",
                "draft_path": str(draft_path),
                "notified_via": notified_via,
                "requires_manual_posting": True,
                "posting_instructions": (
                    "1. Open Lemon8 app on phone\n"
                    "2. Tap '+' to create new post\n"
                    "3. Add images from the URLs/saved photos\n"
                    "4. Paste title and body text\n"
                    "5. Add hashtags\n"
                    "6. Select category and publish"
                ),
            },
        )

    async def _send_telegram(
        self,
        title: str,
        body: str,
        media_urls: list[str],
        tags: list[str],
        category: str,
    ) -> bool:
        """
        Send copy-paste-ready content to Telegram.
        Formats as a clean message the user can directly copy from phone.
        """
        # Build hashtag string
        hashtags = " ".join(f"#{t.strip('#')}" for t in tags) if tags else ""

        # Build the message -- formatted for easy copy-paste on mobile
        msg_parts = [
            " *LEMON8 POST READY*",
            "",
            f" Category: {category}",
            "",
            " TITLE (copy this) ",
            f"`{title}`",
            "",
            " BODY (copy this) ",
            f"`{body}`",
            "",
        ]

        if hashtags:
            msg_parts.extend([
                " HASHTAGS (copy this) ",
                f"`{hashtags}`",
                "",
            ])

        if media_urls:
            msg_parts.extend([
                f" Images ({len(media_urls)}):",
            ])
            for i, url in enumerate(media_urls, 1):
                msg_parts.append(f"  {i}. {url}")
            msg_parts.append("")

        msg_parts.extend([
            " Steps:",
            "1. Open Lemon8 app",
            "2. Tap + to create post",
            "3. Save images above to phone, then add them",
            "4. Long-press to paste title & body",
            "5. Add hashtags",
            "6. Publish!",
        ])

        message = "\n".join(msg_parts)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Send text message
                resp = await client.post(
                    f"https://api.telegram.org/bot{self._telegram_bot_token}/sendMessage",
                    json={
                        "chat_id": self._telegram_chat_id,
                        "text": message,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True,
                    },
                )
                resp.raise_for_status()

                # Send first image as photo (so user can save it directly)
                if media_urls:
                    try:
                        await client.post(
                            f"https://api.telegram.org/bot{self._telegram_bot_token}/sendPhoto",
                            json={
                                "chat_id": self._telegram_chat_id,
                                "photo": media_urls[0],
                                "caption": f"Cover image for: {title}",
                            },
                        )
                    except Exception:
                        pass  # Image send is best-effort

            logger.info(
                "Lemon8 [%s]: Content sent to Telegram (chat: %s)",
                self.account_id,
                self._telegram_chat_id,
            )
            return True

        except Exception as e:
            logger.warning(
                "Lemon8 [%s]: Telegram send failed: %s", self.account_id, e
            )
            return False

    async def _send_webhook(self, draft_data: dict) -> bool:
        """Send draft content to generic webhook (n8n / Slack / etc.)."""
        if not self._webhook_url:
            return False

        payload = {
            "source": "ViralOps",
            "platform": "lemon8",
            "action": "mobile_post_needed",
            "draft_id": draft_data["draft_id"],
            "note": "MOBILE-ONLY: Must post from Lemon8 app on phone",
            "title": draft_data["content"]["title"],
            "body": draft_data["content"]["body"],
            "media_urls": draft_data["content"]["media_urls"],
            "tags": draft_data["content"]["tags"],
            "category": draft_data["content"]["category"],
            "created_at": draft_data["created_at"],
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._webhook_url, json=payload)
                resp.raise_for_status()
            logger.info(
                "Lemon8 [%s]: Webhook notification sent to %s",
                self.account_id,
                self._webhook_url,
            )
            return True
        except Exception as e:
            logger.warning(
                "Lemon8 [%s]: Webhook failed: %s", self.account_id, e
            )
            return False

    async def mark_as_posted(
        self, draft_id: str, lemon8_url: str = ""
    ) -> bool:
        """
        Mark a draft as manually posted.
        Call this after user confirms they posted it from the Lemon8 app.
        Can be triggered via Telegram bot command or API endpoint.
        """
        if not self._staging_dir:
            return False

        draft_path = self._staging_dir / f"{draft_id}.json"
        if not draft_path.exists():
            logger.warning(
                "Lemon8 [%s]: Draft not found: %s", self.account_id, draft_id
            )
            return False

        try:
            data = json.loads(draft_path.read_text(encoding="utf-8"))
            data["status"] = "posted"
            data["posted_at"] = datetime.utcnow().isoformat()
            if lemon8_url:
                data["lemon8_url"] = lemon8_url
            draft_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "Lemon8 [%s]: Draft %s marked as posted", self.account_id, draft_id
            )
            return True
        except Exception as e:
            logger.error(
                "Lemon8 [%s]: Failed to update draft: %s", self.account_id, e
            )
            return False

    async def list_pending_drafts(self) -> list[dict]:
        """List all drafts still waiting to be posted from phone."""
        if not self._staging_dir:
            return []

        pending = []
        for f in sorted(self._staging_dir.glob("lemon8_*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("status") == "pending_manual":
                    pending.append(data)
            except Exception:
                continue
        return pending

    async def get_stats(self) -> dict:
        """Get draft statistics."""
        if not self._staging_dir:
            return {"total": 0, "pending": 0, "posted": 0}

        total = pending = posted = 0
        for f in self._staging_dir.glob("lemon8_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                total += 1
                status = data.get("status", "")
                if status == "pending_manual":
                    pending += 1
                elif status == "posted":
                    posted += 1
            except Exception:
                continue

        return {"total": total, "pending": pending, "posted": posted}

    async def close(self) -> None:
        """No persistent connection to close (mobile-only platform)."""
        self._connected = False