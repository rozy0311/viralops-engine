"""
Telegram Alert Bot — Send notifications when RSS Auto Poster publishes.

Features:
  - Publish alerts (post created, post published)
  - Error alerts (feed unreachable, API failure)
  - Status reports (daily summary, poster status)
  - Rich formatting with Markdown + emoji
  - Rate-limited to prevent spam (max 30 msgs/min)

Config via env vars:
  TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
  TELEGRAM_CHAT_ID=-1001234567890
"""

import os
import time
from datetime import datetime
from typing import Optional
from collections import deque

import httpx
import structlog

logger = structlog.get_logger()

# ── Config ──
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"

# Rate limiting: max 30 messages per minute
MAX_MESSAGES_PER_MINUTE = 30
_message_timestamps: deque = deque(maxlen=MAX_MESSAGES_PER_MINUTE)


def is_configured() -> bool:
    """Check if Telegram bot is configured."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _check_rate_limit() -> bool:
    """Check if we can send another message (rate limit: 30/min)."""
    now = time.time()
    # Remove timestamps older than 60 seconds
    while _message_timestamps and _message_timestamps[0] < now - 60:
        _message_timestamps.popleft()
    return len(_message_timestamps) < MAX_MESSAGES_PER_MINUTE


def send_message(
    text: str,
    chat_id: str = None,
    parse_mode: str = "Markdown",
    disable_preview: bool = True,
) -> dict:
    """
    Send a message to Telegram.

    Args:
        text: Message text (Markdown or HTML)
        chat_id: Override default chat ID
        parse_mode: "Markdown" or "HTML"
        disable_preview: Disable link previews

    Returns: {"success": bool, "message_id": int}
    """
    if not is_configured():
        logger.debug("telegram.not_configured")
        return {"success": False, "error": "Telegram bot not configured"}

    if not _check_rate_limit():
        logger.warning("telegram.rate_limited")
        return {"success": False, "error": "Rate limited (30 msgs/min)"}

    target_chat = chat_id or TELEGRAM_CHAT_ID
    url = f"{TELEGRAM_API_BASE.format(token=TELEGRAM_BOT_TOKEN)}/sendMessage"

    try:
        with httpx.Client(timeout=15) as client:
            response = client.post(url, json={
                "chat_id": target_chat,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_preview,
            })

        data = response.json()
        if data.get("ok"):
            _message_timestamps.append(time.time())
            msg_id = data.get("result", {}).get("message_id")
            logger.info("telegram.sent", message_id=msg_id, length=len(text))
            return {"success": True, "message_id": msg_id}
        else:
            error = data.get("description", "Unknown error")
            logger.error("telegram.api_error", error=error)
            return {"success": False, "error": error}

    except Exception as e:
        logger.error("telegram.send_error", error=str(e))
        return {"success": False, "error": str(e)}


# ════════════════════════════════════════════════
# Pre-built Alert Templates
# ════════════════════════════════════════════════

def alert_post_created(
    poster_name: str,
    title: str,
    platforms: list,
    media_type: str = "none",
    has_music: bool = False,
    post_count: int = 1,
) -> dict:
    """Send alert when RSS Auto Poster creates new posts."""
    platform_str = ", ".join(f"`{p}`" for p in platforms)
    music_icon = "🎵" if has_music else ""
    media_icon = {"image": "🖼️", "video": "🎬", "image_with_music": "🖼️🎵", "none": "📝"}.get(media_type, "📎")

    text = (
        f"📡 *RSS Auto Post Created*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📋 *Poster:* {poster_name}\n"
        f"📰 *Title:* {title[:80]}\n"
        f"{media_icon} *Media:* {media_type} {music_icon}\n"
        f"📱 *Platforms:* {platform_str}\n"
        f"📊 *Posts created:* {post_count}\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return send_message(text)


def alert_tick_summary(
    posters_checked: int,
    total_posted: int,
    results: list = None,
) -> dict:
    """Send summary after RSS Auto Poster tick cycle."""
    if total_posted == 0:
        return {"success": True, "skipped": True, "reason": "nothing_posted"}

    details = ""
    if results:
        for r in results:
            if r.get("posted", 0) > 0:
                details += f"  • `{r.get('name', r.get('poster_id', '?'))}`: {r['posted']} posts\n"

    text = (
        f"✅ *RSS Auto Poster Tick*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🔄 Posters checked: {posters_checked}\n"
        f"📤 Total posted: *{total_posted}*\n"
    )
    if details:
        text += f"\n📊 *Details:*\n{details}"
    text += f"\n🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

    return send_message(text)


def alert_error(
    source: str,
    error: str,
    context: dict = None,
) -> dict:
    """Send error alert."""
    ctx_str = ""
    if context:
        for k, v in context.items():
            ctx_str += f"  • `{k}`: {v}\n"

    text = (
        f"🚨 *Error Alert*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📍 *Source:* {source}\n"
        f"❌ *Error:* {error[:200]}\n"
    )
    if ctx_str:
        text += f"\n📋 *Context:*\n{ctx_str}"
    text += f"\n🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

    return send_message(text)


def alert_poster_state_change(
    poster_name: str,
    poster_id: str,
    new_state: str,
) -> dict:
    """Send alert when a poster is paused/activated."""
    icon = "▶️" if new_state == "active" else "⏸️"
    text = (
        f"{icon} *Poster {new_state.title()}*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📋 *Poster:* {poster_name}\n"
        f"🆔 `{poster_id}`\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return send_message(text)


def alert_daily_summary(
    total_posters: int,
    active_posters: int,
    total_posted_today: int,
    top_poster: str = None,
    errors_today: int = 0,
) -> dict:
    """Send daily summary report."""
    text = (
        f"📊 *Daily Summary — ViralOps Engine*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 Total Posters: {total_posters} ({active_posters} active)\n"
        f"📤 Posts Created Today: *{total_posted_today}*\n"
    )
    if top_poster:
        text += f"🏆 Top Poster: {top_poster}\n"
    if errors_today > 0:
        text += f"⚠️ Errors Today: {errors_today}\n"
    text += (
        f"\n🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"_Powered by ViralOps Engine v2.3.0_"
    )
    return send_message(text)


def send_custom(text: str) -> dict:
    """Send a custom message (for testing)."""
    return send_message(text)


# ════════════════════════════════════════════════
# Bot Info
# ════════════════════════════════════════════════

def get_bot_info() -> dict:
    """Get Telegram bot info (for verifying setup)."""
    if not TELEGRAM_BOT_TOKEN:
        return {"configured": False}

    url = f"{TELEGRAM_API_BASE.format(token=TELEGRAM_BOT_TOKEN)}/getMe"
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url)
        data = response.json()
        if data.get("ok"):
            bot = data["result"]
            return {
                "configured": True,
                "bot_username": bot.get("username"),
                "bot_name": bot.get("first_name"),
                "chat_id": TELEGRAM_CHAT_ID,
            }
    except Exception as e:
        return {"configured": True, "error": str(e)}
    return {"configured": True, "error": "Failed to get bot info"}
