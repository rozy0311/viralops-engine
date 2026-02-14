"""
ViralOps Engine — Alerting
Send alerts via Telegram, email, or webhook on critical events.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger("viralops.alerting")


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class AlertManager:
    """
    Alert manager for ViralOps critical events.
    
    Channels: Telegram, Email, Webhook, Console (fallback)
    """

    def __init__(
        self,
        telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN",
        telegram_chat_id_env: str = "TELEGRAM_CHAT_ID",
        default_channel: AlertChannel = AlertChannel.CONSOLE,
    ):
        self._telegram_token_env = telegram_bot_token_env
        self._telegram_chat_id_env = telegram_chat_id_env
        self._default_channel = default_channel
        self._history: deque[dict] = deque(maxlen=10_000)

    async def send(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        channel: AlertChannel | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Send an alert."""
        ch = channel or self._default_channel

        alert = {
            "message": message,
            "level": level.value,
            "channel": ch.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._history.append(alert)

        if ch == AlertChannel.CONSOLE:
            self._console_alert(message, level)
            return True
        elif ch == AlertChannel.TELEGRAM:
            return await self._telegram_alert(message, level)
        elif ch == AlertChannel.WEBHOOK:
            return await self._webhook_alert(message, level, metadata)

        return False

    def _console_alert(self, message: str, level: AlertLevel) -> None:
        """Console fallback alert."""
        icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        icon = icons.get(level.value, "📢")
        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)
        log_fn("%s %s", icon, message)

    async def _telegram_alert(self, message: str, level: AlertLevel) -> bool:
        """Send alert via Telegram bot."""
        import os
        token = os.environ.get(self._telegram_token_env)
        chat_id = os.environ.get(self._telegram_chat_id_env)
        if not token or not chat_id:
            logger.warning("Telegram not configured, falling back to console")
            self._console_alert(message, level)
            return False

        icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        icon = icons.get(level.value, "📢")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"{icon} *ViralOps Alert*\n{message}",
            "parse_mode": "Markdown",
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            logger.info("Telegram alert sent: %s", message[:50])
            return True
        except Exception as e:
            logger.error("Telegram alert failed: %s", str(e))
            self._console_alert(message, level)
            return False

    async def _webhook_alert(
        self, message: str, level: AlertLevel, metadata: dict | None
    ) -> bool:
        """Send alert via webhook."""
        import os
        webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
        if not webhook_url:
            logger.warning("Webhook URL not configured, falling back to console")
            self._console_alert(message, level)
            return False

        payload = {
            "text": message,
            "level": level.value,
            "source": "viralops-engine",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(webhook_url, json=payload)
                resp.raise_for_status()
            logger.info("Webhook alert sent: %s", message[:50])
            return True
        except Exception as e:
            logger.error("Webhook alert failed: %s", str(e))
            self._console_alert(message, level)
            return False

    # ─── Quick alert methods ────────────────────

    async def kill_switch_activated(self, trigger: str, details: str) -> None:
        await self.send(
            f"🚨 KILL-SWITCH ACTIVATED\nTrigger: {trigger}\n{details}",
            level=AlertLevel.CRITICAL,
        )

    async def publish_failed(self, platform: str, error: str) -> None:
        await self.send(
            f"❌ Publish failed on {platform}: {error}",
            level=AlertLevel.WARNING,
        )

    async def budget_alert(self, remaining_pct: float) -> None:
        await self.send(
            f"💰 Budget alert: {remaining_pct:.0f}% remaining",
            level=AlertLevel.WARNING if remaining_pct > 5 else AlertLevel.CRITICAL,
        )
