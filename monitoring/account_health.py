"""
ViralOps Engine — Account Health Monitor
Track account health across platforms to prevent bans/flags.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.models import AccountHealth

logger = logging.getLogger("viralops.account_health")


class AccountHealthMonitor:
    """
    Monitor social media account health.
    Tracks warnings, flags, and shadow-ban indicators.
    
    Multi-account aware: tracks health per account_id, not just per platform.
    Key format: "platform:account_id" or just "platform" (backward compat).
    """

    def __init__(self):
        self._accounts: dict[str, AccountHealth] = {}

    def _make_key(self, platform: str, account_id: str | None = None) -> str:
        """Create lookup key: 'platform:account_id' or just 'platform'."""
        if account_id:
            return f"{platform}:{account_id}"
        return platform

    def update(
        self,
        platform: str,
        account_id: str | None = None,
        followers: int = 0,
        engagement_rate: float = 0.0,
        warning_count: int = 0,
        is_flagged: bool = False,
        is_shadow_banned: bool = False,
    ) -> AccountHealth:
        """Update account health for a platform (optionally per account)."""
        key = self._make_key(platform, account_id)
        health = AccountHealth(
            platform=platform,
            account_id=account_id or platform,
            followers=followers,
            engagement_rate=engagement_rate,
            warning_count=warning_count,
            is_flagged=is_flagged,
            is_shadow_banned=is_shadow_banned,
            last_checked=datetime.now(timezone.utc),
        )
        self._accounts[key] = health

        if is_flagged or is_shadow_banned:
            logger.warning(
                "Account health alert: %s (flagged=%s, shadow_banned=%s)",
                key, is_flagged, is_shadow_banned,
            )

        return health

    def get_health(self, platform: str, account_id: str | None = None) -> AccountHealth | None:
        key = self._make_key(platform, account_id)
        return self._accounts.get(key)

    def get_all(self) -> dict[str, AccountHealth]:
        return dict(self._accounts)

    def get_all_for_platform(self, platform: str) -> dict[str, AccountHealth]:
        """Get health for all accounts on a platform."""
        return {
            k: v for k, v in self._accounts.items()
            if k == platform or k.startswith(f"{platform}:")
        }

    def is_safe_to_post(self, platform: str, account_id: str | None = None) -> tuple[bool, str]:
        """Check if it's safe to post on a platform/account."""
        key = self._make_key(platform, account_id)
        health = self._accounts.get(key)
        if not health:
            return True, f"No health data for {key} — proceeding with caution"

        if health.is_shadow_banned:
            return False, f"{key}: Shadow banned — DO NOT POST"
        if health.is_flagged:
            return False, f"{key}: Account flagged — pause posting"
        if health.warning_count >= 3:
            return False, f"{key}: {health.warning_count} warnings — pause posting"

        return True, f"{key}: Healthy (engagement: {health.engagement_rate:.1%})"
