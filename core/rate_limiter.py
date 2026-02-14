"""
ViralOps Engine — Rate Limiter
Per-platform, per-account rate limiting with jitter.
Prevents API throttling and account flagging.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("viralops.rate_limiter")


class RateLimiter:
    """
    Token bucket rate limiter with per-platform AND per-account limits + jitter.
    
    Features:
    - Per-platform rate limits
    - Per-account daily caps (multi-account aware)
    - Random jitter to appear human-like
    - Sliding window tracking
    """

    # Default limits per 24h from platforms.yaml
    DEFAULT_LIMITS = {
        "tiktok": 3,
        "instagram_reels": 5,
        "instagram_feed": 5,
        "facebook_reels": 5,
        "youtube_shorts": 5,
        "pinterest": 10,
        "linkedin": 5,
        "twitter_x": 15,
    }

    # Global limits
    GLOBAL_MAX_PER_HOUR = 3
    GLOBAL_MAX_PER_24H = 20

    def __init__(
        self,
        platform_limits: dict[str, int] | None = None,
        jitter_range: tuple[int, int] = (30, 180),
    ):
        self._limits = platform_limits or self.DEFAULT_LIMITS
        self._jitter_min, self._jitter_max = jitter_range
        self._timestamps: dict[str, list[datetime]] = defaultdict(list)
        self._account_timestamps: dict[str, list[datetime]] = defaultdict(list)
        self._global_timestamps: list[datetime] = []

    def can_publish(
        self,
        platform: str,
        account_id: str | None = None,
        account_daily_limit: int | None = None,
    ) -> tuple[bool, str | None]:
        """
        Check if we can publish to this platform (optionally for a specific account).
        
        Args:
            platform: Target platform
            account_id: Specific account (multi-account mode)
            account_daily_limit: Per-account limit from accounts.yaml
        
        Returns (allowed, reason_if_blocked)
        """
        now = datetime.now(timezone.utc)

        # 1. Global hourly limit
        hour_ago = now - timedelta(hours=1)
        recent_global = [t for t in self._global_timestamps if t > hour_ago]
        if len(recent_global) >= self.GLOBAL_MAX_PER_HOUR:
            return False, f"Global hourly limit reached ({self.GLOBAL_MAX_PER_HOUR}/h)"

        # 2. Global daily limit
        day_ago = now - timedelta(hours=24)
        daily_global = [t for t in self._global_timestamps if t > day_ago]
        if len(daily_global) >= self.GLOBAL_MAX_PER_24H:
            return False, f"Global daily limit reached ({self.GLOBAL_MAX_PER_24H}/24h)"

        # 3. Per-platform limit (total across all accounts)
        limit = self._limits.get(platform, 5)
        platform_timestamps = [
            t for t in self._timestamps[platform] if t > day_ago
        ]
        if len(platform_timestamps) >= limit:
            return False, f"Platform {platform} limit reached ({limit}/24h)"

        # 4. Per-account limit (if multi-account mode)
        if account_id:
            acct_limit = account_daily_limit or limit
            acct_key = f"{platform}:{account_id}"
            acct_timestamps = [
                t for t in self._account_timestamps[acct_key] if t > day_ago
            ]
            if len(acct_timestamps) >= acct_limit:
                return False, f"Account {account_id} limit reached ({acct_limit}/24h)"

        return True, None

    def record_publish(self, platform: str, account_id: str | None = None) -> None:
        """Record a successful publish (platform + optional account)."""
        now = datetime.now(timezone.utc)
        self._timestamps[platform].append(now)
        self._global_timestamps.append(now)
        if account_id:
            acct_key = f"{platform}:{account_id}"
            self._account_timestamps[acct_key].append(now)

    def get_account_usage(self, platform: str, account_id: str) -> int:
        """Get number of posts in last 24h for a specific account."""
        day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
        acct_key = f"{platform}:{account_id}"
        return sum(1 for t in self._account_timestamps.get(acct_key, []) if t > day_ago)

    def get_jitter_delay(self) -> float:
        """Get random delay for human-like behavior."""
        return random.uniform(self._jitter_min, self._jitter_max)

    def wait_with_jitter(self) -> float:
        """Sleep for a random jitter duration. Returns seconds waited."""
        delay = self.get_jitter_delay()
        logger.info("Rate limiter: Waiting %.1fs (jitter)", delay)
        time.sleep(delay)
        return delay

    def get_stats(self) -> dict:
        """Get current rate limit stats."""
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(hours=24)
        hour_ago = now - timedelta(hours=1)

        stats = {
            "global_last_hour": len([t for t in self._global_timestamps if t > hour_ago]),
            "global_last_24h": len([t for t in self._global_timestamps if t > day_ago]),
            "platforms": {},
        }

        for platform, limit in self._limits.items():
            count = len([t for t in self._timestamps[platform] if t > day_ago])
            stats["platforms"][platform] = {
                "used": count,
                "limit": limit,
                "remaining": max(0, limit - count),
            }

        return stats

    def cleanup(self) -> None:
        """Remove old timestamps (>48h)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        self._global_timestamps = [t for t in self._global_timestamps if t > cutoff]
        for platform in self._timestamps:
            self._timestamps[platform] = [
                t for t in self._timestamps[platform] if t > cutoff
            ]
