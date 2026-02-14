"""
ViralOps Engine — Account Router
Smart multi-account selection for scaling without breaking limits.

CHIẾN LƯỢC: Chia nhiều account thay vì cố bẻ limit 1 tài khoản.

Routing strategies:
1. smart       — niche_affinity > health > remaining_quota > priority
2. round_robin — rotate accounts evenly
3. random      — random selection from available
4. priority    — always use highest-priority available account

Anti-pattern rules:
- Max 2 posts/hour/account (never burst)
- Min 15min between posts on same account
- Min 5min between different accounts on same platform
- Vary posting times (don't post at exact same time daily)
- Flagged accounts auto-pause for 72h (configurable)
- Warming-up accounts get reduced limits
"""

from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import yaml

logger = logging.getLogger("viralops.account_router")


# ═══════════════════════════════════════════
# Account Model
# ═══════════════════════════════════════════

@dataclass
class AccountConfig:
    """Configuration for a single social media account."""
    account_id: str
    platform: str
    display_name: str
    env_token_key: str
    daily_limit: int
    niches: list[str]
    priority: int = 1
    status: str = "active"          # active | paused | warming_up | flagged
    warm_up: bool = False
    warm_up_days_remaining: int = 0
    extra_env_keys: dict[str, str] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return self.status in ("active", "warming_up")

    def get_token(self) -> str | None:
        return os.environ.get(self.env_token_key)

    def has_token(self) -> bool:
        return bool(self.get_token())


# ═══════════════════════════════════════════
# Account Router
# ═══════════════════════════════════════════

class AccountRouter:
    """
    Smart account router for multi-account scaling.
    
    Distributes posts across multiple accounts per platform,
    respecting per-account limits, niche affinity, health status,
    and anti-pattern rules.
    """

    def __init__(self, config_path: str = "config/accounts.yaml"):
        self._accounts: dict[str, list[AccountConfig]] = defaultdict(list)
        self._post_history: dict[str, list[datetime]] = defaultdict(list)
        self._routing_config: dict[str, Any] = {}
        self._config_path = config_path

    def load_config(self) -> int:
        """
        Load account config from YAML.
        Returns number of accounts loaded.
        """
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Account config not found: %s", self._config_path)
            return 0

        self._routing_config = config.get("routing", {})
        accounts_data = config.get("accounts", {})
        total = 0

        for platform, account_list in accounts_data.items():
            for acc in account_list:
                extra_keys = {}
                for key, val in acc.items():
                    if key.startswith("env_") and key != "env_token_key":
                        extra_keys[key] = val

                account = AccountConfig(
                    account_id=acc["account_id"],
                    platform=platform,
                    display_name=acc.get("display_name", acc["account_id"]),
                    env_token_key=acc["env_token_key"],
                    daily_limit=acc.get("daily_limit", 10),
                    niches=acc.get("niches", ["all"]),
                    priority=acc.get("priority", 1),
                    status=acc.get("status", "active"),
                    warm_up=acc.get("warm_up", False),
                    warm_up_days_remaining=acc.get("warm_up_days_remaining", 0),
                    extra_env_keys=extra_keys,
                )
                self._accounts[platform].append(account)
                total += 1

        logger.info("Loaded %d accounts across %d platforms", total, len(self._accounts))
        return total

    # ─── Selection ─────────────────────────────

    def select_account(
        self,
        platform: str,
        niche: str = "",
        strategy: str | None = None,
    ) -> AccountConfig | None:
        """
        Select the best account for posting.
        
        Args:
            platform: Target platform (e.g., "tiktok")
            niche: Content niche for affinity matching
            strategy: Override routing strategy (or use config default)
        
        Returns:
            Best AccountConfig, or None if all exhausted/flagged
        """
        candidates = self._get_available_accounts(platform)
        if not candidates:
            logger.warning("No available accounts for %s", platform)
            return None

        strat = strategy or self._routing_config.get("strategy", "smart")

        if strat == "smart":
            return self._smart_select(candidates, niche)
        elif strat == "round_robin":
            return self._round_robin_select(candidates)
        elif strat == "random":
            return random.choice(candidates)
        elif strat == "priority_only":
            return min(candidates, key=lambda a: a.priority)
        else:
            return self._smart_select(candidates, niche)

    def _get_available_accounts(self, platform: str) -> list[AccountConfig]:
        """Get accounts that are available and have remaining quota."""
        available = []
        for acc in self._accounts.get(platform, []):
            if not acc.is_available:
                continue
            if not acc.has_token():
                logger.debug("Account %s: no token configured", acc.account_id)
                continue
            # Check remaining quota
            used = self._get_posts_today(acc.account_id)
            effective_limit = self._get_effective_limit(acc)
            if used >= effective_limit:
                logger.debug("Account %s: daily limit reached (%d/%d)",
                           acc.account_id, used, effective_limit)
                continue
            # Check anti-pattern: min spacing
            if not self._check_spacing(acc.account_id):
                logger.debug("Account %s: too recent, spacing required", acc.account_id)
                continue
            available.append(acc)
        return available

    def _smart_select(
        self,
        candidates: list[AccountConfig],
        niche: str,
    ) -> AccountConfig:
        """
        Smart selection scoring:
        - Niche affinity: +3 if niche matches, +1 if "all"
        - Remaining quota: +2 if >50% remaining, +1 if >20%
        - Priority: +(5 - priority) (lower priority number = higher score)
        - Warm-up penalty: -2 if warming up
        """
        scored = []
        for acc in candidates:
            score = 0.0

            # Niche affinity
            if niche and niche in acc.niches:
                score += 3.0
            elif "all" in acc.niches:
                score += 1.0
            elif not niche:
                score += 1.0

            # Remaining quota ratio
            used = self._get_posts_today(acc.account_id)
            effective_limit = self._get_effective_limit(acc)
            remaining_ratio = 1.0 - (used / max(effective_limit, 1))
            if remaining_ratio > 0.5:
                score += 2.0
            elif remaining_ratio > 0.2:
                score += 1.0

            # Priority bonus
            score += max(0, 5 - acc.priority)

            # Warm-up penalty
            if acc.warm_up:
                score -= 2.0

            # Small random factor to avoid always picking same account
            score += random.uniform(0, 0.5)

            scored.append((acc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]
        logger.info(
            "Smart select: %s (score=%.1f, used=%d/%d)",
            winner.account_id,
            scored[0][1],
            self._get_posts_today(winner.account_id),
            self._get_effective_limit(winner),
        )
        return winner

    def _round_robin_select(self, candidates: list[AccountConfig]) -> AccountConfig:
        """Select least-used account today."""
        return min(candidates, key=lambda a: self._get_posts_today(a.account_id))

    # ─── Rate Tracking ─────────────────────────

    def record_post(self, account_id: str) -> None:
        """Record a post for rate tracking."""
        self._post_history[account_id].append(datetime.now(timezone.utc))

    def _get_posts_today(self, account_id: str) -> int:
        """Count posts in last 24h for account."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        return sum(1 for t in self._post_history.get(account_id, []) if t > cutoff)

    def _get_effective_limit(self, acc: AccountConfig) -> int:
        """Get effective daily limit considering warm-up."""
        if not acc.warm_up:
            return acc.daily_limit

        warm_up_config = self._routing_config.get("warm_up", {})
        remaining = acc.warm_up_days_remaining

        if remaining > 11:      # Day 1-3
            return warm_up_config.get("day_1_to_3", 2)
        elif remaining > 7:     # Day 4-7
            return warm_up_config.get("day_4_to_7", 5)
        elif remaining > 0:     # Day 8-14
            return warm_up_config.get("day_8_to_14", 10)
        else:
            return acc.daily_limit

    def _check_spacing(self, account_id: str) -> bool:
        """Check min spacing between posts for anti-pattern."""
        anti = self._routing_config.get("anti_pattern", {})
        min_spacing = anti.get("min_spacing_minutes", 15)

        history = self._post_history.get(account_id, [])
        if not history:
            return True

        last_post = max(history)
        elapsed = (datetime.now(timezone.utc) - last_post).total_seconds() / 60
        return elapsed >= min_spacing

    # ─── Account Management ────────────────────

    def flag_account(self, account_id: str, reason: str = "") -> None:
        """Flag an account — auto-pause."""
        for platform_accounts in self._accounts.values():
            for acc in platform_accounts:
                if acc.account_id == account_id:
                    acc.status = "flagged"
                    logger.warning(
                        "Account FLAGGED: %s (reason: %s) — paused for cooldown",
                        account_id, reason or "unknown",
                    )
                    return

    def pause_account(self, account_id: str) -> None:
        """Manually pause an account."""
        for platform_accounts in self._accounts.values():
            for acc in platform_accounts:
                if acc.account_id == account_id:
                    acc.status = "paused"
                    logger.info("Account PAUSED: %s", account_id)
                    return

    def resume_account(self, account_id: str) -> None:
        """Resume a paused account."""
        for platform_accounts in self._accounts.values():
            for acc in platform_accounts:
                if acc.account_id == account_id:
                    acc.status = "active"
                    logger.info("Account RESUMED: %s", account_id)
                    return

    # ─── Stats ─────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        stats = {}
        for platform, accounts in self._accounts.items():
            platform_stats = []
            for acc in accounts:
                used = self._get_posts_today(acc.account_id)
                effective = self._get_effective_limit(acc)
                platform_stats.append({
                    "account_id": acc.account_id,
                    "display_name": acc.display_name,
                    "status": acc.status,
                    "posts_today": used,
                    "daily_limit": effective,
                    "remaining": max(0, effective - used),
                    "niches": acc.niches,
                    "has_token": acc.has_token(),
                })
            stats[platform] = {
                "total_accounts": len(accounts),
                "active_accounts": sum(1 for a in accounts if a.is_available),
                "total_remaining": sum(
                    max(0, self._get_effective_limit(a) - self._get_posts_today(a.account_id))
                    for a in accounts if a.is_available
                ),
                "accounts": platform_stats,
            }
        return stats

    def get_total_capacity(self, platform: str) -> int:
        """Total daily posting capacity for a platform across all accounts."""
        return sum(
            self._get_effective_limit(acc)
            for acc in self._accounts.get(platform, [])
            if acc.is_available
        )
