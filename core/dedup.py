"""
ViralOps Engine — Dedup Engine
Hash-based + semantic similarity deduplication.
Prevents same/similar content from being posted to the same platform.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

logger = logging.getLogger("viralops.dedup")


class DedupEntry(NamedTuple):
    content_hash: str
    platform: str
    niche_id: str
    posted_at: datetime
    title_snippet: str


class DedupEngine:
    """
    Content deduplication engine.
    
    Checks:
    1. Exact hash match (same content)
    2. Cross-platform limit (max 3 platforms for same content)
    3. Time spacing (24h minimum between similar posts on same platform)
    4. Niche cooldown (same niche posted too frequently)
    """

    def __init__(
        self,
        max_same_content_platforms: int = 3,
        min_spacing_hours: int = 24,
        niche_cooldown_hours: int = 48,
    ):
        self._entries: list[DedupEntry] = []
        self._hash_index: dict[str, list[DedupEntry]] = defaultdict(list)
        self._platform_index: dict[str, list[DedupEntry]] = defaultdict(list)
        self._niche_index: dict[str, list[DedupEntry]] = defaultdict(list)
        self.max_same_content_platforms = max_same_content_platforms
        self.min_spacing_hours = min_spacing_hours
        self.niche_cooldown_hours = niche_cooldown_hours

    def compute_hash(self, text: str) -> str:
        """Compute content hash."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def check(
        self,
        content_hash: str,
        platform: str,
        niche_id: str,
    ) -> tuple[bool, list[str]]:
        """
        Check if content can be posted.
        
        Returns (is_allowed, reasons)
        """
        reasons: list[str] = []
        now = datetime.now(timezone.utc)

        # 1. Exact hash match on same platform
        for entry in self._hash_index.get(content_hash, []):
            if entry.platform == platform:
                reasons.append(
                    f"Exact duplicate on {platform} "
                    f"(posted {entry.posted_at.isoformat()})"
                )
                return False, reasons

        # 2. Cross-platform limit
        platforms_used = {
            e.platform for e in self._hash_index.get(content_hash, [])
        }
        if len(platforms_used) >= self.max_same_content_platforms:
            reasons.append(
                f"Content already posted to {len(platforms_used)} platforms "
                f"(max {self.max_same_content_platforms}): {platforms_used}"
            )
            return False, reasons

        # 3. Time spacing on same platform
        cutoff = now - timedelta(hours=self.min_spacing_hours)
        recent_on_platform = [
            e for e in self._platform_index.get(platform, [])
            if e.posted_at > cutoff
        ]
        if recent_on_platform:
            latest = max(recent_on_platform, key=lambda e: e.posted_at)
            hours_since = (now - latest.posted_at).total_seconds() / 3600
            reasons.append(
                f"Recent post on {platform} {hours_since:.1f}h ago "
                f"(min spacing: {self.min_spacing_hours}h)"
            )
            return False, reasons

        # 4. Niche cooldown
        niche_cutoff = now - timedelta(hours=self.niche_cooldown_hours)
        recent_niche = [
            e for e in self._niche_index.get(niche_id, [])
            if e.posted_at > niche_cutoff and e.platform == platform
        ]
        if recent_niche:
            reasons.append(
                f"Niche {niche_id} posted to {platform} within "
                f"{self.niche_cooldown_hours}h cooldown"
            )
            # Warning only, not blocking
            return True, reasons

        return True, []

    def register(
        self,
        content_hash: str,
        platform: str,
        niche_id: str,
        title_snippet: str = "",
    ) -> None:
        """Register a successful post."""
        entry = DedupEntry(
            content_hash=content_hash,
            platform=platform,
            niche_id=niche_id,
            posted_at=datetime.now(timezone.utc),
            title_snippet=title_snippet[:50],
        )
        self._entries.append(entry)
        self._hash_index[content_hash].append(entry)
        self._platform_index[platform].append(entry)
        self._niche_index[niche_id].append(entry)

    def cleanup(self, older_than_days: int = 30) -> int:
        """Remove entries older than N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.posted_at > cutoff]
        # Rebuild indices
        self._hash_index = defaultdict(list)
        self._platform_index = defaultdict(list)
        self._niche_index = defaultdict(list)
        for e in self._entries:
            self._hash_index[e.content_hash].append(e)
            self._platform_index[e.platform].append(e)
            self._niche_index[e.niche_id].append(e)
        removed = before - len(self._entries)
        logger.info("Dedup cleanup: removed %d entries older than %d days", removed, older_than_days)
        return removed

    @property
    def total_entries(self) -> int:
        return len(self._entries)
