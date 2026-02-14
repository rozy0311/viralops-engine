"""
ViralOps Engine — Engagement Tracker
Track post engagement metrics for optimization.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from core.models import EngagementMetrics

logger = logging.getLogger("viralops.engagement")


class EngagementTracker:
    """Track and analyze engagement across platforms."""

    MAX_PER_PLATFORM = 5_000  # Cap per-platform metrics to prevent OOM

    def __init__(self):
        self._metrics: dict[str, list[EngagementMetrics]] = defaultdict(list)

    def record(
        self,
        platform: str,
        post_id: str,
        views: int = 0,
        likes: int = 0,
        comments: int = 0,
        shares: int = 0,
        saves: int = 0,
        hours_since_post: float = 1.0,
    ) -> EngagementMetrics:
        """Record engagement metrics for a post."""
        metrics = EngagementMetrics(
            post_id=post_id,
            platform=platform,
            views=views,
            likes=likes,
            comments=comments,
            shares=shares,
            views_per_hour=views / max(hours_since_post, 0.1),
            measured_at=datetime.now(timezone.utc),
        )
        self._metrics[platform].append(metrics)
        # Trim old entries to prevent OOM
        if len(self._metrics[platform]) > self.MAX_PER_PLATFORM:
            self._metrics[platform] = self._metrics[platform][-self.MAX_PER_PLATFORM:]
        return metrics

    def get_platform_avg(self, platform: str) -> dict[str, float]:
        """Get average metrics for a platform."""
        entries = self._metrics.get(platform, [])
        if not entries:
            return {"avg_views": 0, "avg_engagement": 0, "avg_vph": 0}

        return {
            "avg_views": sum(e.views for e in entries) / len(entries),
            "avg_engagement": sum(e.engagement_rate for e in entries) / len(entries),
            "avg_vph": sum(e.views_per_hour for e in entries) / len(entries),
        }

    def get_top_performers(self, limit: int = 5) -> list[EngagementMetrics]:
        """Get top performing posts across all platforms."""
        all_metrics = []
        for entries in self._metrics.values():
            all_metrics.extend(entries)
        all_metrics.sort(key=lambda m: m.engagement_rate, reverse=True)
        return all_metrics[:limit]
