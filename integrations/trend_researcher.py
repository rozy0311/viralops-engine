"""
ViralOps Engine — Trend Research Client
Built-in trend research using public APIs and scraping.

Replaces external Gumloop/TikTok MCP dependency.
ViralOps handles its own trend research directly.

Sources:
- Google Trends (pytrends)
- Platform hashtag search (via official APIs)
- Internal niche performance history
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("viralops.trends")


class TrendResearcher:
    """
    Built-in trend research engine.
    No external dependencies — ViralOps does its own research.
    
    Sources:
    1. Internal niche history (from state/DB)
    2. Google Trends API (pytrends)
    3. Platform hashtag APIs (official OAuth)
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._history: list[dict] = []

    async def get_trending_hashtags(
        self,
        niche_keywords: list[str],
        platform: str = "tiktok",
        limit: int = 20,
    ) -> list[dict]:
        """
        Get trending hashtags for a niche.
        
        Returns: [{"tag": "#PlantBased", "volume": 1000000, "trend": "rising"}]
        """
        cache_key = f"hashtags:{platform}:{','.join(niche_keywords)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = []

        # Source 1: Google Trends (if pytrends available)
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq()
            pytrends.build_payload(niche_keywords[:5], timeframe="now 7-d")
            related = pytrends.related_queries()

            for kw in niche_keywords:
                if kw in related and related[kw].get("rising") is not None:
                    for _, row in related[kw]["rising"].iterrows():
                        results.append({
                            "tag": f"#{row['query'].replace(' ', '')}",
                            "volume": int(row.get("value", 0)),
                            "trend": "rising",
                            "source": "google_trends",
                        })
        except ImportError:
            logger.debug("pytrends not installed — skipping Google Trends")
        except Exception as e:
            logger.warning("Google Trends error: %s", e)

        # Source 2: Fallback to niche config hashtags
        if not results:
            import yaml
            try:
                with open("config/niches.yaml", "r", encoding="utf-8") as f:
                    niches = yaml.safe_load(f)
                for cat in niches.get("niches", {}).values():
                    for sub in cat.get("sub_niches", []):
                        layers = sub.get("hashtags_layer", {})
                        for tag in layers.get("trending", []):
                            results.append({
                                "tag": tag,
                                "volume": sub.get("search_volume", 0),
                                "trend": "stable",
                                "source": "niche_config",
                            })
            except FileNotFoundError:
                pass

        results = results[:limit]
        self._cache[cache_key] = results
        return results

    async def analyze_niche_performance(
        self,
        niche_id: str,
        past_metrics: list[dict] | None = None,
    ) -> dict:
        """
        Analyze niche performance from internal history.
        """
        if not past_metrics:
            return {
                "niche_id": niche_id,
                "data_available": False,
                "recommendation": "No history — start with pilot post",
            }

        total_views = sum(m.get("views", 0) for m in past_metrics)
        total_engagement = sum(m.get("engagement", 0) for m in past_metrics)
        count = len(past_metrics)

        return {
            "niche_id": niche_id,
            "data_available": True,
            "posts_count": count,
            "avg_views": total_views // max(count, 1),
            "avg_engagement": total_engagement / max(count, 1),
            "trend": "growing" if count > 3 else "insufficient_data",
            "recommendation": (
                "Continue posting — good engagement"
                if total_engagement / max(count, 1) > 0.03
                else "Consider rotating niche or changing approach"
            ),
        }

    def clear_cache(self) -> None:
        """Clear trend cache."""
        self._cache.clear()
