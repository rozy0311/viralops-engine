"""
ViralOps Engine — Monitoring Dashboard
Aggregates metrics, account health, and system status.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from core.kill_switch import KillSwitch
from core.rate_limiter import RateLimiter

logger = logging.getLogger("viralops.dashboard")


class Dashboard:
    """
    Monitoring dashboard for the ViralOps Engine.
    Aggregates all system metrics into a single view.
    """

    def __init__(
        self,
        kill_switch: KillSwitch | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self._kill_switch = kill_switch or KillSwitch()
        self._rate_limiter = rate_limiter or RateLimiter()
        self._publish_history: deque[dict] = deque(maxlen=10_000)
        self._error_history: deque[dict] = deque(maxlen=5_000)

    def record_publish(
        self,
        platform: str,
        post_url: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        entry = {
            "platform": platform,
            "post_url": post_url,
            "success": success,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._publish_history.append(entry)
        if not success and error:
            self._error_history.append(entry)
            self._kill_switch.record_metric("error_rate", 1.0)

    def get_summary(self) -> dict[str, Any]:
        """Get full dashboard summary."""
        total = len(self._publish_history)
        success = sum(1 for p in self._publish_history if p["success"])
        failed = total - success

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "publish_stats": {
                "total": total,
                "success": success,
                "failed": failed,
                "success_rate": f"{success / max(total, 1):.0%}",
            },
            "rate_limits": self._rate_limiter.get_stats(),
            "kill_switch": self._kill_switch.get_status(),
            "recent_errors": self._error_history[-5:],
            "platform_breakdown": self._platform_breakdown(),
        }

    def _platform_breakdown(self) -> dict[str, dict]:
        """Per-platform publish stats."""
        breakdown: dict[str, dict] = {}
        for entry in self._publish_history:
            p = entry["platform"]
            if p not in breakdown:
                breakdown[p] = {"total": 0, "success": 0, "failed": 0}
            breakdown[p]["total"] += 1
            if entry["success"]:
                breakdown[p]["success"] += 1
            else:
                breakdown[p]["failed"] += 1
        return breakdown

    def print_summary(self) -> str:
        """Human-readable summary."""
        s = self.get_summary()
        lines = [
            "═══ ViralOps Dashboard ═══",
            f"🕐 {s['timestamp']}",
            f"📊 Total: {s['publish_stats']['total']} | "
            f"✅ {s['publish_stats']['success']} | "
            f"❌ {s['publish_stats']['failed']} | "
            f"Rate: {s['publish_stats']['success_rate']}",
            "",
        ]
        ks = s["kill_switch"]
        if ks["active"]:
            lines.append("🚨 KILL-SWITCH ACTIVE")
        else:
            lines.append("✅ Kill-switch: OK")

        for p, stats in s["platform_breakdown"].items():
            lines.append(f"  {p}: {stats['success']}/{stats['total']}")

        return "\n".join(lines)
