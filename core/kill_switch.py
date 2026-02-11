"""
ViralOps Engine — Kill Switch / Circuit Breaker
Emergency stop for the entire system when thresholds are breached.

5 Trigger Types (from guardrails.yaml):
1. flag_rate     — Content flags per hour > threshold
2. ban_risk      — Account ban probability > threshold  
3. spend_rate    — Budget burn rate > threshold
4. error_rate    — Publishing errors per hour > threshold
5. engagement_drop — Engagement drops below threshold
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import yaml

logger = logging.getLogger("viralops.kill_switch")


class KillSwitchAction(Enum):
    PAUSE_ALL = "pause_all"
    PAUSE_PLATFORM = "pause_platform"
    ALERT_HUMAN = "alert_human"
    REDUCE_VOLUME = "reduce_volume"
    NONE = "none"


class KillSwitchTrigger:
    """A single trigger configuration."""

    def __init__(
        self,
        name: str,
        threshold: float,
        action: KillSwitchAction,
        window_hours: float = 1.0,
    ):
        self.name = name
        self.threshold = threshold
        self.action = action
        self.window_hours = window_hours


class KillSwitch:
    """
    Circuit breaker for the ViralOps Engine.
    
    Monitors metrics and triggers emergency actions when thresholds are breached.
    """

    def __init__(self):
        self._active = False
        self._triggers: list[KillSwitchTrigger] = []
        self._triggered_events: list[dict] = []
        self._metrics_buffer: dict[str, list[tuple[datetime, float]]] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load triggers from guardrails.yaml or use hardcoded defaults."""
        try:
            with open("config/guardrails.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            ks_cfg = cfg.get("kill_switch", {})

            # Try list-based format: kill_switch.thresholds: [...]
            thresholds = ks_cfg.get("thresholds", [])
            if thresholds:
                for t in thresholds:
                    self._triggers.append(KillSwitchTrigger(
                        name=t.get("trigger", "unknown"),
                        threshold=float(t.get("threshold", 0)),
                        action=KillSwitchAction(t.get("action", "alert_human")),
                        window_hours=float(t.get("window_hours", 1.0)),
                    ))
                return

            # Try dict-based format: kill_switch.error_rate_daily: {threshold: ...}
            action_map = {
                "STOP_ALL": KillSwitchAction.PAUSE_ALL,
                "STOP_PLATFORM": KillSwitchAction.PAUSE_PLATFORM,
                "REDUCE_FREQUENCY": KillSwitchAction.REDUCE_VOLUME,
                "STOP_REVIEW": KillSwitchAction.ALERT_HUMAN,
                "DOWNGRADE_MODEL": KillSwitchAction.ALERT_HUMAN,
            }
            for key, val in ks_cfg.items():
                if isinstance(val, dict) and "threshold" in val:
                    action_str = val.get("action", "STOP_ALL")
                    self._triggers.append(KillSwitchTrigger(
                        name=key,
                        threshold=float(val["threshold"]),
                        action=action_map.get(action_str, KillSwitchAction.ALERT_HUMAN),
                        window_hours=float(val.get("cooldown_hours", 1.0)),
                    ))

            if self._triggers:
                return
        except (FileNotFoundError, yaml.YAMLError):
            pass

        # Fallback defaults
        self._triggers = [
            KillSwitchTrigger("flag_rate", 3.0, KillSwitchAction.PAUSE_ALL),
            KillSwitchTrigger("ban_risk", 0.7, KillSwitchAction.PAUSE_ALL),
            KillSwitchTrigger("spend_rate", 2.0, KillSwitchAction.PAUSE_ALL),
            KillSwitchTrigger("error_rate", 5.0, KillSwitchAction.PAUSE_PLATFORM),
            KillSwitchTrigger("engagement_drop", 0.5, KillSwitchAction.ALERT_HUMAN),
        ]

    @property
    def is_active(self) -> bool:
        return self._active

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric data point."""
        if metric_name not in self._metrics_buffer:
            self._metrics_buffer[metric_name] = []
        self._metrics_buffer[metric_name].append((datetime.utcnow(), value))

    def check_all(self) -> list[dict]:
        """
        Check all triggers against current metrics.
        Returns list of triggered events (may be empty).
        """
        events = []
        now = datetime.utcnow()

        for trigger in self._triggers:
            buffer = self._metrics_buffer.get(trigger.name, [])
            if not buffer:
                continue

            # Get values within window
            cutoff = now - timedelta(hours=trigger.window_hours)
            window_values = [v for t, v in buffer if t > cutoff]

            if not window_values:
                continue

            # Check threshold
            current = (
                sum(window_values) / len(window_values)
                if trigger.name == "engagement_drop"
                else sum(window_values)
            )

            triggered = False
            if trigger.name == "engagement_drop":
                triggered = current < trigger.threshold
            else:
                triggered = current > trigger.threshold

            if triggered:
                event = {
                    "trigger": trigger.name,
                    "threshold": trigger.threshold,
                    "current_value": current,
                    "action": trigger.action.value,
                    "triggered_at": now.isoformat(),
                    "window_hours": trigger.window_hours,
                }
                events.append(event)
                self._triggered_events.append(event)

                logger.critical(
                    "KILL-SWITCH TRIGGERED: %s (current=%.2f, threshold=%.2f) → %s",
                    trigger.name, current, trigger.threshold, trigger.action.value,
                )

                if trigger.action == KillSwitchAction.PAUSE_ALL:
                    self._active = True

        return events

    def reset(self) -> None:
        """Reset kill switch (manual human action only)."""
        logger.warning("Kill switch manually reset")
        self._active = False

    def get_status(self) -> dict:
        """Get current kill switch status."""
        return {
            "active": self._active,
            "triggers_configured": len(self._triggers),
            "triggered_events": len(self._triggered_events),
            "recent_events": self._triggered_events[-5:] if self._triggered_events else [],
            "metrics_tracked": list(self._metrics_buffer.keys()),
        }
