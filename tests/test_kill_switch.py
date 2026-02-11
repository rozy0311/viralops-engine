"""Tests for Kill Switch / Circuit Breaker."""

import pytest
from core.kill_switch import KillSwitch, KillSwitchAction


class TestKillSwitch:

    def setup_method(self):
        self.ks = KillSwitch()

    def test_initial_state_inactive(self):
        assert not self.ks.is_active

    def test_no_metrics_no_trigger(self):
        events = self.ks.check_all()
        assert len(events) == 0

    def test_error_rate_trigger(self):
        # Record enough errors to trigger error_rate_daily (threshold 0.03)
        # If loaded from YAML, trigger name is "error_rate_daily"
        # If fallback, trigger name is "error_rate" (threshold 5.0)
        trigger_names = [t.name for t in self.ks._triggers]

        if "error_rate_daily" in trigger_names:
            # YAML-loaded config: threshold is 0.03
            for _ in range(10):
                self.ks.record_metric("error_rate_daily", 1.0)
            events = self.ks.check_all()
            triggered = [e for e in events if e["trigger"] == "error_rate_daily"]
        else:
            # Fallback config: threshold is 5.0
            for _ in range(10):
                self.ks.record_metric("error_rate", 1.0)
            events = self.ks.check_all()
            triggered = [e for e in events if e["trigger"] == "error_rate"]

        assert len(triggered) > 0

    def test_pause_all_activates_kill_switch(self):
        # Find a trigger with PAUSE_ALL action
        trigger_names = [t.name for t in self.ks._triggers]

        if "account_restriction" in trigger_names:
            # YAML-loaded: account_restriction threshold=1, action=STOP_PLATFORM
            # error_rate_daily threshold=0.03, action=STOP_ALL
            for _ in range(10):
                self.ks.record_metric("error_rate_daily", 1.0)
        else:
            # Fallback: flag_rate threshold=3.0, action=PAUSE_ALL
            for _ in range(10):
                self.ks.record_metric("flag_rate", 1.0)

        self.ks.check_all()
        assert self.ks.is_active

    def test_reset(self):
        self.ks._active = True
        assert self.ks.is_active
        self.ks.reset()
        assert not self.ks.is_active

    def test_status(self):
        status = self.ks.get_status()
        assert "active" in status
        assert "triggers_configured" in status
        assert status["triggers_configured"] > 0
