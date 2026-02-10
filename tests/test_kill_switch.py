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
        # Record enough errors to trigger
        for _ in range(10):
            self.ks.record_metric("error_rate", 1.0)

        events = self.ks.check_all()
        triggered = [e for e in events if e["trigger"] == "error_rate"]
        assert len(triggered) > 0

    def test_pause_all_activates_kill_switch(self):
        # Record high flag rate to trigger pause_all
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
