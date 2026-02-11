"""Tests for Rate Limiter."""

import pytest
from core.rate_limiter import RateLimiter


class TestRateLimiter:

    def setup_method(self):
        self.limiter = RateLimiter(jitter_range=(0, 0))  # No jitter for tests

    def test_first_publish_allowed(self):
        allowed, reason = self.limiter.can_publish("tiktok")
        assert allowed
        assert reason is None

    def test_platform_limit_reached(self):
        # Use a platform with limit < global hourly limit
        # Or use a limiter with higher global limit
        # TikTok limit is 3/24h, global is 3/h
        # Since both are 3, global may fire first.
        # Use a custom limiter with higher global limit to isolate platform test
        limiter = RateLimiter(jitter_range=(0, 0))
        # Override global limit for this test
        limiter.GLOBAL_MAX_PER_HOUR = 100

        for _ in range(3):
            limiter.record_publish("tiktok")

        allowed, reason = limiter.can_publish("tiktok")
        assert not allowed
        assert "tiktok" in reason.lower()

    def test_different_platforms_independent(self):
        limiter = RateLimiter(jitter_range=(0, 0))
        # Override global limit so it doesn't interfere
        limiter.GLOBAL_MAX_PER_HOUR = 100

        for _ in range(3):
            limiter.record_publish("tiktok")

        allowed, _ = limiter.can_publish("instagram_reels")
        assert allowed

    def test_global_hourly_limit(self):
        # Global limit is 3/hour
        for _ in range(3):
            self.limiter.record_publish("instagram_reels")

        allowed, reason = self.limiter.can_publish("tiktok")
        assert not allowed
        assert "hourly" in reason.lower()

    def test_stats(self):
        self.limiter.record_publish("tiktok")
        self.limiter.record_publish("tiktok")
        stats = self.limiter.get_stats()
        assert stats["global_last_hour"] == 2
        assert stats["platforms"]["tiktok"]["used"] == 2
        assert stats["platforms"]["tiktok"]["remaining"] == 1

    def test_jitter_range(self):
        limiter = RateLimiter(jitter_range=(10, 20))
        delay = limiter.get_jitter_delay()
        assert 10 <= delay <= 20
