"""Tests for Dedup Engine."""

import pytest
from datetime import datetime, timedelta
from core.dedup import DedupEngine


class TestDedupEngine:

    def setup_method(self):
        self.engine = DedupEngine(
            max_same_content_platforms=3,
            min_spacing_hours=24,
            niche_cooldown_hours=48,
        )

    def test_first_post_allowed(self):
        allowed, reasons = self.engine.check("hash1", "tiktok", "raw-almonds")
        assert allowed
        assert len(reasons) == 0

    def test_duplicate_blocked(self):
        self.engine.register("hash1", "tiktok", "raw-almonds", "Test post")
        allowed, reasons = self.engine.check("hash1", "tiktok", "raw-almonds")
        assert not allowed
        assert any("Exact duplicate" in r for r in reasons)

    def test_same_content_different_platform_allowed(self):
        self.engine.register("hash1", "tiktok", "raw-almonds")
        allowed, _ = self.engine.check("hash1", "instagram_reels", "raw-almonds")
        assert allowed

    def test_cross_platform_limit(self):
        self.engine.register("hash1", "tiktok", "raw-almonds")
        self.engine.register("hash1", "instagram_reels", "raw-almonds")
        self.engine.register("hash1", "facebook_reels", "raw-almonds")
        allowed, reasons = self.engine.check("hash1", "linkedin", "raw-almonds")
        assert not allowed
        assert any("3 platforms" in r for r in reasons)

    def test_compute_hash_deterministic(self):
        h1 = self.engine.compute_hash("Hello World")
        h2 = self.engine.compute_hash("hello world")
        assert h1 == h2  # Normalized to lowercase

    def test_cleanup_removes_old(self):
        self.engine.register("old", "tiktok", "test")
        # Manually age the entry
        self.engine._entries[0] = self.engine._entries[0]._replace(
            posted_at=datetime.utcnow() - timedelta(days=31)
        )
        removed = self.engine.cleanup(older_than_days=30)
        assert removed == 1
        assert self.engine.total_entries == 0
