"""Tests for Queue Adapter with retry and DLQ."""

import pytest
from datetime import datetime, timezone
from core.queue_adapter import QueueAdapter, DeadLetterQueue
from core.models import PublishResult, QueueItem


class TestQueueAdapter:

    def setup_method(self):
        self.queue = QueueAdapter(max_retries=2, base_delay=0.01, max_delay=0.1)

    def test_enqueue(self):
        item = self.queue.enqueue("pack-1", "tiktok", "hash1")
        assert item is not None
        assert item.platform == "tiktok"
        assert self.queue.pending_count == 1

    def test_enqueue_dedup(self):
        self.queue.enqueue("pack-1", "tiktok", "hash1")
        duplicate = self.queue.enqueue("pack-1", "tiktok", "hash1")
        assert duplicate is None  # Blocked by idempotency

    def test_process_success(self):
        self.queue.enqueue("pack-1", "tiktok", "hash1")

        def mock_publisher(item: QueueItem) -> PublishResult:
            return PublishResult(
                queue_item_id=item.id, platform=item.platform,
                success=True, post_url="https://example.com/post/1",
                published_at=datetime.now(timezone.utc),
            )

        result = self.queue.process_next(mock_publisher)
        assert result is not None
        assert result.success
        assert self.queue.published_count == 1
        assert self.queue.pending_count == 0

    def test_process_failure_goes_to_dlq(self):
        self.queue.enqueue("pack-1", "tiktok", "hash1")

        def mock_fail(item: QueueItem) -> PublishResult:
            raise Exception("Network error")

        result = self.queue.process_next(mock_fail)
        assert result is None
        assert self.queue.dlq_count == 1

    def test_priority_ordering(self):
        self.queue.enqueue("low", "linkedin", "h1", priority=10)
        self.queue.enqueue("high", "tiktok", "h2", priority=1)
        self.queue.enqueue("mid", "instagram_reels", "h3", priority=5)

        items = self.queue._queue
        assert items[0].content_pack_id == "high"
        assert items[-1].content_pack_id == "low"


class TestDeadLetterQueue:

    def test_add_and_retrieve(self):
        dlq = DeadLetterQueue()
        item = QueueItem(
            id="q1", content_pack_id="p1", platform="tiktok",
            scheduled_at=datetime.now(timezone.utc), priority=5,
            retry_count=3, content_hash="hash1",
        )
        dlq.add(item, "Network timeout", 3)
        assert dlq.size() == 1
        assert dlq.items()[0]["error"] == "Network timeout"
