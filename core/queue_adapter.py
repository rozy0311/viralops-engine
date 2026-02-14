"""
ViralOps Engine — Queue Adapter
Queue management with retry, exponential backoff, Dead Letter Queue (DLQ).
Idempotent publish with dedup check.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

from core.models import PublishMode, PublishResult, QueueItem

logger = logging.getLogger("viralops.queue")


class DeadLetterQueue:
    """Dead Letter Queue for failed publish attempts."""

    def __init__(self, max_size: int = 1000):
        self._queue: deque[dict] = deque(maxlen=max_size)

    def add(self, item: QueueItem, error: str, attempt: int) -> None:
        self._queue.append({
            "queue_item_id": item.id,
            "content_pack_id": item.content_pack_id,
            "platform": item.platform,
            "error": error,
            "attempt": attempt,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.warning(
            "DLQ: Item %s moved to DLQ after %d attempts. Error: %s",
            item.id, attempt, error,
        )

    def items(self) -> list[dict]:
        return list(self._queue)

    def size(self) -> int:
        return len(self._queue)


class QueueAdapter:
    """
    Publish queue with retry, backoff, and DLQ.
    
    Features:
    - Idempotent publish (content_hash dedup)
    - Exponential backoff with jitter
    - Dead Letter Queue for failed items
    - Priority ordering
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
    ):
        self._queue: list[QueueItem] = []
        self._published_hashes: set[str] = set()
        self._enqueued_hashes: set[str] = set()
        self._results: list[PublishResult] = []
        self.dlq = DeadLetterQueue()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def enqueue(
        self,
        content_pack_id: str,
        platform: str,
        content_hash: str,
        scheduled_time: datetime | None = None,
        priority: int = 5,
    ) -> QueueItem | None:
        """
        Add item to publish queue. Returns None if duplicate.
        """
        # Idempotency check — block if already enqueued OR published
        dedup_key = f"{content_hash}:{platform}"
        if dedup_key in self._published_hashes or dedup_key in self._enqueued_hashes:
            logger.info("Queue: Skipping duplicate %s for %s", content_pack_id, platform)
            return None

        item = QueueItem(
            id=str(uuid4())[:8],
            content_pack_id=content_pack_id,
            platform=platform,
            scheduled_at=scheduled_time or datetime.now(timezone.utc) + timedelta(minutes=5),
            priority=priority,
            retry_count=0,
            content_hash=content_hash,
        )

        self._enqueued_hashes.add(dedup_key)
        self._queue.append(item)
        self._queue.sort(key=lambda x: (x.priority, x.scheduled_at or datetime.min))

        logger.info(
            "Queue: Enqueued %s for %s at %s (priority=%d)",
            item.id, platform, item.scheduled_at, priority,
        )
        return item

    def process_next(
        self,
        publisher_fn: Callable[[QueueItem], PublishResult],
    ) -> PublishResult | None:
        """
        Process the next item in queue with retry logic.
        """
        if not self._queue:
            return None

        item = self._queue.pop(0)

        for attempt in range(1, self.max_retries + 1):
            try:
                result = publisher_fn(item)

                if result.success:
                    self._published_hashes.add(f"{item.content_hash}:{item.platform}")
                    self._results.append(result)
                    logger.info("Queue: Published %s to %s", item.id, item.platform)
                    return result

                # Non-retryable failure
                if result.error and "banned" in result.error.lower():
                    self.dlq.add(item, result.error or "banned", attempt)
                    return result

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    "Queue: Attempt %d/%d failed for %s: %s",
                    attempt, self.max_retries, item.id, error_msg,
                )

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** (attempt - 1)),
                        self.max_delay,
                    )
                    logger.info("Queue: Retrying in %.1fs", delay)
                    time.sleep(delay)
                else:
                    self.dlq.add(item, error_msg, attempt)

        return None

    def process_all(
        self,
        publisher_fn: Callable[[QueueItem], PublishResult],
    ) -> list[PublishResult]:
        """Process all items in queue."""
        results = []
        while self._queue:
            result = self.process_next(publisher_fn)
            if result:
                results.append(result)
        return results

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def published_count(self) -> int:
        return len(self._results)

    @property
    def dlq_count(self) -> int:
        return self.dlq.size()
