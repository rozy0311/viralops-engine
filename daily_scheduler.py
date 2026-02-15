#!/usr/bin/env python3
"""
ViralOps Engine — Daily Auto-Scheduler  (v1.0)
===============================================
Runs once per day via Windows Task Scheduler (or cron).
  1. Picks the best unused high-score topic from niche_hunter.db
  2. Generates quality content + AI image
  3. Publishes to TikTok via Publer
  4. Logs every run to  D:\\vops-data\\scheduler_log.jsonl
  5. Handles TikTok 24-hour rate-limit gracefully (backs off)

Usage
-----
  # One-shot (publish 1 post now):
  python daily_scheduler.py

  # Continuous daemon (publish 1 post per day, sleeping between):
  python daily_scheduler.py --daemon

  # Dry-run (generate content, skip publish):
  python daily_scheduler.py --dry-run

  # Force a specific topic id:
  python daily_scheduler.py --topic-id 274

Windows Task Scheduler
----------------------
  Action:  pythonw.exe  daily_scheduler.py
  Trigger: Daily at 07:00 AM (US/Pacific) — TikTok peak slot
  Working directory: <project root>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import sqlite3
import sys
import time
import traceback
from pathlib import Path

# ── Project bootstrap ──────────────────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_DIR))

from dotenv import load_dotenv
load_dotenv(PROJ_DIR / ".env")

# Ensure required env-vars
os.environ.setdefault("DATA_ROOT", r"D:\vops-data")

# ── Logging ────────────────────────────────────────────────────────
DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"D:\vops-data"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_ROOT / "scheduler.log"
JSONL_LOG = DATA_ROOT / "scheduler_log.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("viralops.scheduler")

# ── Constants ──────────────────────────────────────────────────────
DB_PATH = PROJ_DIR / "niche_hunter.db"
MIN_SCORE = 8.5              # Minimum acceptable topic score
MAX_RETRIES = 3              # Retry count on transient publish failures
RETRY_WAIT_SECS = 120        # Wait 2min between retries
RATE_LIMIT_WAIT_HOURS = 6    # If TikTok rate-limited, wait 6h and retry (daemon mode)
TIKTOK_OPTIMAL_HOURS = [7, 10, 19, 21]   # US/Pacific peak hours

# Known-used topic IDs (historical — dedup also checks posts table)
USED_IDS: set[int] = {247, 248, 257, 274, 313, 327}


# ── Helpers ────────────────────────────────────────────────────────
def _jsonl_log(entry: dict) -> None:
    """Append one JSON line to the scheduler log."""
    entry["ts"] = dt.datetime.utcnow().isoformat() + "Z"
    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_used_ids_from_log() -> set[int]:
    """Parse JSONL to find all topic IDs that were already published."""
    ids: set[int] = set(USED_IDS)
    if JSONL_LOG.exists():
        for line in JSONL_LOG.read_text("utf-8").splitlines():
            try:
                obj = json.loads(line)
                if obj.get("status") == "published" and obj.get("topic_id"):
                    ids.add(int(obj["topic_id"]))
            except Exception:
                pass
    return ids


def pick_topic(force_id: int | None = None) -> tuple[int, str, float, str, str] | None:
    """Pick the highest-scoring unused topic from niche_hunter.db."""
    used = _load_used_ids_from_log()

    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            "SELECT id, topic, final_score, niche, COALESCE(hook, '') "
            "FROM niche_scores "
            "WHERE final_score >= ? "
            "ORDER BY final_score DESC",
            (MIN_SCORE,),
        ).fetchall()
    finally:
        conn.close()

    if force_id:
        for r in rows:
            if r[0] == force_id:
                return r
        log.error("Topic ID %d not found in DB.", force_id)
        return None

    # Filter out used topics
    for r in rows:
        if r[0] not in used:
            return r

    log.warning("All high-score topics exhausted!  (used %d IDs)", len(used))
    return None


def _is_rate_limited_error(exc: Exception) -> bool:
    """Check if the error is a TikTok 24h rate limit."""
    msg = str(exc).lower()
    return "too many posts" in msg or "rate" in msg or "429" in msg


# ── Main pipeline ─────────────────────────────────────────────────
def run_once(
    dry_run: bool = False,
    force_topic_id: int | None = None,
) -> bool:
    """
    Generate + publish a single post.
    Returns True on success, False on failure / rate-limit.
    """
    log.info("═══ ViralOps Daily Scheduler — run_once() ═══")

    # 1. Pick topic
    topic_row = pick_topic(force_topic_id)
    if not topic_row:
        _jsonl_log({"event": "no_topic", "status": "skipped"})
        return False

    tid, topic, score, niche, hook = topic_row
    log.info("TOPIC: %s  (id=%d, score=%.1f, niche=%s)", topic, tid, score, niche)
    _jsonl_log({
        "event": "topic_picked",
        "topic_id": tid,
        "topic": topic,
        "score": score,
        "niche": niche,
    })

    # 2. Generate quality content
    from llm_content import generate_quality_post  # noqa: E402
    log.info("Generating content …")
    pack = generate_quality_post(topic=topic, score=score, location="LA", season="")
    if not pack:
        log.error("Content generation FAILED for topic %d", tid)
        _jsonl_log({"event": "gen_failed", "topic_id": tid, "status": "failed"})
        return False

    title = pack.get("title", topic)[:80]
    content_len = len(pack.get("content", ""))
    hashtags = pack.get("hashtags", [])
    log.info("Generated: '%s'  (%d chars, %d hashtags)", title, content_len, len(hashtags))

    # 3. Build caption (verify dot-line spacing)
    from publish_microniche import build_caption  # noqa: E402
    caption = build_caption(pack, "LA", "")
    dot_count = caption.count("\n.\n")
    log.info("Caption: %d chars, %d dot-breaks", len(caption), dot_count)

    if dry_run:
        log.info("DRY-RUN — skipping publish.  Caption preview:\n%s", caption[:500])
        _jsonl_log({
            "event": "dry_run",
            "topic_id": tid,
            "topic": topic,
            "title": title,
            "content_len": content_len,
            "caption_len": len(caption),
            "dot_breaks": dot_count,
            "status": "dry_run",
        })
        return True

    # 4. Publish via Publer → TikTok
    from publish_microniche import main as publish_main  # noqa: E402

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("Publish attempt %d/%d …", attempt, MAX_RETRIES)
        try:
            success = publish_main(content_pack_override=pack)
            if success:
                log.info("✅  Published successfully!  Topic: %s (id=%d)", topic, tid)
                _jsonl_log({
                    "event": "published",
                    "topic_id": tid,
                    "topic": topic,
                    "title": title,
                    "content_len": content_len,
                    "caption_len": len(caption),
                    "dot_breaks": dot_count,
                    "status": "published",
                })
                return True
            else:
                log.warning("publish_main returned False (attempt %d)", attempt)
        except Exception as exc:
            if _is_rate_limited_error(exc):
                log.warning("TikTok rate-limited: %s", exc)
                _jsonl_log({
                    "event": "rate_limited",
                    "topic_id": tid,
                    "status": "rate_limited",
                    "error": str(exc),
                })
                return False  # Caller will handle wait/retry
            log.error("Publish error (attempt %d): %s", attempt, exc)
            traceback.print_exc()

        if attempt < MAX_RETRIES:
            log.info("Waiting %ds before retry …", RETRY_WAIT_SECS)
            time.sleep(RETRY_WAIT_SECS)

    log.error("All %d publish attempts failed for topic %d", MAX_RETRIES, tid)
    _jsonl_log({
        "event": "publish_failed",
        "topic_id": tid,
        "status": "failed",
    })
    return False


# ── Daemon mode ────────────────────────────────────────────────────
def _next_optimal_slot() -> dt.datetime:
    """Find the next TikTok optimal posting time (US/Pacific)."""
    now = dt.datetime.now()
    today_slots = sorted(TIKTOK_OPTIMAL_HOURS)

    for h in today_slots:
        candidate = now.replace(hour=h, minute=random.randint(0, 14), second=0, microsecond=0)
        if candidate > now + dt.timedelta(minutes=5):
            return candidate

    # All today's slots passed → tomorrow's first slot
    tomorrow = now + dt.timedelta(days=1)
    h = today_slots[0]
    return tomorrow.replace(hour=h, minute=random.randint(0, 14), second=0, microsecond=0)


def daemon_loop() -> None:
    """Run forever, publishing 1 post per day at the next optimal TikTok time slot."""
    log.info("═══ DAEMON MODE — 1 post/day at TikTok peak hours ═══")
    log.info("Optimal hours: %s", TIKTOK_OPTIMAL_HOURS)

    while True:
        target = _next_optimal_slot()
        wait = (target - dt.datetime.now()).total_seconds()
        if wait > 0:
            log.info("Next post at %s  (sleeping %.0f min)", target.strftime("%Y-%m-%d %H:%M"), wait / 60)
            time.sleep(wait)

        ok = run_once()
        if not ok:
            # Rate-limited or failed → wait RATE_LIMIT_WAIT_HOURS and retry
            log.info("Failed/rate-limited. Waiting %dh before retry …", RATE_LIMIT_WAIT_HOURS)
            time.sleep(RATE_LIMIT_WAIT_HOURS * 3600)
            continue

        # Success — wait until next day's first optimal slot
        next_day = dt.datetime.now() + dt.timedelta(days=1)
        first_hour = sorted(TIKTOK_OPTIMAL_HOURS)[0]
        wake = next_day.replace(hour=first_hour, minute=random.randint(0, 14), second=0)
        sleep_secs = (wake - dt.datetime.now()).total_seconds()
        if sleep_secs > 0:
            log.info("Done for today. Next run: %s (%.1f hours)", wake.strftime("%Y-%m-%d %H:%M"), sleep_secs / 3600)
            time.sleep(sleep_secs)


# ── CLI ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ViralOps Daily Auto-Scheduler")
    parser.add_argument("--daemon", action="store_true", help="Continuous 1-post/day loop")
    parser.add_argument("--dry-run", action="store_true", help="Generate content but skip publish")
    parser.add_argument("--topic-id", type=int, default=None, help="Force a specific topic ID")
    args = parser.parse_args()

    if args.daemon:
        daemon_loop()
    else:
        ok = run_once(dry_run=args.dry_run, force_topic_id=args.topic_id)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
