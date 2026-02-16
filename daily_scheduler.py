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
TIKTOK_OPTIMAL_HOURS = [7, 10, 12, 15, 19, 21]   # US/Pacific — 6 slots for multi-account
TIKTOK_RL_RETRY_SLOTS = 3   # After rate-limit, try this many later slots before giving up

# Known-used topic IDs (historical — dedup also checks posts table)
USED_IDS: set[int] = {247, 248, 257, 274, 313, 327}

# ── Multi-Account Manager ──────────────────────────────────────────
from core.tiktok_accounts import get_account_manager  # noqa: E402


# ── Telegram Notification ──────────────────────────────────────────
def _notify_telegram(message: str) -> bool:
    """Send a notification to Telegram (sync, best-effort)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        import httpx
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        r = httpx.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.warning("Telegram notify failed: %s", e)
        return False


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
    return "too many posts" in msg or "rate limit" in msg or "rate_limit" in msg or "ratelimit" in msg or "429" in msg or "openapi" in msg


# ── Main pipeline ─────────────────────────────────────────────────
def run_once(
    dry_run: bool = False,
    force_topic_id: int | None = None,
    account_id: str | None = None,
) -> bool:
    """
    Generate + publish a single post.
    If account_id is provided, publish to that specific Publer TikTok account.
    Otherwise, the account manager picks the next available via round-robin.
    Returns True on success, False on failure / rate-limit.
    """
    mgr = get_account_manager()
    acct = None
    if account_id:
        # Specific account requested
        all_accts = mgr.get_all()
        acct = next((a for a in all_accts if a["id"] == account_id), None)
        if not acct:
            log.error("Account %s not found!", account_id)
            return False
        log.info("═══ ViralOps — run_once() → account '%s' ═══", acct.get("label", account_id))
    else:
        acct_obj = mgr.next_account()
        if not acct_obj:
            log.warning("All TikTok accounts at daily limit — skipping")
            _notify_telegram("⏳ *ViralOps* — All TikTok accounts at daily limit. Waiting for tomorrow.")
            return False
        acct = acct_obj.to_dict()
        log.info("═══ ViralOps — run_once() → account '%s' (%d/%d today) ═══",
                 acct["label"], acct["posts_today"], acct["max_daily"])

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
        _notify_telegram(f"❌ *ViralOps* — Content generation FAILED\nTopic: {topic[:60]}\nID: {tid}")
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

    # 4. Publish via Publer → TikTok (multi-account)
    from publish_microniche import main as publish_main  # noqa: E402
    target_account_id = acct["id"]
    log.info("Publishing to account: %s (%s)", acct.get("label", "?"), target_account_id)

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("Publish attempt %d/%d …", attempt, MAX_RETRIES)
        try:
            result = publish_main(content_pack_override=pack, tiktok_account_id=target_account_id)
            # result: "published" | "draft" | False
            if result == "published":
                mgr.record_post(target_account_id)
                stats = mgr.get_stats()
                log.info("✅  Published to '%s'!  Topic: %s (id=%d)  [%d/%d today]",
                         acct.get("label", "?"), topic, tid,
                         stats["posts_today"], stats["daily_capacity"])
                _jsonl_log({
                    "event": "published",
                    "topic_id": tid,
                    "topic": topic,
                    "title": title,
                    "content_len": content_len,
                    "caption_len": len(caption),
                    "dot_breaks": dot_count,
                    "status": "published",
                    "account_id": target_account_id,
                    "account_label": acct.get("label", ""),
                })
                _notify_telegram(
                    f"✅ *ViralOps — Published!*\n"
                    f"📝 {title}\n"
                    f"🎵 Account: {acct.get('label', target_account_id[:8])}\n"
                    f"📊 {content_len} chars | {stats['posts_today']}/{stats['daily_capacity']} today\n"
                    f"🆔 Topic #{tid} (score {score:.1f})"
                )
                return True
            elif result == "draft":
                # Hybrid fallback: API rate-limited → Publer draft created
                mgr.record_draft(target_account_id)
                stats = mgr.get_stats()
                log.info("📋  Draft created for '%s' (API rate-limited).  Topic: %s (id=%d)",
                         acct.get("label", "?"), topic, tid)
                _jsonl_log({
                    "event": "draft_fallback",
                    "topic_id": tid,
                    "topic": topic,
                    "title": title,
                    "content_len": content_len,
                    "status": "draft_pending",
                    "account_id": target_account_id,
                    "account_label": acct.get("label", ""),
                })
                # Telegram notification already sent by publish_microniche._notify_draft_fallback
                return True  # Content was generated and saved — draft counts as success
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
                _notify_telegram(
                    f"⏳ *ViralOps — TikTok Rate Limited*\n"
                    f"Topic: {topic[:50]}\n"
                    f"Will retry at next slot (daemon) or tomorrow."
                )
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
    _notify_telegram(
        f"❌ *ViralOps — Publish FAILED*\n"
        f"Topic: {topic[:50]} (#{tid})\n"
        f"All {MAX_RETRIES} attempts exhausted."
    )
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
    """Run forever, publishing N posts/day across multiple TikTok accounts.

    Multi-account round-robin strategy:
      - Each account gets max 3 posts/day (safe from TikTok spam detection)
      - Posts spread across optimal time slots with random jitter
      - When all accounts hit daily limit → sleep until tomorrow
    """
    mgr = get_account_manager()
    stats = mgr.get_stats()
    log.info("═══ DAEMON MODE — Multi-Account (%d accounts, %d posts/day capacity) ═══",
             stats["enabled_accounts"], stats["daily_capacity"])
    log.info("Optimal hours: %s  |  Jitter: %d-%d min",
             TIKTOK_OPTIMAL_HOURS, stats["jitter_min"] // 60 if stats["jitter_min"] > 60 else stats["jitter_min"],
             stats["jitter_max"] // 60 if stats["jitter_max"] > 60 else stats["jitter_max"])

    tg_ok = _notify_telegram(
        f"🚀 *ViralOps Daemon Started*\n"
        f"📊 {stats['enabled_accounts']} accounts × {MAX_POSTS_PER_ACCOUNT_PER_DAY} max/each\n"
        f"🎯 Capacity: {stats['daily_capacity']} posts/day\n"
        f"⏰ Hours: {TIKTOK_OPTIMAL_HOURS}"
    )
    from core.tiktok_accounts import MAX_POSTS_PER_ACCOUNT_PER_DAY  # noqa: E402
    if tg_ok:
        log.info("Telegram notification: ON")
    else:
        log.info("Telegram notification: OFF (no token/chat_id or failed)")

    while True:
        # Refresh stats at start of each cycle
        stats = mgr.get_stats()

        # Check if all accounts are at daily limit
        if stats["remaining_today"] <= 0:
            log.info("All accounts at daily limit (%d/%d). Sleeping until tomorrow.",
                     stats["posts_today"], stats["daily_capacity"])
            next_day = dt.datetime.now() + dt.timedelta(days=1)
            first_hour = sorted(TIKTOK_OPTIMAL_HOURS)[0]
            wake = next_day.replace(hour=first_hour, minute=random.randint(0, 14), second=0)
            sleep_secs = (wake - dt.datetime.now()).total_seconds()
            if sleep_secs > 0:
                log.info("Next run: %s (%.1f hours)", wake.strftime("%Y-%m-%d %H:%M"), sleep_secs / 3600)
                time.sleep(sleep_secs)
            continue

        # Wait for next optimal slot
        target = _next_optimal_slot()
        wait = (target - dt.datetime.now()).total_seconds()
        if wait > 0:
            log.info("Next post at %s  (sleeping %.0f min)  [%d/%d posted today]",
                     target.strftime("%Y-%m-%d %H:%M"), wait / 60,
                     stats["posts_today"], stats["daily_capacity"])
            time.sleep(wait)

        # Post one — round-robin picks the account
        ok = run_once()

        if ok:
            # Add random jitter before next post
            jitter = mgr.get_jitter_seconds()
            stats = mgr.get_stats()
            if stats["remaining_today"] > 0:
                log.info("✅ Posted! %d/%d today. Next in %d min (jitter).",
                         stats["posts_today"], stats["daily_capacity"], jitter // 60)
                time.sleep(jitter)
            else:
                log.info("✅ Posted! All %d slots filled for today.", stats["daily_capacity"])
        else:
            # Failed/rate-limited — try remaining slots
            retries_left = TIKTOK_RL_RETRY_SLOTS
            while retries_left > 0:
                retries_left -= 1
                next_slot = _next_optimal_slot()
                if next_slot.date() > dt.datetime.now().date():
                    log.info("No more slots today. Will try tomorrow.")
                    break
                wait = (next_slot - dt.datetime.now()).total_seconds()
                if wait > 0:
                    log.info("Retrying at %s (%.0f min)…", next_slot.strftime("%H:%M"), wait / 60)
                    time.sleep(wait)
                ok = run_once()
                if ok:
                    break
            if not ok:
                _notify_telegram("😴 *ViralOps* — All retry slots used today. Waiting until tomorrow.")


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
