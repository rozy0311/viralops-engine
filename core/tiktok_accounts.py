"""
ViralOps Engine — TikTok Multi-Account Manager
================================================
Manages N TikTok accounts for round-robin posting via Publer.

Strategy (from TikTok rate-limit research):
  - Max 3 posts/account/day  → safe from TikTok spam detection
  - Unaudited app: 5 active creators/24h → fits 5 accounts perfectly
  - Random jitter between posts → avoids pattern detection
  - Round-robin across accounts → maximizes daily output

Config source:  DB `settings` table, key = 'tiktok_accounts'
Fallback:       PUBLER_TIKTOK_ACCOUNTS env var (comma-separated IDs)
Legacy:         Hardcoded '698c95e5b1ab790def1352c1' as Account #1

Usage:
  from core.tiktok_accounts import TikTokAccountManager
  mgr = TikTokAccountManager()
  account = mgr.next_account()        # round-robin pick
  mgr.record_post(account["id"])      # track daily usage
"""
from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("viralops.tiktok_accounts")

# ── Constants ──────────────────────────────────────────────────────
MAX_POSTS_PER_ACCOUNT_PER_DAY = 3   # TikTok safe limit
DEFAULT_JITTER_MIN = 15              # Minutes min gap between posts
DEFAULT_JITTER_MAX = 45              # Minutes max gap between posts

# Legacy hardcoded account
LEGACY_TIKTOK_ID = "698c95e5b1ab790def1352c1"

PROJ_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJ_DIR / "viralops.db"


@dataclass
class TikTokAccount:
    """Single TikTok account connected via Publer."""

    id: str                          # Publer account ID (e.g. '698c95e5b1ab790def1352c1')
    label: str = ""                  # Display name (e.g. 'The Rike Root Stories')
    enabled: bool = True             # Can be toggled on/off from UI
    max_daily: int = MAX_POSTS_PER_ACCOUNT_PER_DAY
    posts_today: int = 0             # Tracked per day (API-published)
    drafts_today: int = 0            # Tracked per day (draft fallbacks)
    last_post_at: str = ""           # ISO timestamp of last post
    niche_filter: str = ""           # Optional: restrict to specific niche

    @property
    def total_today(self) -> int:
        """Total activity today (API posts + drafts)."""
        return self.posts_today + self.drafts_today

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "enabled": self.enabled,
            "max_daily": self.max_daily,
            "posts_today": self.posts_today,
            "drafts_today": self.drafts_today,
            "total_today": self.total_today,
            "last_post_at": self.last_post_at,
            "niche_filter": self.niche_filter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TikTokAccount":
        return cls(
            id=d["id"],
            label=d.get("label", ""),
            enabled=d.get("enabled", True),
            max_daily=d.get("max_daily", MAX_POSTS_PER_ACCOUNT_PER_DAY),
            posts_today=d.get("posts_today", 0),
            drafts_today=d.get("drafts_today", 0),
            last_post_at=d.get("last_post_at", ""),
            niche_filter=d.get("niche_filter", ""),
        )


class TikTokAccountManager:
    """
    Multi-account round-robin manager for TikTok posting.

    Stores config in DB `settings` table.
    Tracks daily post counts in `tiktok_daily_log` table.
    Round-robin index persisted across restarts.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or DB_PATH)
        self._accounts: list[TikTokAccount] = []
        self._rr_index: int = 0  # Round-robin pointer
        self._today: str = ""
        self._init_db()
        self._load()

    # ──────────────────────────────────────────────
    # DB Setup
    # ──────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Ensure tables exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS tiktok_daily_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    post_count INTEGER DEFAULT 0,
                    draft_count INTEGER DEFAULT 0,
                    last_post_at TEXT DEFAULT '',
                    UNIQUE(account_id, date)
                );
            """)
            # Ensure draft_count column exists (migration for existing DBs)
            try:
                conn.execute("SELECT draft_count FROM tiktok_daily_log LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE tiktok_daily_log ADD COLUMN draft_count INTEGER DEFAULT 0")
            conn.commit()
        finally:
            conn.close()

    # ──────────────────────────────────────────────
    # Load / Save
    # ──────────────────────────────────────────────

    def _load(self) -> None:
        """Load accounts from DB, falling back to env var, then legacy ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = 'tiktok_accounts'"
            ).fetchone()
        finally:
            conn.close()

        if row:
            data = json.loads(row["value"])
            self._accounts = [TikTokAccount.from_dict(a) for a in data.get("accounts", [])]
            self._rr_index = data.get("rr_index", 0)
        else:
            # Fallback: env var
            env_ids = os.environ.get("PUBLER_TIKTOK_ACCOUNTS", "").strip()
            if env_ids:
                for i, aid in enumerate(env_ids.split(","), 1):
                    aid = aid.strip()
                    if aid:
                        self._accounts.append(TikTokAccount(id=aid, label=f"TikTok #{i}"))
            else:
                # Legacy single account
                self._accounts.append(
                    TikTokAccount(id=LEGACY_TIKTOK_ID, label="The Rike Root Stories")
                )
            self._save()

        # Refresh daily counters
        self._refresh_daily_counts()
        logger.info(
            "TikTok accounts loaded: %d (%d enabled)",
            len(self._accounts),
            sum(1 for a in self._accounts if a.enabled),
        )

    def _save(self) -> None:
        """Persist accounts + round-robin index to DB."""
        data = {
            "accounts": [a.to_dict() for a in self._accounts],
            "rr_index": self._rr_index,
        }
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('tiktok_accounts', ?)",
                (json.dumps(data),),
            )
            conn.commit()
        finally:
            conn.close()

    def _refresh_daily_counts(self) -> None:
        """Load today's post counts from tiktok_daily_log."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._today = today

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT account_id, post_count, COALESCE(draft_count, 0) as draft_count, last_post_at "
                "FROM tiktok_daily_log WHERE date = ?",
                (today,),
            ).fetchall()
        finally:
            conn.close()

        counts = {r["account_id"]: (r["post_count"], r["draft_count"], r["last_post_at"]) for r in rows}
        for acct in self._accounts:
            if acct.id in counts:
                acct.posts_today = counts[acct.id][0]
                acct.drafts_today = counts[acct.id][1]
                acct.last_post_at = counts[acct.id][2]
            else:
                acct.posts_today = 0
                acct.drafts_today = 0
                acct.last_post_at = ""

    # ──────────────────────────────────────────────
    # Round-Robin Selection
    # ──────────────────────────────────────────────

    def next_account(self, niche: str = "") -> Optional[TikTokAccount]:
        """
        Pick the next available account via round-robin.

        Returns None if ALL accounts have hit their daily limit.
        Skips disabled accounts and accounts that reached max_daily.
        """
        self._refresh_daily_counts()

        enabled = [a for a in self._accounts if a.enabled]
        if not enabled:
            logger.warning("No enabled TikTok accounts!")
            return None

        # Try round-robin from current index
        n = len(enabled)
        for offset in range(n):
            idx = (self._rr_index + offset) % n
            acct = enabled[idx]

            # Skip if daily limit reached
            if acct.posts_today >= acct.max_daily:
                continue

            # Skip if niche filter doesn't match
            if niche and acct.niche_filter and acct.niche_filter != niche:
                continue

            # Found a valid account
            self._rr_index = (idx + 1) % n
            self._save()
            logger.info(
                "Selected account: '%s' (%s) — %d/%d posts today",
                acct.label, acct.id, acct.posts_today, acct.max_daily,
            )
            return acct

        logger.warning("All accounts at daily limit! (%d accounts × %d max)",
                        len(enabled), MAX_POSTS_PER_ACCOUNT_PER_DAY)
        return None

    def record_post(self, account_id: str) -> None:
        """Record a successful post for the given account (increments daily count)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO tiktok_daily_log (account_id, date, post_count, last_post_at)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(account_id, date)
                   DO UPDATE SET post_count = post_count + 1, last_post_at = ?""",
                (account_id, today, now_iso, now_iso),
            )
            conn.commit()
        finally:
            conn.close()

        # Update in-memory
        for acct in self._accounts:
            if acct.id == account_id:
                acct.posts_today += 1
                acct.last_post_at = now_iso
                break

        logger.info("Recorded post for '%s' (now %d today)", account_id, 
                     next((a.posts_today for a in self._accounts if a.id == account_id), 0))

    def record_draft(self, account_id: str) -> None:
        """Record a draft fallback for the given account (increments draft count).

        Drafts are created when TikTok API rate-limits → Publer saves as draft
        for manual publish via TikTok UI. Drafts don't count toward API limit
        but count toward total daily activity tracking.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO tiktok_daily_log (account_id, date, draft_count, last_post_at)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(account_id, date)
                   DO UPDATE SET draft_count = draft_count + 1, last_post_at = ?""",
                (account_id, today, now_iso, now_iso),
            )
            conn.commit()
        finally:
            conn.close()

        # Update in-memory
        for acct in self._accounts:
            if acct.id == account_id:
                acct.drafts_today += 1
                acct.last_post_at = now_iso
                break

        logger.info("Recorded draft for '%s' (now %d drafts + %d posts today)", account_id,
                     next((a.drafts_today for a in self._accounts if a.id == account_id), 0),
                     next((a.posts_today for a in self._accounts if a.id == account_id), 0))

    # ──────────────────────────────────────────────
    # Account CRUD (for Web UI)
    # ──────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        """Return all accounts with current daily stats."""
        self._refresh_daily_counts()
        return [a.to_dict() for a in self._accounts]

    def add_account(self, account_id: str, label: str = "", max_daily: int = 3, niche_filter: str = "") -> dict:
        """Add a new TikTok account."""
        # Check duplicate
        if any(a.id == account_id for a in self._accounts):
            return {"success": False, "error": f"Account {account_id} already exists"}

        acct = TikTokAccount(
            id=account_id,
            label=label or f"TikTok #{len(self._accounts) + 1}",
            max_daily=max_daily,
            niche_filter=niche_filter,
        )
        self._accounts.append(acct)
        self._save()
        logger.info("Added TikTok account: '%s' (%s)", acct.label, acct.id)
        return {"success": True, "account": acct.to_dict()}

    def update_account(self, account_id: str, updates: dict) -> dict:
        """Update account settings (label, enabled, max_daily, niche_filter)."""
        for acct in self._accounts:
            if acct.id == account_id:
                if "label" in updates:
                    acct.label = updates["label"]
                if "enabled" in updates:
                    acct.enabled = bool(updates["enabled"])
                if "max_daily" in updates:
                    acct.max_daily = max(1, min(10, int(updates["max_daily"])))
                if "niche_filter" in updates:
                    acct.niche_filter = updates["niche_filter"]
                self._save()
                return {"success": True, "account": acct.to_dict()}
        return {"success": False, "error": f"Account {account_id} not found"}

    def remove_account(self, account_id: str) -> dict:
        """Remove a TikTok account."""
        before = len(self._accounts)
        self._accounts = [a for a in self._accounts if a.id != account_id]
        if len(self._accounts) < before:
            # Adjust round-robin index
            if self._rr_index >= len(self._accounts):
                self._rr_index = 0
            self._save()
            logger.info("Removed TikTok account: %s", account_id)
            return {"success": True}
        return {"success": False, "error": f"Account {account_id} not found"}

    def reorder_accounts(self, ordered_ids: list[str]) -> dict:
        """Reorder accounts by providing ordered list of IDs."""
        id_map = {a.id: a for a in self._accounts}
        reordered = []
        for aid in ordered_ids:
            if aid in id_map:
                reordered.append(id_map.pop(aid))
        # Append any remaining (not in ordered list)
        reordered.extend(id_map.values())
        self._accounts = reordered
        self._rr_index = 0
        self._save()
        return {"success": True, "accounts": [a.to_dict() for a in self._accounts]}

    # ──────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get overall multi-account stats."""
        self._refresh_daily_counts()
        enabled = [a for a in self._accounts if a.enabled]
        total_capacity = sum(a.max_daily for a in enabled)
        total_used = sum(a.posts_today for a in enabled)
        total_drafts = sum(a.drafts_today for a in enabled)
        return {
            "total_accounts": len(self._accounts),
            "enabled_accounts": len(enabled),
            "daily_capacity": total_capacity,
            "posts_today": total_used,
            "drafts_today": total_drafts,
            "total_today": total_used + total_drafts,
            "remaining_today": total_capacity - total_used,
            "strategy": "round_robin_hybrid",
            "jitter_min": DEFAULT_JITTER_MIN,
            "jitter_max": DEFAULT_JITTER_MAX,
            "accounts": [a.to_dict() for a in self._accounts],
        }

    def get_jitter_seconds(self) -> int:
        """Return random jitter in seconds to wait between posts."""
        return random.randint(DEFAULT_JITTER_MIN * 60, DEFAULT_JITTER_MAX * 60)


# ── Singleton ──────────────────────────────────────────────────────
_manager: Optional[TikTokAccountManager] = None


def get_account_manager(db_path: str | Path | None = None) -> TikTokAccountManager:
    """Get or create the singleton TikTokAccountManager."""
    global _manager
    if _manager is None:
        _manager = TikTokAccountManager(db_path=db_path)
    return _manager
