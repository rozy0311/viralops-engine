"""ViralOps — Auto Niche Topics Batch Publisher

Goal
----
Publish a few posts using auto-picked topics from niche_hunter.db (unused, high score)
so we don't reuse a hardcoded TOPICS list.

By default this runs in DRY-RUN mode and prints the selected topics.

Usage
-----
  # Preview 3 topics (no publish)
  python scripts/batch_auto_niche_topics.py

  # Publish 3 topics × 4 accounts = 12 posts
  python scripts/batch_auto_niche_topics.py --publish

  # Publish 5 topics
  python scripts/batch_auto_niche_topics.py --publish --limit 5

  # Refresh niche_hunter.db first (local scoring only, no Gemini)
  python scripts/batch_auto_niche_topics.py --refresh-db local

  # Refresh niche_hunter.db using Gemini expansion (if keys available)
  python scripts/batch_auto_niche_topics.py --refresh-db full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
import sqlite3
import sys
import time
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)

TIKTOK_1 = "698c95e5b1ab790def1352c1"   # The Rike Root Stories
TIKTOK_2 = "69951ea30c4677f27c12d98c"   # The Rike Stories
# Legacy IDs kept for backwards compatibility; prefer Publer discovery.
FACEBOOK = "699522978f7449fba2dceebe"    # The Rike (Facebook)
PINTEREST = "69951f098f7449fba2dceadd"   # therike (Pinterest)


def _parse_ideas_file_list(raw_value: str) -> list[str]:
    """Parse a multi-file ideas source string.

    Accepts a single file path or a list separated by: newline, ',', or ';'.
    """
    raw = str(raw_value or "").strip()
    if not raw:
        return []

    # Allow simple list syntax from copy/paste.
    if raw.startswith("[") and raw.endswith("]"):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                raw = "\n".join([str(x) for x in arr])
        except Exception:
            pass

    parts = re.split(r"[\n,;]+", raw)
    out: list[str] = []
    for p in parts:
        p = str(p or "").strip().strip('"').strip("'")
        if not p:
            continue
        if os.path.isfile(p):
            out.append(p)
    # De-dupe while preserving order.
    uniq: list[str] = []
    seen: set[str] = set()
    for p in out:
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _auto_detect_default_ideas_file() -> str:
    """Best-effort auto-detect for curated micro/nano niche ideas files.

    Priority:
      1) Env: VIRALOPS_IDEAS_FILE (supports multi-file lists)
      2) Sibling folder next to this repo (matches current workspace layout):
         - Prefer part 1 + part 2 winner-post files (explicitly curated)
         - Then fall back to the big ideas table file(s)
    """
    env_raw = str(os.environ.get("VIRALOPS_IDEAS_FILE", "") or "").strip()
    if _parse_ideas_file_list(env_raw):
        return env_raw

    # Repo root: .../Agent Multi-Channel Scheduler Content Factory — ViralOps Engine
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling_dir = os.path.dirname(proj_root)

    curated_dir = os.path.join(sibling_dir, "Micro Niche Blogs - Universal Caption - hashtag")

    # Prefer the curated winner-post parts (user-maintained).
    part1 = os.path.join(curated_dir, "part 1 micro niche - nano niche blog - winner post.md")
    part2 = os.path.join(curated_dir, "part 2 micro niche - nano niche blog - winner post.md")
    curated: list[str] = []
    # Prefer part 2 first (usually contains the large ideas list).
    if os.path.isfile(part2):
        curated.append(part2)
    if os.path.isfile(part1):
        curated.append(part1)
    if curated:
        # Store as a multi-file string so argparse can keep a single --ideas-file value.
        return ";".join(curated)

    # Fallback: dedicated ideas table file(s) (often store hundreds of ideas as rows).
    for name in ("Micro niche blogs idea.md", "Micro niche blogs idea.txt"):
        candidate = os.path.join(curated_dir, name)
        if os.path.isfile(candidate):
            return candidate

    return ""


async def _get_tiktok_accounts() -> list[tuple[str, str]]:
    """Return list of (account_id, label) for TikTok.

    Priority:
      1) Env `VIRALOPS_TIKTOK_ACCOUNT_IDS` (comma-separated Publer account IDs)
      2) Publer discovery: all connected TikTok accounts in workspace
      3) Fallback to the two historical hardcoded IDs if present
    """
    env_ids = os.environ.get("VIRALOPS_TIKTOK_ACCOUNT_IDS", "").strip()
    if env_ids:
        ids = [x.strip() for x in env_ids.split(",") if x.strip()]
        return [(aid, f"tiktok_{i+1}") for i, aid in enumerate(ids)]

    # Discover from Publer (best for scaling: auto-updates when you connect more accounts)
    try:
        from integrations.publer_publisher import PublerPublisher

        pub = PublerPublisher()
        await pub.connect()
        accounts = await pub.get_accounts()
        out: list[tuple[str, str]] = []
        for acc in accounts or []:
            acc_type = str(acc.get("type", acc.get("provider", "")) or "").lower()
            if "tiktok" not in acc_type:
                continue
            acc_id = str(acc.get("id", acc.get("_id", "")) or "").strip()
            if not acc_id:
                continue
            name = str(acc.get("name", "") or "").strip()
            label = name or acc_id[-6:]
            out.append((acc_id, label))
        if out:
            return out
    except Exception:
        pass

    # Final fallback: keep legacy behavior
    legacy: list[tuple[str, str]] = []
    if TIKTOK_1:
        legacy.append((TIKTOK_1, "RikeRoot"))
    if TIKTOK_2:
        legacy.append((TIKTOK_2, "RikeStories"))
    return legacy


def _log_published_to_viralops_db(
    pack: dict[str, Any],
    platforms: list[str],
    extra: dict[str, Any] | None = None,
    *,
    status: str = "published",
) -> None:
    """Persist a post attempt to web/viralops.db so future runs can dedup/debug."""
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return

    title = str(pack.get("title", "") or "").strip()
    body = str(pack.get("universal_caption_block", "") or pack.get("content_formatted", "") or "").strip()
    if not title:
        return

    payload = dict(extra or {})
    payload.update({
        "_source": pack.get("_source", ""),
        "_niche_score": pack.get("_niche_score", None),
        "_review_score": pack.get("_review_score", None),
        "_gen_provider": pack.get("_gen_provider", ""),
        "_gen_model": pack.get("_gen_model", ""),
    })

    status = (status or payload.get("status") or "published").strip().lower()

    published_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO posts (title, body, platforms, status, published_at, extra_fields) VALUES (?, ?, ?, ?, ?, ?)",
            (
                title,
                body,
                json.dumps(platforms),
                status,
                published_at,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _load_post_from_viralops_db(post_id: int) -> dict[str, Any] | None:
    """Load a previously-generated post from web/viralops.db.

    This is used for TikTok-only retry runs where we want to re-post the exact
    same content instead of regenerating.
    """
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, title, body, extra_fields FROM posts WHERE id = ?",
            (int(post_id),),
        ).fetchone()
        if not row:
            return None

        title = str(row["title"] or "").strip()
        body = str(row["body"] or "").strip()
        extra_raw = str(row["extra_fields"] or "").strip()
        extra: dict[str, Any] = {}
        if extra_raw:
            try:
                extra = json.loads(extra_raw)
            except Exception:
                extra = {}

        topic = str(extra.get("topic") or "").strip()

        # Best-effort: recover hashtags embedded in the body.
        tags = re.findall(r"#([A-Za-z][A-Za-z0-9_]{1,30})", body)
        seen: set[str] = set()
        hashtags: list[str] = []
        for t in tags:
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            hashtags.append(t)

        pack: dict[str, Any] = {
            "title": title,
            "content_formatted": body,
            "universal_caption_block": body,
            "hashtags": hashtags,
            "_topic": topic or title,
            "_source": "viralops_db_retry",
        }

        # Ensure we have an image prompt so TikTok can generate a photo if needed.
        try:
            from llm_content import build_image_prompt

            pack["image_prompt"] = build_image_prompt(pack)
        except Exception:
            pass

        return pack
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _ensure_topic_history_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS topic_history (
            topic TEXT PRIMARY KEY,
            source TEXT,
            created_at TEXT
        )
        """
    )


def _topic_already_used(topic: str) -> bool:
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return False
    t = (topic or "").strip()
    if not t:
        return True
    conn = sqlite3.connect(db_path)
    try:
        _ensure_topic_history_table(conn)
        row = conn.execute("SELECT 1 FROM topic_history WHERE topic = ?", (t,)).fetchone()
        return bool(row)
    finally:
        conn.close()


def _mark_topic_used(topic: str, source: str) -> None:
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return
    t = (topic or "").strip()
    if not t:
        return
    conn = sqlite3.connect(db_path)
    try:
        _ensure_topic_history_table(conn)
        conn.execute(
            "INSERT OR IGNORE INTO topic_history (topic, source, created_at) VALUES (?, ?, ?)",
            (t, source or "", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _reset_ideas_topic_history(*, ideas_topics: list[str]) -> int:
    """Remove ideas-file topics from topic_history so they can be reused."""
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return 0
    topics = [t.strip() for t in (ideas_topics or []) if str(t).strip()]
    if not topics:
        return 0

    conn = sqlite3.connect(db_path)
    try:
        _ensure_topic_history_table(conn)
        q = "DELETE FROM topic_history WHERE source = ? AND topic = ?"
        deleted = 0
        for t in topics:
            for src in ("ideas_file", "ideas_file_table", "ideas_file_text"):
                cur = conn.execute(q, (src, t))
                deleted += int(cur.rowcount or 0)
        conn.commit()
        return deleted
    finally:
        conn.close()


def _reset_direct_post_history(*, direct_topics: list[str] | None = None, any_source: bool = True) -> int:
    """Remove extracted direct-post topics from topic_history.

    Default behavior (any_source=True) deletes matching topics regardless of source.
    This is needed because a direct-post topic can also be backfilled as `publer_dedup`.

    If direct_topics is not provided, deletes all rows where source=ideas_direct_post.
    """
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return 0

    conn = sqlite3.connect(db_path)
    try:
        _ensure_topic_history_table(conn)

        if direct_topics:
            deleted = 0
            for t in direct_topics:
                t = str(t or "").strip()
                if not t:
                    continue
                if any_source:
                    cur = conn.execute("DELETE FROM topic_history WHERE topic = ?", (t,))
                else:
                    cur = conn.execute(
                        "DELETE FROM topic_history WHERE source = ? AND topic = ?",
                        ("ideas_direct_post", t),
                    )
                deleted += int(cur.rowcount or 0)
            conn.commit()
            return deleted

        cur = conn.execute("DELETE FROM topic_history WHERE source = ?", ("ideas_direct_post",))
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def _refresh_db(mode: str) -> None:
    """Refresh niche_hunter.db by running niche_hunter workflow."""
    from niche_hunter import run_niche_hunter

    if mode == "local":
        run_niche_hunter(use_gemini=False, top_n=20, save_db=True)
        return

    if mode == "full":
        run_niche_hunter(use_gemini=True, top_n=20, save_db=True)
        return

    raise ValueError(f"Unknown refresh-db mode: {mode}")


def _pick_topics(limit: int, min_score: float) -> list[tuple[str, float, str, str]]:
    """Pick unused topics from niche_hunter.db via llm_content helper."""
    from llm_content import get_unused_topics

    # get_unused_topics already sorts by final_score DESC and filters published titles
    candidates = get_unused_topics(top_n=max(20, limit * 5))

    picked: list[tuple[str, float, str, str]] = []
    for topic, score, niche, hook in candidates:
        if score < min_score:
            continue
        picked.append((topic, float(score), niche or "", hook or ""))
        if len(picked) >= limit:
            break

    return picked


def _strip_md_links(text: str) -> str:
    # Remove markdown links: [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", text)
    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove trailing citation artifacts like ".nanalyze" / ".pbpc" that occur
    # when the source was like ".[nanalyze](...)".
    text = re.sub(r"\.[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*\s*$", "", text)
    return text


def _clean_direct_answer_text(text: str) -> str:
    """Clean a copied GenAI answer so it can be posted directly.

    Goal: keep the answer, remove meta/prompt/table noise.
    """
    if not text:
        return ""

    s = str(text).replace("\ufeff", "").replace("\u200b", "")

    # If the user stored a Q/A format, keep only the assistant part.
    m = re.search(r"\bgen\s*ai\s*:\s*", s, flags=re.IGNORECASE)
    if m:
        s = s[m.end():]

    lines = [ln.rstrip() for ln in s.splitlines()]
    out: list[str] = []
    skip_table = False
    for ln in lines:
        raw = ln.strip()
        if not raw:
            out.append("")
            continue

        low = raw.lower()

        # Strip obvious meta / prompt instructions
        if low in ("text", "3 attachments"):
            continue
        if low.startswith((
            "write in english only",
            "context:",
            "quy tắc",
            "next time",
            "prompt nâng cao",
            "full tự động",
            "cách test",
            "rồi ví dụ",
            "gửi tui prompt",
            "bạn là chuyên gia",
            "original example:",
        )):
            continue

        # Drop markdown tables or tab-separated tables
        if raw.startswith("|") and raw.endswith("|"):
            continue
        if "\t" in raw and raw.count("\t") >= 2:
            continue
        if low.startswith(("comparison table", "category", "niche level", "material source")):
            skip_table = True
            continue
        if skip_table:
            # exit table section once we hit a non-tab/non-table line
            if ("\t" not in raw) and (not raw.startswith("|")) and ("\t" not in ln):
                skip_table = False
            else:
                continue

        # Remove stray zero-width and normalize spaces
        raw = re.sub(r"[\u2800\u200B\u200C\u200D\uFEFF]", "", raw)
        raw = _strip_md_links(raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        out.append(raw)

    # Collapse excessive blank lines
    cleaned_lines: list[str] = []
    blank_run = 0
    for ln in out:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(ln)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned


def _title_from_text(text: str) -> str:
    """Pick a reasonable title from the first non-empty line/sentence."""
    if not text:
        return ""
    for ln in str(text).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # Use first line, trimmed.
        t = re.sub(r"\s+", " ", ln).strip()
        # If the first line is huge, shorten to first sentence.
        if len(t) > 120:
            parts = re.split(r"(?<=[.!?])\s+", t)
            t = parts[0].strip() if parts else t[:120].strip()
        return t[:120]
    return ""


def _build_direct_content_pack(answer_text: str) -> dict[str, Any]:
    """Create a minimal content_pack for direct publishing (no LLM rewrite)."""
    cleaned = _clean_direct_answer_text(answer_text)
    title = _title_from_text(cleaned) or "Micro Niche Tip"

    # Ensure we have an image prompt so TikTok/Pinterest can generate an image.
    img_title = re.sub(r"[^A-Za-z0-9 ]+", " ", title).strip()
    img_title = re.sub(r"\s+", " ", img_title)[:90]
    image_prompt = f"Vertical 9:16 photo illustration of {img_title}, natural light, high quality, realistic, no text"

    return {
        "title": title,
        "content_formatted": cleaned,
        "universal_caption_block": cleaned,
        "hashtags": [],
        "image_prompt": image_prompt,
        "_source": "ideas_direct_post",
        "_review_pass": True,
        "_review_score": 10.0,
        "_rubric_total_100": 100.0,
    }


def _pick_topics_from_ideas_file(file_path: str, limit: int) -> list[tuple[str, float, str, str]]:
    """Backwards-compatible single-file wrapper."""
    return _pick_topics_from_ideas_files([file_path], limit=limit)


def _pick_topics_from_ideas_files(file_paths: list[str], limit: int) -> list[tuple[str, float, str, str]]:
    """Extract curated micro/nano niche idea candidates from one or more files."""
    paths = [p for p in (file_paths or []) if str(p or "").strip()]
    if not paths:
        return []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"ideas file not found: {p}")

    # Parse each file independently, then merge & de-dupe.
    all_items: list[tuple[str, float, str, str]] = []
    for p in paths:
        all_items.extend(_pick_topics_from_single_ideas_file(p, limit=max(50, limit)))

    # De-dupe while preserving order; prefer first occurrence.
    seen: set[str] = set()
    out: list[tuple[str, float, str, str]] = []
    for topic, score, niche, hook in all_items:
        key = str(topic or "").strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append((topic, float(score), niche or "", hook or ""))
        if len(out) >= limit:
            break
    return out


def _pick_topics_from_single_ideas_file(file_path: str, limit: int) -> list[tuple[str, float, str, str]]:
    """Extract idea candidates from a single ideas file.

    Supports:
      - Markdown tables (|...|)
      - Bullet lists (- Idea ...)
      - Plain-paragraph ideas (one idea per paragraph; common in 'winner post part 2')
      - Inline numbered lists inside a long line (e.g., "...: 1. ... 2. ...")
    """
    raw = open(file_path, "r", encoding="utf-8", errors="replace").read()
    raw = raw.replace("\ufeff", "").replace("\u200b", "")
    lines = raw.splitlines()

    base = os.path.basename(file_path).lower()
    enable_paragraphs = True
    # Part 1 is primarily rubric/spec; avoid selecting explanatory paragraphs as ideas.
    if "part 1" in base and "winner post" in base:
        enable_paragraphs = False

    def _maybe_parse_score(text: str) -> float | None:
        # Examples: "T:9 D:9 LC:8 US:9 → 8.75" or "8.5" or "Score: 9.25"
        s = str(text or "").strip()
        if not s:
            return None
        m = re.search(r"(?:→|=|:)\s*(\d+(?:\.\d+)?)\s*$", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        m2 = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
        if m2:
            try:
                return float(m2.group(1))
            except Exception:
                return None
        return None

    def _parse_markdown_tables() -> list[tuple[str, float, str, str]]:
        """Parse idea rows from markdown tables.

        Supports common columns like:
          - Micro-Niche / Nano-Niche
          - Example Hook / 3 Hook Openers
          - Score
        """
        out: list[tuple[str, float, str, str]] = []

        def split_row(row: str) -> list[str]:
            parts = [p.strip() for p in row.strip().strip("|").split("|")]
            return parts

        i = 0
        while i < len(lines):
            ln = lines[i]
            if not ln.strip().startswith("|"):
                i += 1
                continue

            # Find header row + separator row
            header = ln
            j = i + 1
            if j >= len(lines):
                i += 1
                continue
            sep = lines[j]
            if not (sep.strip().startswith("|") and re.match(r"^\|\s*[-: ]+\|", sep.strip())):
                # Not a standard markdown table
                i += 1
                continue

            cols = split_row(header)
            # Build column lookup
            col_map: dict[str, int] = {}
            for idx, c in enumerate(cols):
                key = re.sub(r"\s+", " ", c.lower()).strip()
                if key:
                    col_map[key] = idx

            def find_col(*needles: str) -> int | None:
                for k, idx in col_map.items():
                    for n in needles:
                        if n in k:
                            return idx
                return None

            topic_col = find_col("nano-niche", "nano niche", "micro-niche", "micro niche", "micro-niche (", "nano-niche (")
            hook_col = find_col("example hook", "hook openers", "hook")
            score_col = find_col("score")

            # Walk subsequent rows until table ends
            k = j + 1
            while k < len(lines) and lines[k].strip().startswith("|"):
                row = lines[k].strip()
                if re.match(r"^\|\s*[-: ]+\|", row):
                    k += 1
                    continue

                cells = split_row(row)
                if not cells or len(cells) < 2:
                    k += 1
                    continue

                def cell(idx: int | None) -> str:
                    if idx is None:
                        return ""
                    if idx < 0 or idx >= len(cells):
                        return ""
                    return str(cells[idx] or "").strip()

                topic = _strip_md_links(cell(topic_col))
                topic = re.sub(r"\s+", " ", topic).strip().rstrip(".")
                if topic and len(topic) >= 18:
                    hook_raw = _strip_md_links(cell(hook_col))
                    hook_raw = re.sub(r"\s+", " ", hook_raw).strip()
                    # If multiple hooks are present separated by ';', pick the first.
                    hook = hook_raw.split(";")[0].strip().strip('"') if hook_raw else ""

                    score = _maybe_parse_score(cell(score_col))
                    if score is None:
                        score = 9.2

                    out.append((topic, float(score), "ideas_file_table", hook))
                    if len(out) >= max(20, limit):
                        # Don't early break too hard; caller will cap. But keep a safety cap.
                        pass

                k += 1

            i = k

        return out

    def _split_inline_numbered_list(text: str) -> list[str]:
        # Extract items in a compact numbered sequence: "...: 1. item 2. item ..."
        s = str(text or "")
        hits = list(re.finditer(r"\b(\d{1,3})\.\s+", s))
        if len(hits) < 10:
            return []
        items: list[str] = []
        for idx, m in enumerate(hits):
            start = m.end()
            end = hits[idx + 1].start() if idx + 1 < len(hits) else len(s)
            chunk = s[start:end]
            chunk = re.sub(r"\s+", " ", chunk).strip().rstrip(".")
            if len(chunk) >= 18:
                items.append(chunk)
        return items

    def _extract_plain_paragraph_ideas(raw_text: str) -> list[str]:
        """Extract one-idea-per-paragraph blocks (common in part 2 files)."""
        blocks = re.split(r"\n\s*\n+", raw_text)
        out: list[str] = []

        # Soft keyword gate to avoid pulling explanatory paragraphs (esp. from part 1/spec files).
        idea_kw = re.compile(
            r"\b(zone|clay|raised\s+bed|raised\s+beds|garden|gardening|balcony|vertical|trellis|compost|keyhole|microgreen|led|herb|herbs|mushroom|mycelium|substrate|monotub|shiitake|oyster|lion'?s\s+mane|biochar|fiber|hemp|flax|bamboo|jute|sisal|cold\s+frame|row\s+cover)\b",
            re.IGNORECASE,
        )

        def looks_like_heading(s: str) -> bool:
            if not s:
                return True
            low = s.lower().strip()
            if low in (
                "micro niche topics",
                "nano niche topics",
                "comparison table",
                "mushroom farming micro niches",
                "herbs micro niches",
                "mycelium-based materials",
                "fiber plant materials",
                "other plant-based crafts",
                "other plant composites",
            ):
                return True
            if low.startswith((
                "tìm các micro niche",
                "write in english only",
                "micro niche and nano niche topics similar",
                "micro niche ideas",
                "nano niche ideas",
                "gửi tui prompt",
                "context:",
                "quy tắc",
                "next time",
                "prompt nâng cao",
                "full tự động",
                "cách test",
                "rồi ví dụ",
            )):
                return True
            if low.startswith("here are ") and "additional micro niche ideas" in low:
                return True
            # Section headers often have no punctuation and are short.
            if (len(s) <= 70) and ("." not in s) and ("," not in s) and (" – " not in s) and ("-" not in s):
                return True
            # Headings like "Mycelium & Fungi Composites (1-10)"
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9 '&\\/]+\(\d+\s*-\s*\d+\)\s*$", s.strip()):
                return True
            return False

        for b in blocks:
            if not b or not b.strip():
                continue

            # Drop table-like blocks (tab-separated)
            if "\t" in b and b.count("\t") >= 2:
                continue

            s = re.sub(r"\s+", " ", b.strip())
            s = _strip_md_links(s)
            s = re.sub(r"\s+", " ", s).strip()
            s = s.strip("-• ").strip().rstrip(". ").rstrip("—–- ")

            if not s or len(s) < 18:
                continue
            if looks_like_heading(s):
                continue

            # Skip obvious table headers / schema rows
            low = s.lower()
            if low.startswith(("niche level", "category", "material source")):
                continue
            if "keyword idea" in low and "specificity" in low:
                continue

            # Keep ideas reasonably sized; allow longer "winner pattern" lines, but drop huge essays.
            if len(s) > 420:
                continue

            low = s.lower()
            # Drop meta/explanatory paragraphs that describe the idea list itself.
            # These frequently appear in part-2 files and should never be published.
            if low.startswith(("micro niche ideas", "nano niche ideas")):
                continue
            # Require at least one domain keyword OR digits (e.g., "20 ... designs") to be considered an idea.
            if not idea_kw.search(s) and not re.search(r"\d", s):
                continue

            out.append(s)
        return out

    def _extract_delimited_direct_posts() -> list[str]:
        """Extract full multi-paragraph answer blocks separated by delimiter lines.

        Supports lines like:
          - "_____" (5+ underscores)
          - "=====" (5+ equals)
          - "-----" (5+ dashes)

        Intended use: user pastes a GenAI answer and separates posts with a delimiter.
        """
        delims = re.compile(r"^(?:_{5,}|={5,}|-{5,})\s*$")
        chunks: list[list[str]] = []
        cur: list[str] = []
        found_delim = False
        for ln in raw.splitlines():
            if delims.match(str(ln or "").strip()):
                found_delim = True
                if cur:
                    chunks.append(cur)
                    cur = []
                continue
            cur.append(ln)
        if cur:
            chunks.append(cur)

        # If there are no delimiters, do NOT treat the entire file as a single direct post.
        if not found_delim:
            return []

        posts: list[str] = []
        for ch in chunks:
            txt = "\n".join(ch).strip()
            if not txt or len(txt) < 600:
                continue

            # Heuristic: multi-section answer tends to have separators or emoji headings.
            sep_cnt = txt.count("⸻") + txt.count("---")
            emoji_head_cnt = len(re.findall(r"(?m)^[^A-Za-z0-9\s]{1,3}\s*\w", txt))
            if sep_cnt < 2 and emoji_head_cnt < 3:
                continue

            cleaned = _clean_direct_answer_text(txt)
            if cleaned and len(cleaned) >= 400:
                posts.append(cleaned)

        return posts

    def _extract_qa_direct_posts() -> list[tuple[str, str]]:
        """Extract direct-post blocks from a Q/A style log.

        Supported markers:
          - "tui:" / "user:" for the question
          - "genai:" / "assistant:" for the answer

        This matches the workflow: user asks → GenAI answers → copy/paste to file.
        """
        q_pat = re.compile(r"^\s*(?:tui|user)\s*:\s*", re.IGNORECASE)
        a_pat = re.compile(r"^\s*(?:gen\s*ai|genai|assistant)\s*:\s*", re.IGNORECASE)

        answers: list[tuple[str, str]] = []
        in_answer = False
        cur: list[str] = []
        cur_q: str = ""

        def flush() -> None:
            nonlocal cur, cur_q
            if not cur:
                return
            txt = "\n".join(cur).strip()
            cleaned = _clean_direct_answer_text(txt)
            if cleaned and len(cleaned) >= 240:
                q = (cur_q or "").strip()
                if not q:
                    q = _title_from_text(cleaned)
                answers.append((q, cleaned))
            cur = []
            cur_q = ""

        for ln in raw.splitlines():
            s = str(ln or "")
            if q_pat.match(s):
                # New question starts → end current answer
                if in_answer:
                    flush()
                in_answer = False
                cur_q = q_pat.sub("", s).strip()
                continue
            if a_pat.match(s):
                # Start of answer
                if in_answer:
                    flush()
                in_answer = True
                # Keep remainder of the line after marker
                s2 = a_pat.sub("", s).strip()
                if s2:
                    cur.append(s2)
                continue
            if in_answer:
                cur.append(s)

        if in_answer:
            flush()

        return answers

    def _extract_qa_direct_posts_with_mask() -> tuple[list[tuple[str, str]], list[bool]]:
        """Extract Q/A posts and a mask of lines consumed by those answers.

        We treat the answer as running until the next `tui:` marker.
        Additionally, for mixed files (like your part 2), we allow an early stop when
        we hit a known "reset" heading after a long answer so the rest of the file
        can still be parsed as normal ideas.
        """
        q_pat = re.compile(r"^\s*(?:tui|user)\s*:\s*", re.IGNORECASE)
        a_pat = re.compile(r"^\s*(?:gen\s*ai|genai|assistant)\s*:\s*", re.IGNORECASE)
        reset_pat = re.compile(r"^\s*(?:tìm\s+các\s+micro\s+niche|micro\s+niche\s+and\s+nano\s+niche\s+topics\s+similar\b)", re.IGNORECASE)

        posts: list[tuple[str, str]] = []
        mask = [False] * len(lines)

        cur_q = ""
        cur_a: list[str] = []
        in_answer = False

        def flush() -> None:
            nonlocal cur_q, cur_a, in_answer
            if not cur_a:
                cur_q = ""
                in_answer = False
                return
            txt = "\n".join(cur_a).strip()
            cleaned = _clean_direct_answer_text(txt)
            if cleaned and len(cleaned) >= 240:
                q = (cur_q or "").strip() or _title_from_text(cleaned)
                posts.append((q, cleaned))
            cur_q = ""
            cur_a = []
            in_answer = False

        for i, ln in enumerate(lines):
            s = str(ln or "")
            if q_pat.match(s):
                # Start a new question; close any open answer.
                if in_answer:
                    flush()
                cur_q = q_pat.sub("", s).strip()
                mask[i] = True
                continue

            if a_pat.match(s):
                # Start answer
                if in_answer:
                    flush()
                in_answer = True
                mask[i] = True
                s2 = a_pat.sub("", s).strip()
                if s2:
                    cur_a.append(s2)
                continue

            if in_answer:
                # Early-stop heuristic for mixed files: if we hit a reset heading after a long answer,
                # treat it as not part of the answer so the rest of the file is parseable.
                if reset_pat.match(s) and len(cur_a) >= 20:
                    flush()
                    # Do NOT mask this line; it's part of the regular ideas file.
                    continue

                cur_a.append(s)
                mask[i] = True

        if in_answer:
            flush()

        return posts, mask

    # 0) Direct-post blocks (full GenAI answers) — highest priority
    direct_posts: list[str] = []
    direct_post_map: dict[str, str] = {}
    qa_mask: list[bool] = [False] * len(lines)
    if enable_paragraphs:
        # Prefer Q/A blocks (fully automatic from your "tui:" / "genAI:" workflow)
        qa_posts, qa_mask = _extract_qa_direct_posts_with_mask()
        if qa_posts:
            for q, ans in qa_posts:
                q2 = re.sub(r"\s+", " ", str(q or "").strip())
                if not q2 or len(q2) < 8:
                    continue
                direct_posts.append(q2)
                direct_post_map[q2.lower()] = ans
        # Optional: delimiter-based blocks for power users
        if not direct_posts:
            del_posts = _extract_delimited_direct_posts()
            # For delimiter-based blocks (no question), use title as the topic key.
            for ans in del_posts:
                title = _title_from_text(ans)
                if not title:
                    continue
                direct_posts.append(title)
                direct_post_map[title.lower()] = ans

    # Build filtered text for non-direct extraction (remove Q/A blocks if present).
    filtered_lines = [ln for i, ln in enumerate(lines) if not qa_mask[i]]
    filtered_raw = "\n".join(filtered_lines)
    filtered_lines_list = filtered_raw.splitlines()

    # 1) Table-first parsing (some files store hundreds of ideas as rows)
    # Use filtered lines so we don't parse tables inside a direct-post answer.
    lines = filtered_lines_list
    raw = filtered_raw
    table_items = _parse_markdown_tables()

    # Start scanning from the micro-niche ideas area if present.
    start_idx = 0
    for i, ln in enumerate(lines):
        if "micro niche idea" in ln.lower() and "plant" in ln.lower():
            start_idx = i
            break
        if "mycelium-based materials" in ln.lower():
            start_idx = i
            break

    bullet_candidates: list[str] = []
    for ln in lines[start_idx:]:
        # Stop once we leave the ideas block (comparison table / checklist etc.)
        if ln.strip().lower().startswith("## comparison table"):
            break
        if ln.strip().startswith("|"):
            continue  # skip tables
        if not ln.lstrip().startswith("-"):
            continue

        # Skip Q/A markers if any made it through
        if ln.strip().lower().startswith(("tui:", "user:", "genai:", "gen ai:", "assistant:")):
            continue

        idea = ln.lstrip().lstrip("-").strip()
        if not idea:
            continue

        idea = _strip_md_links(idea)
        idea = re.sub(r"\s+", " ", idea).strip()
        idea = idea.rstrip(".")

        # Filter out ultra-short bullets
        if len(idea) < 18:
            continue

        bullet_candidates.append(idea)

    # 2) Paragraph-style idea parsing (winner post part 2 is often written like this)
    paragraph_candidates: list[str] = []
    if enable_paragraphs:
        paragraph_candidates = _extract_plain_paragraph_ideas(raw)

    # 3) Inline numbered list extraction (common in copied “25 layouts” style blocks)
    inline_numbered: list[str] = []
    for ln in lines:
        if ln.strip().lower().startswith(("tui:", "user:", "genai:", "gen ai:", "assistant:")):
            continue
        inline_numbered.extend(_split_inline_numbered_list(ln))

    # Merge: direct posts first, then table items, then bullets, then inline numbered, then paragraphs
    merged: list[str] = []
    merged.extend(direct_posts)
    merged.extend([t for (t, *_rest) in table_items])
    merged.extend(bullet_candidates)
    merged.extend(inline_numbered)
    merged.extend(paragraph_candidates)

    # If this is part 2 and we still have too few candidates, add a looser line-based pass
    # to reach "a few hundred" ideas when the file contains many one-liners.
    if ("part 2" in base) and (len(merged) < 300):
        loose: list[str] = []
        for ln in lines:
            s = str(ln or "").strip()
            if not s or len(s) < 18 or len(s) > 220:
                continue
            low = s.lower()
            if low.startswith(("tui:", "user:", "genai:", "gen ai:", "assistant:")):
                continue
            if low in ("micro niche topics", "nano niche topics", "comparison table"):
                continue
            if low.startswith(("write in english only", "context:", "quy tắc", "gửi tui prompt")):
                continue
            if s.startswith("|"):
                continue
            if "\t" in s and s.count("\t") >= 2:
                continue
            # Keep sentence-like lines
            if (s.endswith(".") or s.endswith(")") or ("," in s) or ("–" in s) or ("-" in s)):
                loose.append(_strip_md_links(s).strip().rstrip("."))
        merged.extend(loose)

    # De-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for c in merged:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
        if len(uniq) >= limit:
            break

    # Return (topic, score, niche, hook)
    # If topic came from a table row, try to return its specific score/hook.
    tmap = {t.lower(): (t, score, niche, hook) for (t, score, niche, hook) in table_items}
    out: list[tuple[str, float, str, str]] = []
    for t in uniq:
        # Direct post: topic is the Q (idea), hook stores the full answer to publish.
        d_ans = direct_post_map.get(str(t or "").strip().lower())
        if d_ans:
            out.append((str(t).strip(), 9.7, "ideas_direct_post", d_ans))
            continue
        item = tmap.get(t.lower())
        if item:
            out.append(item)
        else:
            out.append((t, 9.2, "ideas_file_text", ""))
    return out


def _seed_keywords_from_ideas(ideas: list[str]) -> set[str]:
    seeds: set[str] = set()
    for idea in ideas:
        seeds |= _norm_tokens(idea)
    return seeds


def _rank_db_candidates_by_seed(
    db_candidates: list[tuple[str, float, str, str]],
    seed_keywords: set[str],
) -> list[tuple[str, float, str, str]]:
    if not seed_keywords:
        return db_candidates

    def score_item(item: tuple[str, float, str, str]) -> float:
        topic = item[0]
        tset = _norm_tokens(topic)
        if not tset:
            return 0.0
        return len(tset & seed_keywords) / max(1, len(tset))

    # Prefer higher keyword overlap; stable tie-break by original order
    return sorted(db_candidates, key=score_item, reverse=True)


def _norm_tokens(text: str) -> set[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    toks = [t for t in re.split(r"\s+", text) if t]
    # Drop very short tokens (noise)
    return {t for t in toks if len(t) >= 4}


def _overlap_ratio(topic: str, hay: str) -> float:
    """How much of topic's keyword set appears in hay (0-1)."""
    tset = _norm_tokens(topic)
    if not tset:
        return 0.0
    hset = _norm_tokens(hay)
    return len(tset & hset) / max(1, len(tset))


def _jaccard(a: str, b: str) -> float:
    sa = _norm_tokens(a)
    sb = _norm_tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _shared_count(a: str, b: str) -> int:
    return len(_norm_tokens(a) & _norm_tokens(b))


async def _fetch_recent_publer_text(pages: int, limit: int, status: str = "") -> list[str]:
    """Fetch recent Publer posts text across N pages."""
    from integrations.publer_publisher import PublerPublisher

    pub = PublerPublisher()
    await pub.connect()
    texts: list[str] = []
    for page in range(1, pages + 1):
        posts = await pub.get_posts(status=status, limit=limit, page=page)
        for p in posts:
            title = str(p.get("title", "") or "")
            body = str(p.get("text", p.get("body", "") or "") or "")
            # Keep it bounded for speed
            texts.append((title + "\n" + body[:800]).strip())
        await asyncio.sleep(0.2)
    return [t for t in texts if t]


async def _dedup_against_publer(
    candidates: list[tuple[str, float, str, str]],
    limit: int,
    pages: int,
    per_page: int,
    min_overlap: float,
    min_shared: int,
    min_jaccard: float,
) -> list[tuple[str, float, str, str]]:
    """Filter out candidates that look too similar to recent Publer posts."""
    recent = await _fetch_recent_publer_text(pages=pages, limit=per_page)
    if not recent:
        return candidates[:limit]

    kept: list[tuple[str, float, str, str]] = []
    for topic, score, niche, hook in candidates:
        is_dup = False
        for txt in recent:
            # Primary: dedup by topic/idea string (most stable across edits).
            if _overlap_ratio(topic, txt) >= min_overlap:
                is_dup = True
                break
            if _shared_count(topic, txt) >= min_shared and _jaccard(topic, txt) >= min_jaccard:
                is_dup = True
                break

            # Secondary: for long direct-post answers (hook), also check overlap against the answer
            # so we don't repost nearly identical content when the topic wording changes.
            if hook and len(hook) >= 240:
                # Use stricter thresholds for long text to reduce false positives.
                if _shared_count(hook, txt) >= max(min_shared + 2, 6) and _jaccard(hook, txt) >= min_jaccard:
                    is_dup = True
                    break
        if is_dup:
            # Backfill local history so future runs skip even if Publer dedup is off / shallow.
            try:
                _mark_topic_used(topic, source="publer_dedup")
            except Exception:
                pass
            continue
        kept.append((topic, score, niche, hook))
        if len(kept) >= limit:
            break

    return kept


async def _generate_content(topic: str, score: float, *, idea_line: str = "") -> dict[str, Any]:
    from llm_content import generate_quality_post

    # Season is computed by month inside generate_quality_post().
    # Optional override: set VIRALOPS_SEASON=Winter|Spring|Summer|Fall
    season_override = os.environ.get("VIRALOPS_SEASON", "").strip()
    pack = await asyncio.to_thread(
        generate_quality_post,
        topic=topic,
        score=score,
        season=season_override,
        idea_line=idea_line,
    )
    if not pack or not pack.get("content_formatted"):
        raise ValueError(f"Failed generate: {topic[:80]}")
    return pack


async def _publish_tiktok(cp: dict[str, Any], account_id: str, label: str) -> dict:
    from web.app import _prepare_tiktok_content
    from integrations.publer_publisher import PublerPublisher

    pc = await _prepare_tiktok_content(cp, "tiktok")
    pc["account_ids"] = [account_id]
    pc["_account_label"] = label
    pub = PublerPublisher()
    r = await pub.publish(pc)
    if not isinstance(r, dict):
        return {"success": False, "error": str(r)}

    # Simple collision guard: if TikTok requires a 1-minute gap, wait and retry once.
    err0 = str(r.get("error", "") or "").lower()
    if (not r.get("success")) and ("one minute gap" in err0 or "another post at this time" in err0):
        try:
            gap = int(os.environ.get("VIRALOPS_TIKTOK_MIN_GAP_SECS", "65") or "65")
        except Exception:
            gap = 65
        await asyncio.sleep(max(5, gap))
        r = await pub.publish(pc)
        if not isinstance(r, dict):
            return {"success": False, "error": str(r)}

    # TikTok OpenAPI spam guard: if too many posts in last 24h, auto-schedule.
    err = str(r.get("error", "") or "")
    err_l = err.lower()
    is_tiktok_limit = (
        ("tiktok" in err_l and "openapi" in err_l)
        or ("too many posts" in err_l and "24 hours" in err_l)
        or ("blocked" in err_l and "minimize spam" in err_l)
    )
    if (not r.get("success")) and is_tiktok_limit:
        hours = float(os.environ.get("VIRALOPS_TIKTOK_SCHEDULE_FALLBACK_HOURS", "25") or "25")
        base = datetime.now(timezone.utc) + timedelta(hours=hours)
        # Stagger schedules per account to avoid same-minute collisions.
        try:
            offset_min = int(os.environ.get("VIRALOPS_TIKTOK_SCHEDULE_STAGGER_MIN", "2") or "2")
        except Exception:
            offset_min = 2
        # Deterministic offset from account_id
        acct_bump = sum(ord(c) for c in str(account_id)) % max(1, offset_min)

        last: dict | None = None
        for attempt in range(4):
            schedule_at = (base + timedelta(minutes=acct_bump + attempt * 2)).replace(microsecond=0).isoformat()
            pc2 = dict(pc)
            pc2["schedule_at"] = schedule_at
            r2 = await pub.publish(pc2)
            if isinstance(r2, dict) and r2.get("success"):
                r2["_fallback"] = "scheduled_due_to_tiktok_openapi_limit"
                r2["_scheduled_at"] = schedule_at
                return r2
            last = r2 if isinstance(r2, dict) else {"success": False, "error": str(r2)}
            err2 = str((last or {}).get("error", "") or "").lower()
            if "one minute gap" not in err2 and "another post at this time" not in err2:
                break
            await asyncio.sleep(0.5)

        return last or {"success": False, "error": "schedule fallback failed"}

    return r


async def _publish_facebook(cp: dict[str, Any]) -> dict:
    from web.app import _prepare_facebook_content
    from integrations.publer_publisher import PublerPublisher

    pc = await _prepare_facebook_content(cp)
    if not pc:
        return {"success": False, "error": "FB empty"}
    pc["platforms"] = ["facebook"]

    pub = PublerPublisher()
    await pub.connect()

    # 1) Explicit override
    fb_override = os.environ.get("VIRALOPS_FACEBOOK_ACCOUNT_ID", "").strip()
    if fb_override:
        pc["account_ids"] = [fb_override]
    else:
        # 2) Discover currently-connected FB accounts in Publer workspace
        fb_ids = await pub.get_account_ids("facebook")
        if fb_ids:
            pc["account_ids"] = [fb_ids[0]]
        elif FACEBOOK:
            # 3) Last resort legacy ID
            pc["account_ids"] = [FACEBOOK]

    r = await pub.publish(pc)
    if isinstance(r, dict) and (not r.get("success")):
        err = str(r.get("error", "") or "")
        if "composer is in a bad state" in err.lower():
            r["error"] = (
                err
                + " | FIX: In Publer dashboard → Accounts, reselect/refresh the Facebook Page connection (disconnect/reconnect if needed)."
            )
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def _publish_pinterest(cp: dict[str, Any]) -> dict:
    from web.app import _prepare_pinterest_content
    from integrations.publer_publisher import PublerPublisher

    # Pass alt text through to the Pinterest formatter (web/app.py) without
    # polluting the top of the pin description with an "ALT:" prefix.
    alt = str(
        cp.get("image_alt")
        or cp.get("image_alt_text")
        or cp.get("title")
        or cp.get("_topic")
        or ""
    ).strip()
    alt = re.sub(r"\s+", " ", alt)[:150] if alt else ""
    if alt:
        cp = dict(cp)
        cp["image_alt_text"] = alt

    pc = await _prepare_pinterest_content(cp)
    if not pc:
        return {"success": False, "error": "Pinterest no image"}
    pc["platforms"] = ["pinterest"]

    pub = PublerPublisher()
    await pub.connect()

    # 1) Explicit override
    pin_override = os.environ.get("VIRALOPS_PINTEREST_ACCOUNT_ID", "").strip()
    if pin_override:
        pc["account_ids"] = [pin_override]
    else:
        # 2) Discover currently-connected Pinterest accounts in Publer workspace
        pin_ids = await pub.get_account_ids("pinterest")
        if pin_ids:
            pc["account_ids"] = [pin_ids[0]]
        elif PINTEREST:
            # 3) Last resort legacy ID
            pc["account_ids"] = [PINTEREST]

    # Pinterest requires selecting an album/board.
    album_override = os.environ.get("VIRALOPS_PINTEREST_ALBUM_ID", "").strip()
    if album_override:
        pc["album_id"] = album_override
    else:
        try:
            accounts = await pub.get_accounts()
            target_id = (pc.get("account_ids") or [""])[0]
            acc = next((a for a in (accounts or []) if str(a.get("id", a.get("_id", ""))) == str(target_id)), None)
            albums = (acc or {}).get("albums") or []
            if albums and isinstance(albums, list) and isinstance(albums[0], dict) and albums[0].get("id"):
                pc["album_id"] = str(albums[0].get("id"))
        except Exception:
            pass

    r = await pub.publish(pc)
    # Best-effort: fetch the created post objects so we can print the Pinterest pin link.
    # Publer can be eventually consistent; poll briefly.
    try:
        if isinstance(r, dict) and r.get("success") and r.get("created_post_ids"):
            created_ids = [str(x) for x in (r.get("created_post_ids") or []) if str(x).strip()]
            if created_ids:
                created_posts: list[dict[str, Any]] = []

                for _attempt in range(12):
                    if created_posts:
                        break
                    # Scan a few pages of recent posts.
                    found: list[dict[str, Any]] = []
                    for page in (1, 2, 3, 4, 5):
                        recent = await pub.get_posts(limit=50, page=page)
                        for cid in created_ids[:5]:
                            p = next((pp for pp in (recent or []) if str(pp.get("id")) == cid), None)
                            if not isinstance(p, dict):
                                continue
                            found.append(
                                {
                                    "id": p.get("id"),
                                    "url": p.get("url"),
                                    "post_link": p.get("post_link"),
                                    "state": p.get("state"),
                                }
                            )
                    # De-dupe by id
                    seen_ids: set[str] = set()
                    for it in found:
                        iid = str(it.get("id") or "")
                        if not iid or iid in seen_ids:
                            continue
                        seen_ids.add(iid)
                        created_posts.append(it)

                    if not created_posts:
                        await asyncio.sleep(1.0)

                if created_posts:
                    r["created_posts"] = created_posts
                    # Convenience: expose first post_link and url
                    first = created_posts[0] if created_posts else {}
                    first_link = str((first or {}).get("post_link") or "").strip()
                    first_url = str((first or {}).get("url") or "").strip()
                    if first_link:
                        r["post_link"] = first_link
                    if first_url:
                        r["url"] = first_url
    except Exception:
        pass

    # Fallback: match the created Pinterest post by destination URL.
    # This covers cases where Publer doesn't return created_post_ids or the post isn't
    # immediately visible for id-based lookup.
    try:
        if isinstance(r, dict) and r.get("success"):
            if not str(r.get("post_link") or "").strip():
                desired_url = str(pc.get("url") or "").strip()
                if desired_url:
                    for _attempt in range(18):
                        match = None
                        for page in (1, 2, 3, 4, 5):
                            recent = await pub.get_posts(limit=50, page=page)
                            match = next(
                                (
                                    p
                                    for p in (recent or [])
                                    if isinstance(p, dict)
                                    and str(p.get("url") or "").strip() == desired_url
                                ),
                                None,
                            )
                            if isinstance(match, dict):
                                break
                        if isinstance(match, dict):
                            r["url"] = match.get("url")
                            r["matched_post_id"] = match.get("id")
                            link = str(match.get("post_link") or "").strip()
                            if link:
                                r["post_link"] = link
                            break
                        await asyncio.sleep(1.0)
    except Exception:
        pass
    if isinstance(r, dict) and (not r.get("success")):
        err = str(r.get("error", "") or "")
        if "composer is in a bad state" in err.lower():
            r["error"] = (
                err
                + " | FIX: In Publer dashboard → Accounts, reselect/refresh the Pinterest Business connection (disconnect/reconnect if needed)."
            )
        # Some Publer setups reject content_type="pin" for Pinterest; retry with alternatives.
        if ("post type is not valid" in err.lower()) or ("album can't be blank" in err.lower()):
            for ct in ("photo", "status", "video"):
                pc2 = dict(pc)
                pc2["content_type"] = ct
                r2 = await pub.publish(pc2)
                if isinstance(r2, dict) and r2.get("success"):
                    r2["_fallback"] = f"pinterest_content_type_{ct}"
                    return r2
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


def _build_shopify_article_from_pack(cp: dict[str, Any], *, draft: bool) -> dict[str, Any]:
    """Best-effort map a ViralOps content pack to Shopify Article fields.

    Shopify Admin API requires: title + body_html.
    We also try to fill common Shopify UI fields: excerpt/summary, tags, author, SEO.
    """
    title = str(cp.get("title") or "").strip()
    if not title:
        title = str(cp.get("_topic") or "").strip()

    # Prefer an explicit HTML field if present.
    body_html = (
        cp.get("body_html")
        or cp.get("article_html")
        or cp.get("blog_html")
        or ""
    )

    if not body_html:
        # Convert markdown-ish text to simple HTML paragraphs.
        src = str(cp.get("content_formatted") or cp.get("universal_caption_block") or cp.get("body") or "").strip()
        # Strip bare URLs from body; we want Pinterest to carry the destination link.
        src = re.sub(r"https?://\S+", "", src)
        parts = [ln.strip() for ln in src.splitlines() if ln.strip()]
        body_html = "\n".join(f"<p>{ln}</p>" for ln in parts)

    def _clean_meta(s: str) -> str:
        s = re.sub(r"<[^>]+>", " ", s or "")
        s = re.sub(r"https?://\S+", "", s)
        # Remove most emoji/symbol codepoints for SERP stability
        s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = s.replace('"', "").replace("'", "")
        return s

    def _make_meta_description() -> str:
        # Prefer explicit seo_description if it's already good.
        candidate = _clean_meta(str(cp.get("seo_description") or "").strip())
        if 80 <= len(candidate) <= 160:
            return candidate[:155]

        hook = _clean_meta(str(cp.get("hook") or "").strip())
        step_1 = _clean_meta(str(cp.get("step_1") or "").strip())
        step_2 = _clean_meta(str(cp.get("step_2") or "").strip())
        result = _clean_meta(str(cp.get("result") or "").strip())
        micro = _clean_meta(str(cp.get("micro_keywords") or "").strip())

        kws: list[str] = []
        if micro:
            for t in re.split(r"[•,\n]+", micro):
                tt = t.strip()
                if tt and len(tt) >= 3:
                    kws.append(tt)
                if len(kws) >= 2:
                    break

        base = hook or _clean_meta(title)

        parts: list[str] = []
        if base:
            parts.append(base.rstrip(".!?") + ".")
        steps = ", ".join([p for p in [step_1, step_2] if p])
        if steps:
            parts.append(f"Step-by-step: {steps}.")
        if result:
            parts.append(result.rstrip(".!?") + ".")
        if kws:
            parts.append("Keywords: " + ", ".join(kws) + ".")

        meta = _clean_meta(" ".join(parts).strip())
        if len(meta) > 155:
            meta = meta[:155].rsplit(" ", 1)[0]
        if len(meta) < 70:
            meta = _clean_meta(title)[:70] + ". " + meta
            meta = meta[:155].rsplit(" ", 1)[0] if len(meta) > 155 else meta
        return meta

    seo_description = _make_meta_description()

    summary_html = str(cp.get("summary_html") or "").strip()
    if not summary_html and seo_description:
        summary_html = f"<p>{seo_description}</p>"

    # Tags
    tags: list[str] = []
    micro = str(cp.get("micro_keywords") or "").strip()
    if micro:
        for t in re.split(r"[•,\n]+", micro):
            tt = t.strip()
            if tt:
                tags.append(tt)
    niche = str(cp.get("_niche") or cp.get("niche") or "").strip()
    if niche:
        tags.append(niche)
    tags = [t[:60] for t in tags if t][:20]

    author = str(
        os.environ.get("SHOPIFY_AUTHOR", "")
        or os.environ.get("VIRALOPS_SHOPIFY_AUTHOR", "")
        or os.environ.get("BLOG_AUTHOR", "")
        or os.environ.get("AUTHOR", "")
    ).strip()

    # Image handling
    # - Shopify featured image needs a PUBLIC URL; if we only have a local file, we upload it after article creation.
    image_url = ""
    image_local_path = ""
    candidate = str(cp.get("image_url") or cp.get("media_url") or "").strip()
    if candidate.startswith(("http://", "https://")):
        image_url = candidate
    else:
        local = str(cp.get("_ai_image_path") or cp.get("media_local_path") or "").strip()
        if local and os.path.isfile(local):
            image_local_path = local

    image_alt = _clean_meta(str(cp.get("image_alt") or title).strip())
    if not image_alt:
        image_alt = _clean_meta(title)

    out: dict[str, Any] = {
        "title": title,
        "body_html": body_html,
        "summary_html": summary_html,
        "tags": ", ".join(tags) if tags else "",
        "author": author,
        "seo_title": title[:70],
        "seo_description": seo_description,
        "published": (not draft),
        "image_alt": image_alt,
    }
    if image_url:
        out["image_url"] = image_url
    if image_local_path:
        out["image_local_path"] = image_local_path
    return out


async def _publish_shopify_blog(cp: dict[str, Any], *, draft: bool) -> dict:
    """Create a Shopify blog article and return its public URL so Pinterest can match 1-1."""
    from integrations.shopify_blog_publisher import ShopifyBlogPublisher
    from core.models import QueueItem

    shopify_account_id = os.environ.get("VIRALOPS_SHOPIFY_ACCOUNT_ID", "shopify_viralops").strip() or "shopify_viralops"
    publisher = ShopifyBlogPublisher(account_id=shopify_account_id)
    ok = await publisher.connect()
    if not ok:
        return {"success": False, "error": "Shopify connect failed (check SHOPIFY_SHOP/SHOPIFY_ACCESS_TOKEN/SHOPIFY_BLOG_ID)"}

    content = _build_shopify_article_from_pack(cp, draft=draft)
    qi = QueueItem(
        id=f"shopify_{uuid.uuid4().hex[:10]}",
        content_pack_id=str(cp.get("id") or cp.get("content_pack_id") or "batch"),
        platform="shopify_blog",
    )

    try:
        res = await publisher.publish(qi, content)
    finally:
        try:
            await publisher.close()
        except Exception:
            pass

    if not getattr(res, "success", False):
        return {"success": False, "error": getattr(res, "error", "Shopify publish failed")}
    return {
        "success": True,
        "post_url": getattr(res, "post_url", ""),
        "post_id": getattr(res, "post_id", ""),
        "admin_url": (getattr(res, "metadata", {}) or {}).get("admin_url", ""),
        "handle": (getattr(res, "metadata", {}) or {}).get("handle", ""),
        "published": (getattr(res, "metadata", {}) or {}).get("published", True),
    }


async def _ensure_ai_image_in_pack(cp: dict[str, Any]) -> dict[str, Any]:
    """Ensure the pack has a local image path we can reuse across platforms.

    Why:
      - Shopify featured image (and inline injection) needs either a public URL
        or an attachment base64 — we use a local file path for attachment.
      - FB/Pinterest already generate images on-demand, but that generation is
        scoped to each formatter and doesn't flow back into the shared pack.
      - By generating once and storing it in `_ai_image_path`/`media_local_path`,
        Shopify/FB/Pinterest can all reuse the same image.
    """
    import tempfile
    import uuid as _uuid

    def _as_local_path(val: Any) -> str:
        v = str(val or "").strip()
        if not v:
            return ""
        if v.startswith(("http://", "https://")):
            return ""
        return v if os.path.isfile(v) else ""

    # If we already have a usable local file path, normalize keys and return.
    existing_local = (
        _as_local_path(cp.get("_ai_image_path"))
        or _as_local_path(cp.get("media_local_path"))
        or _as_local_path(cp.get("image_local_path"))
        or _as_local_path(cp.get("media_url"))
        or _as_local_path(cp.get("image_url"))
    )
    if existing_local:
        out = dict(cp)
        out["_ai_image_path"] = existing_local
        out["media_local_path"] = existing_local
        return out

    # If there's a public URL already, keep it as-is (Shopify can use image_url).
    existing_url = str(cp.get("image_url") or cp.get("media_url") or "").strip()
    if existing_url.startswith(("http://", "https://")):
        return cp

    prompt = str(cp.get("image_prompt") or "").strip()
    if not prompt:
        return cp

    try:
        from llm_content import generate_ai_image

        output_dir = tempfile.mkdtemp(prefix="viralops_shared_img_")
        # Use png (Pollinations/LLM generators commonly produce png) but the
        # generator may write jpeg depending on backend; we only care about path.
        output_path = os.path.join(
            output_dir, f"shared_{_uuid.uuid4().hex[:8]}.png"
        )
        img_path = await asyncio.to_thread(generate_ai_image, prompt, output_path)
        img_path = str(img_path or "").strip()
        if img_path and os.path.isfile(img_path):
            out = dict(cp)
            out["_ai_image_path"] = img_path
            out["media_local_path"] = img_path
            return out
    except Exception:
        pass

    return cp


async def _publish_all(
    cp: dict[str, Any],
    *,
    platforms: list[str],
    publish_shopify: bool,
    shopify_draft: bool,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    # Ensure Shopify has an image available. (FB/Pinterest can also reuse it.)
    if publish_shopify and ("shopify_blog" in platforms):
        cp = await _ensure_ai_image_in_pack(cp)

    if "tiktok" in platforms:
        # TikTok multi-account: publish DISTINCT variants per account to avoid
        # duplicate/copyright detection.
        from llm_content import make_tiktok_account_variant

        base_topic = str(cp.get("_topic") or cp.get("title") or "").strip() or "(unknown topic)"
        tiktok_accounts = await _get_tiktok_accounts()

        try:
            tiktok_gap = int(os.environ.get("VIRALOPS_TIKTOK_MIN_GAP_SECS", "65") or "65")
        except Exception:
            tiktok_gap = 65
        tiktok_gap = max(5, tiktok_gap)

        for idx, (account_id, label) in enumerate(tiktok_accounts, 1):
            variant_id = f"tiktok{idx}"
            cp_variant = make_tiktok_account_variant(
                cp,
                topic=base_topic,
                account_label=label,
                variant_id=variant_id,
            )
            # IMPORTANT: pass the explicit Publer account ID into the pack so `_prepare_tiktok_content()`
            # does not round-robin-pick a different account (and logs match the actual target).
            variant_pack = dict(cp_variant)
            variant_pack["account_ids"] = [account_id]
            variant_pack["_account_label"] = f"tiktok#{idx}"
            results[f"tiktok{idx}"] = await _publish_tiktok(variant_pack, account_id, label)
            # TikTok requires a 1-minute gap between posts; enforce a safe delay.
            if idx < len(tiktok_accounts):
                await asyncio.sleep(tiktok_gap)

    if "facebook" in platforms:
        results["facebook"] = await _publish_facebook(cp)
        await asyncio.sleep(2)

    # Optional: publish Shopify first, then attach the article URL to Pinterest.
    pin_pack = cp
    if publish_shopify and ("shopify_blog" in platforms):
        shop = await _publish_shopify_blog(cp, draft=shopify_draft)
        results["shopify_blog"] = shop
        if ("pinterest" in platforms):
            if shopify_draft:
                results["pinterest"] = {
                    "success": False,
                    "error": "Skipped Pinterest: Shopify article was created as draft (not public), so the destination link would 404.",
                }
                return results
            if shop.get("success") and shop.get("post_url"):
                pin_pack = dict(cp)
                pin_pack["destination_url"] = shop["post_url"]
                pin_pack["url"] = shop["post_url"]
            else:
                # Enforce 1-1 match: if Shopify failed, do NOT publish Pinterest with a wrong/missing link.
                results["pinterest"] = {
                    "success": False,
                    "error": "Skipped Pinterest: Shopify blog publish failed so destination link would not match.",
                }
                return results

    if "pinterest" in platforms:
        results["pinterest"] = await _publish_pinterest(pin_pack)

    return results


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="Actually publish. Default is dry-run.")
    ap.add_argument(
        "--skip-quality-gate",
        action="store_true",
        help="Publish even if the quality gate fails (useful for quick smoke tests).",
    )
    ap.add_argument(
        "--shopify",
        action="store_true",
        help="Also publish a Shopify blog article first, then set Pinterest destination link to the Shopify article URL.",
    )
    ap.add_argument(
        "--shopify-draft",
        action="store_true",
        help="Create Shopify article as draft (unpublished). Still uses the computed URL for Pinterest if available.",
    )
    ap.add_argument(
        "--platforms",
        type=str,
        default="tiktok,facebook,pinterest",
        help="Comma-separated platforms to publish. Example: shopify_blog,pinterest",
    )
    ap.add_argument(
        "--ideas-file",
        type=str,
        default=_auto_detect_default_ideas_file(),
        help=(
            "Optional: path to a Markdown file to source micro-niche idea bullet topics from. "
            "If provided (or auto-detected via VIRALOPS_IDEAS_FILE), topics are taken from this file first, "
            "then fall back to niche_hunter.db after ideas are exhausted."
        ),
    )
    ap.add_argument(
        "--ideas-reset-history",
        action="store_true",
        help="Reset (unmark) ideas-file topics in web/viralops.db topic_history so the curated ideas list can be reused.",
    )
    ap.add_argument(
        "--direct-only",
        action="store_true",
        help="Only publish direct-post blocks extracted from tui:/genAI: (skip normal ideas list).",
    )
    ap.add_argument(
        "--direct-reset-history",
        action="store_true",
        help="Reset (unmark) ideas_direct_post topics in web/viralops.db topic_history so tui:/genAI: blocks can be reused.",
    )
    ap.add_argument("--limit", type=int, default=3, help="How many topics to process.")
    ap.add_argument(
        "--force-topic",
        type=str,
        default="",
        help="Publish exactly this topic first (bypasses ideas/db picking). Useful for retries.",
    )
    ap.add_argument(
        "--retry-post-id",
        type=int,
        default=0,
        help="Retry publish from a stored row in web/viralops.db posts table (TikTok-only recommended).",
    )
    ap.add_argument("--min-score", type=float, default=8.5, help="Min niche score to accept.")
    ap.add_argument("--refresh-db", choices=["local", "full"], default=None, help="Refresh niche_hunter.db before picking topics.")
    ap.add_argument("--no-publer-dedup", action="store_true", help="Disable Publer-level dedup against recent posts.")
    ap.add_argument("--publer-pages", type=int, default=5, help="How many Publer pages to scan for dedup.")
    ap.add_argument("--publer-per-page", type=int, default=50, help="How many posts per page to fetch for dedup.")
    ap.add_argument(
        "--publer-min-overlap",
        type=float,
        default=0.65,
        help="Dedup threshold: fraction of topic keywords seen in a recent Publer post.",
    )
    ap.add_argument(
        "--publer-min-shared",
        type=int,
        default=3,
        help="Dedup threshold: minimum shared keywords with a recent Publer post.",
    )
    ap.add_argument(
        "--publer-min-jaccard",
        type=float,
        default=0.22,
        help="Dedup threshold: minimum Jaccard similarity (on keyword tokens) when shared-keywords threshold is met.",
    )
    args = ap.parse_args()

    raw_platforms = [p.strip().lower() for p in str(args.platforms or "").split(",") if p.strip()]
    # Allow a couple of friendly aliases.
    normalized: list[str] = []
    for p in raw_platforms:
        if p in ("shopify", "shopifyblog", "shopify_blog"):
            normalized.append("shopify_blog")
            continue
        if p in ("fb", "facebook"):
            normalized.append("facebook")
            continue
        if p in ("pin", "pinterest"):
            normalized.append("pinterest")
            continue
        if p in ("tt", "tiktok"):
            normalized.append("tiktok")
            continue
        normalized.append(p)
    # De-dupe while keeping order.
    platforms: list[str] = []
    for p in normalized:
        if p not in platforms:
            platforms.append(p)

    # If --shopify is enabled, ensure shopify_blog is included and that it runs before pinterest (for link matching).
    if args.shopify and ("shopify_blog" not in platforms):
        platforms = ["shopify_blog"] + platforms
    if "shopify_blog" in platforms and "pinterest" in platforms:
        # Ensure order: shopify_blog then pinterest.
        platforms = [p for p in platforms if p not in ("shopify_blog", "pinterest")] \
            + ["shopify_blog", "pinterest"]

    if args.refresh_db:
        print(f"[AUTO] Refreshing niche_hunter.db via mode={args.refresh_db}...")
        _refresh_db(args.refresh_db)

    # Pick extra candidates first, then filter by Publer recent history
    ideas_candidates: list[tuple[str, float, str, str]] = []
    seed_keywords: set[str] = set()

    # ── Retry mode: load an existing DB post and publish (typically TikTok-only) ──
    if int(args.retry_post_id or 0) > 0:
        if not args.publish:
            print("--retry-post-id requires --publish")
            return 2
        cp = _load_post_from_viralops_db(int(args.retry_post_id))
        if not cp:
            print(f"Retry post not found: id={args.retry_post_id}")
            return 3
        # Force TikTok-only unless caller explicitly restricts platforms.
        if "tiktok" not in platforms:
            platforms = ["tiktok"]
        else:
            platforms = ["tiktok"]

        # Wait a bit before retry to avoid TikTok 1-minute collision.
        try:
            gap = int(os.environ.get("VIRALOPS_TIKTOK_MIN_GAP_SECS", "65") or "65")
        except Exception:
            gap = 65
        wait = max(5, gap)
        print(f"[RETRY] Waiting {wait}s before TikTok retry...")
        await asyncio.sleep(wait)

        results = await _publish_all(cp, platforms=platforms, publish_shopify=False, shopify_draft=False)
        for pf, r in results.items():
            ok = bool(r.get("success"))
            msg = "OK" if ok else f"FAIL: {str(r.get('error', '?'))[:140]}"
            print(f"    {'✓' if ok else '✗'} {pf}: {msg}")

        all_ok = all(bool(r.get("success")) for r in results.values())
        _log_published_to_viralops_db(
            cp,
            platforms=platforms,
            extra={"retry_post_id": int(args.retry_post_id), "results": results},
            status="published" if all_ok else "failed",
        )
        return 0 if all_ok else 10

    # ── Force-topic mode: publish an explicit topic first ──
    forced_topic = str(args.force_topic or "").strip()

    active_ideas_file = str(args.ideas_file or "").strip()

    active_ideas_files = _parse_ideas_file_list(active_ideas_file)

    if args.direct_only and (not active_ideas_files):
        print("[DIRECT] --direct-only requires an ideas file (VIRALOPS_IDEAS_FILE or --ideas-file).")
        return 4

    if active_ideas_files:
        # Pull plenty of candidates so we don't accidentally cap a "hundreds of ideas" file.
        ideas_cap = max(200, args.limit * 200)
        ideas_candidates = _pick_topics_from_ideas_files(active_ideas_files, limit=ideas_cap)

        # If requested, clear the extracted direct-post topics from history so they can be reused.
        # We do this BEFORE --direct-only filtering so we can always find the direct topics list.
        if args.direct_reset_history:
            direct_topics = [
                t for (t, _s, niche, _h) in ideas_candidates
                if (str(niche or "").strip().lower() == "ideas_direct_post")
            ]
            deleted2 = _reset_direct_post_history(direct_topics=direct_topics, any_source=True)
            print(
                f"[IDEAS] Reset direct-post history: deleted={deleted2} rows (topics cleared across any source)"
            )

        if args.direct_only:
            ideas_candidates = [
                c for c in ideas_candidates
                if (str(c[2] or "").strip().lower() == "ideas_direct_post")
            ]

        if args.ideas_reset_history:
            deleted = _reset_ideas_topic_history(ideas_topics=[t for (t, *_rest) in ideas_candidates])
            print(f"[IDEAS] Reset topic_history: deleted={deleted} rows (source in ideas_file*)")
        seed_keywords = _seed_keywords_from_ideas([t for (t, *_rest) in ideas_candidates])

    def _is_ideas_niche(n: str | None) -> bool:
        return (str(n or "").strip().lower() in ("ideas_file", "ideas_file_table", "ideas_file_text"))

    def _prepend_idea_line_to_pack(pack: dict[str, Any], idea_line: str) -> dict[str, Any]:
        """Copy/paste the exact idea line into the post BEFORE the GenAI answer.

        This applies only to ideas-file topics (not direct-post blocks).
        """
        idea = str(idea_line or "").strip()
        if not idea:
            return pack

        cf = str(pack.get("content_formatted") or "")
        ucb = str(pack.get("universal_caption_block") or "")

        def _already_has(text: str) -> bool:
            t = (text or "").strip()
            if not t:
                return False
            # Consider it present if the idea line appears in the first ~400 chars.
            head = t[:400].lower()
            return idea.lower() in head

        out = dict(pack)
        out["_idea_line"] = idea

        # IMPORTANT: user requested to remove hook title entirely for channels.
        # For ideas-list posts, the first line MUST be the idea line verbatim.
        prefix = idea

        if cf and (not _already_has(cf)):
            out["content_formatted"] = f"{prefix}\n\n{cf.lstrip()}"
        elif not cf:
            out["content_formatted"] = prefix

        if ucb and (not _already_has(ucb)):
            out["universal_caption_block"] = f"{prefix}\n\n{ucb.lstrip()}"
        elif not ucb:
            out["universal_caption_block"] = prefix

        return out

    def _strip_generated_hook_line(pack: dict[str, Any], generated_title_line: str) -> dict[str, Any]:
        """Remove the LLM-generated hook/title line that is embedded as the first line of the body.

        For ideas-list posts, user wants heading/title to be ONLY the idea line.
        The LLM still writes a hook as the first line of content_formatted; strip it.
        """

        gen = str(generated_title_line or "").strip()
        if not gen:
            return pack

        def _norm(s: str) -> str:
            s = re.sub(r"\s+", " ", str(s or "").strip().lower())
            return s

        def _strip(text: str) -> str:
            t = str(text or "")
            if not t.strip():
                return t
            lines = t.splitlines()
            if not lines:
                return t
            # Find first non-empty line
            idx0 = 0
            while idx0 < len(lines) and not lines[idx0].strip():
                idx0 += 1
            if idx0 >= len(lines):
                return t
            first = lines[idx0].strip()
            if _norm(first) == _norm(gen):
                # Drop that line and any immediately following blank lines
                j = idx0 + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                return "\n".join(lines[:idx0] + lines[j:]).strip()
            return t

        out = dict(pack)
        out["content_formatted"] = _strip(out.get("content_formatted", ""))
        out["universal_caption_block"] = _strip(out.get("universal_caption_block", ""))
        return out

    # Candidate selection policy:
    # - If there are any unused ideas remaining, publish ONLY from ideas list.
    # - Only after the ideas list is exhausted, fall back to niche_hunter.db.
    # - In --direct-only mode, never fall back.
    candidates: list[tuple[str, float, str, str]] = []

    # Always allow forcing a topic first.
    if forced_topic:
        candidates.append((forced_topic, 10.0, "forced_topic", ""))

    if args.direct_only:
        candidates.extend(list(ideas_candidates))
    elif active_ideas_files and ideas_candidates:
        # Start with ideas only; if none remain after history filter, we'll load DB below.
        candidates.extend(list(ideas_candidates))
    else:
        # No ideas file configured; use DB.
        db_candidates = _pick_topics(limit=max(20, args.limit * 10), min_score=args.min_score)
        candidates.extend(db_candidates)

    if active_ideas_files:
        print(f"[IDEAS] Active ideas file(s): {' | '.join(active_ideas_files)}")
        print(f"[IDEAS] Parsed ideas candidates: {len(ideas_candidates)}")

    # Filter out already-used topics (persisted history)
    candidates = [c for c in candidates if not _topic_already_used(c[0])]

    # If we have an ideas file configured, enforce "use up ideas first" strictly.
    if active_ideas_files:
        remaining_ideas = sum(1 for c in candidates if _is_ideas_niche(c[2]))
        # NOTE: ideas_direct_post is handled by --direct-only. In normal mode we
        # treat it as a non-ideas candidate (it can still be published if explicitly selected).
        print(f"[IDEAS] Remaining (unused) ideas after history filter: {remaining_ideas}")

        if (not args.direct_only) and remaining_ideas > 0:
            # Remove DB candidates (and keep forced_topic if any).
            candidates = [c for c in candidates if (c[2] == "forced_topic") or _is_ideas_niche(c[2])]
        elif (not args.direct_only) and remaining_ideas == 0:
            # Ideas exhausted: now load DB fallback.
            db_candidates = _pick_topics(limit=max(20, args.limit * 10), min_score=args.min_score)
            if seed_keywords:
                db_candidates = _rank_db_candidates_by_seed(db_candidates, seed_keywords)
            # Keep forced_topic (already first) + DB.
            # Also keep any remaining non-ideas candidates that might still be in list.
            forced = [c for c in candidates if c[2] == "forced_topic"]
            candidates = forced + db_candidates

    if args.direct_only and not candidates:
        print("[DIRECT] No unused tui:/genAI: direct-post blocks found.")
        print("[DIRECT] Add a new tui:/genAI: block in part 2, or rerun with --direct-reset-history to repost existing blocks.")
        return 5
    if not args.no_publer_dedup:
        topics = await _dedup_against_publer(
            candidates,
            limit=args.limit,
            pages=args.publer_pages,
            per_page=args.publer_per_page,
            min_overlap=args.publer_min_overlap,
            min_shared=args.publer_min_shared,
            min_jaccard=args.publer_min_jaccard,
        )
    else:
        topics = candidates[: args.limit]

    print("=" * 70)
    print("  AUTO NICHE TOPICS — Batch")
    print(f"  Picked {len(topics)}/{args.limit} topics (min_score={args.min_score})")
    if not args.no_publer_dedup:
        print(
            f"  Publer dedup: pages={args.publer_pages}, per_page={args.publer_per_page}, min_overlap={args.publer_min_overlap}, min_shared={args.publer_min_shared}, min_jaccard={args.publer_min_jaccard}"
        )
    print(f"  Mode: {'PUBLISH' if args.publish else 'DRY-RUN'}")
    print(f"  Platforms: {', '.join(platforms) if platforms else '(none)'}")
    print("=" * 70)

    if not topics:
        print("No topics found (maybe exhausted). Try --refresh-db full")
        return 2

    for idx, (topic, score, niche, hook) in enumerate(topics, 1):
        print(f"\n[{idx}/{len(topics)}] score={score:.2f} niche={niche}")
        if hook:
            print(f"  hook: {hook[:140]}")
        print(f"  topic: {topic}")

        if not args.publish:
            continue

        try:
            # If the ideas file contains a full GenAI answer block, publish it directly.
            if (niche or "").strip().lower() == "ideas_direct_post":
                cp = _build_direct_content_pack(hook or topic)
                # Keep the original idea as the title/topic key for history + logs.
                if topic:
                    cp["title"] = str(topic)[:120]
                    cp["_topic"] = str(topic)
            else:
                cp = await _generate_content(
                    topic,
                    score=9.0,
                    idea_line=topic if _is_ideas_niche(niche) else "",
                )

                # If this topic comes from the curated ideas list, copy/paste the
                # exact idea line into the article BEFORE the GenAI answer.
                if _is_ideas_niche(niche):
                    # Treat the idea line as the canonical title.
                    generated_hook_title = str(cp.get("title") or "").strip()
                    cp["title"] = str(topic).strip()
                    cp["_topic"] = str(topic).strip()
                    # Remove the LLM hook line embedded in the body.
                    cp = _strip_generated_hook_line(cp, generated_title_line=generated_hook_title)
                    cp = _prepend_idea_line_to_pack(cp, idea_line=topic)
            title = cp.get("title", "?")
            chars = len(cp.get("content_formatted", ""))
            review_score = float(cp.get("_review_score", 0.0) or 0.0)
            rubric_total = float(cp.get("_rubric_total_100", 0.0) or 0.0)
            review_pass = bool(cp.get("_review_pass"))
            print(f"  ✓ Generated: {title[:90]} ({chars} chars, review={review_score:.1f}/10)")

            # Hard gate by default: never publish if quality review didn't pass.
            # For smoke tests, allow override.
            if (not review_pass) and (not args.skip_quality_gate):
                fb = str(cp.get("_review_feedback", "") or "").strip()
                print(
                    f"  ⚠ SKIP PUBLISH: quality gate failed (review_pass={review_pass}, "
                    f"tiktok_avg={review_score:.1f}/10, rubric={rubric_total:.0f}/100)."
                )
                if fb:
                    print(f"  Feedback: {fb[:220]}")

                _log_published_to_viralops_db(
                    {
                        "title": cp.get("title", ""),
                        "content_formatted": cp.get("content_formatted", ""),
                        "universal_caption_block": cp.get("universal_caption_block", ""),
                    },
                    platforms=platforms,
                    extra={
                        "topic": topic,
                        "picked_score": score,
                        "picked_niche": niche,
                        "picked_hook": hook,
                        "review_pass": review_pass,
                        "tiktok_avg": review_score,
                        "rubric_total_100": rubric_total,
                        "feedback": fb,
                    },
                    status="failed",
                )
                continue

            if (not review_pass) and args.skip_quality_gate:
                fb = str(cp.get("_review_feedback", "") or "").strip()
                print(
                    f"  ⚠ OVERRIDE: publishing despite gate fail (review_pass={review_pass}, "
                    f"tiktok_avg={review_score:.1f}/10, rubric={rubric_total:.0f}/100)."
                )
                if fb:
                    print(f"  Feedback: {fb[:220]}")

            results = await _publish_all(
                cp,
                platforms=platforms,
                publish_shopify=bool(args.shopify),
                shopify_draft=bool(args.shopify_draft),
            )
            for pf, r in results.items():
                ok = bool(r.get("success"))
                if ok and r.get("scheduled"):
                    when = str(r.get("_scheduled_at") or r.get("schedule_at") or "").strip()
                    msg = f"SCHEDULED {when}".strip()
                else:
                    msg = "OK" if ok else f"FAIL: {str(r.get('error', '?'))[:140]}"
                print(f"    {'✓' if ok else '✗'} {pf}: {msg}")

            # Print match summary (Shopify URL ↔ Pinterest pin link)
            shop = results.get("shopify_blog") or {}
            pin = results.get("pinterest") or {}
            shop_url = str(shop.get("post_url") or "").strip()
            shop_admin = str(shop.get("admin_url") or "").strip()
            pin_link = str(pin.get("post_link") or "").strip()
            pin_dest = str(pin.get("url") or "").strip()
            pin_post_id = ""
            if isinstance(pin, dict):
                # Prefer matched_post_id, then created_post_ids, then job_id/post_id.
                pin_post_id = str(pin.get("matched_post_id") or "").strip()
                if not pin_post_id:
                    cids = pin.get("created_post_ids") or []
                    if isinstance(cids, list) and cids:
                        pin_post_id = str(cids[0] or "").strip()
                if not pin_post_id:
                    pin_post_id = str(pin.get("post_id") or "").strip()
            if (not pin_link) or (not pin_dest):
                created_posts = pin.get("created_posts") or []
                if isinstance(created_posts, list) and created_posts and isinstance(created_posts[0], dict):
                    if not pin_link:
                        pin_link = str(created_posts[0].get("post_link") or "").strip()
                    if not pin_dest:
                        pin_dest = str(created_posts[0].get("url") or "").strip()
            if shop_url:
                print(f"      shopify_url: {shop_url}")
            if shop_admin:
                print(f"      shopify_admin: {shop_admin}")
            if pin_link:
                print(f"      pinterest_pin: {pin_link}")
            if pin_dest:
                print(f"      pinterest_dest: {pin_dest}")
            if (not pin_link) and pin_post_id:
                print(f"      pinterest_post_id: {pin_post_id}")

            # Persist only if ALL platform publishes succeeded.
            all_ok = all(bool(r.get("success")) for r in results.values())
            if all_ok:
                _log_published_to_viralops_db(
                    cp,
                    platforms=platforms,
                    extra={
                        "topic": topic,
                        "picked_score": score,
                        "picked_niche": niche,
                        "picked_hook": hook,
                        "shopify_post_url": (results.get("shopify_blog") or {}).get("post_url", ""),
                        "pinterest_dest_url": (results.get("shopify_blog") or {}).get("post_url", "") or cp.get("destination_url") or cp.get("url") or "",
                    },
                )
                _mark_topic_used(topic, source=niche or "")
            else:
                # Record a failure entry (for debugging) but do NOT mark topic used.
                _log_published_to_viralops_db(
                    {"title": cp.get("title", ""), "content_formatted": cp.get("content_formatted", ""), "universal_caption_block": cp.get("universal_caption_block", "")},
                    platforms=platforms,
                    extra={
                        "topic": topic,
                        "picked_score": score,
                        "picked_niche": niche,
                        "picked_hook": hook,
                        "results": results,
                    },
                    status="failed",
                )

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:160]}")

        time.sleep(5)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
