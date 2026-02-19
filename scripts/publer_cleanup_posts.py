"""Publer cleanup helper (safe by default).

This script helps you *inspect* recent Publer posts and optionally delete by
keyword match. Default is DRY-RUN listing only.

Why this exists
---------------
Batch publish scripts historically did not persist returned Publer post IDs.
So if you want to clean up already-published posts, we must:
  1) List posts via Publer API
  2) Filter by keyword(s)
  3) Delete explicit matches

Usage
-----
  # List last 30 posts
  python scripts/publer_cleanup_posts.py

  # Dry-run filter
  python scripts/publer_cleanup_posts.py --match "Zone 6a" --match "Hydrozoning"

  # Actually delete matches (requires BOTH flags)
  python scripts/publer_cleanup_posts.py --match "Zone 6a" --delete --yes

Notes
-----
- Publer API deletion deletes the Publer post record; whether it removes the
  already-published social post depends on Publer/network behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)



def _textify(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return str(obj)
    except Exception:
        return ""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--status", type=str, default="")
    ap.add_argument("--match", action="append", default=[], help="Keyword filter; can be provided multiple times")
    ap.add_argument("--debug", action="store_true", help="Print raw GET /posts response status/text")
    ap.add_argument("--delete", action="store_true", help="Delete matched posts")
    ap.add_argument("--yes", action="store_true", help="Required confirmation for delete")
    args = ap.parse_args()

    from integrations.publer_publisher import PublerPublisher

    pub = PublerPublisher()
    # Ensure connected (so debug can call raw request)
    await pub.connect()
    if args.debug:
        params: dict[str, Any] = {"limit": args.limit, "page": args.page}
        if args.status:
            params["status"] = args.status
        try:
            resp = await pub._request("GET", "posts", params=params)  # noqa: SLF001
            print(f"[DEBUG] GET /posts -> {resp.status_code}")
            print(f"[DEBUG] body: {resp.text[:500]}")
        except Exception as e:
            print(f"[DEBUG] GET /posts error: {e}")

    posts = await pub.get_posts(status=args.status, limit=args.limit, page=args.page)

    needles = [m.lower() for m in args.match if m and m.strip()]

    matches: list[dict] = []
    for p in posts:
        pid = _textify(p.get("id", p.get("_id", "")))
        title = _textify(p.get("title", ""))
        text = _textify(p.get("text", p.get("body", "")))
        created = _textify(p.get("created_at", p.get("createdAt", "")))
        status = _textify(p.get("status", ""))

        hay = f"{title}\n{text}".lower()
        ok = True
        for n in needles:
            if n not in hay:
                ok = False
                break
        if needles and not ok:
            continue

        matches.append({
            "id": pid,
            "status": status,
            "created": created,
            "title": title,
            "text_preview": (text[:120] + "...") if len(text) > 120 else text,
        })

    print("=" * 80)
    print(f"Publer posts fetched: {len(posts)} | matches: {len(matches)}")
    if needles:
        print(f"Filter: {needles}")
    print("Mode:", "DELETE" if args.delete else "DRY-RUN")
    print("=" * 80)

    for m in matches:
        print(f"- id={m['id']} status={m['status']} created={m['created']}")
        if m["title"]:
            print(f"  title: {m['title']}")
        print(f"  text: {m['text_preview']}")

    if args.delete:
        if not args.yes:
            print("\nRefusing to delete: pass --yes to confirm")
            return 3
        if not needles:
            print("\nRefusing to delete: pass at least one --match keyword")
            return 4

        deleted = 0
        for m in matches:
            pid = m["id"]
            if not pid:
                continue
            ok = await pub.delete_post(pid)
            print(f"DELETE id={pid}: {'OK' if ok else 'FAIL'}")
            if ok:
                deleted += 1

        print(f"\nDeleted: {deleted}/{len(matches)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
