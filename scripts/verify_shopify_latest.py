from __future__ import annotations

import asyncio
import re
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

# Ensure repo root is on sys.path so `integrations.*` imports work when running
# this script from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> int:
    from integrations.shopify_blog_publisher import ShopifyBlogPublisher

    pub = ShopifyBlogPublisher(account_id="shopify_viralops")
    ok = await pub.connect()
    print("connect=", ok)
    if not ok:
        return 2

    try:
        articles = await pub.list_articles(limit=5)
        latest = articles[0] if articles else {}
        article_id = str(latest.get("id", "") or "").strip()
        print("latest_article_id=", article_id)
        if not article_id:
            return 3

        art = await pub.get_article(article_id)
        if not art:
            print("get_article failed")
            return 4

        body = str(art.get("body_html") or "")
        featured = art.get("image") or {}
        if not isinstance(featured, dict):
            featured = {}

        has_inline_img = bool(re.search(r"<img\b", body, re.I))
        has_inline_alt = bool(re.search(r"<img[^>]+\balt=\"[^\"]+\"", body, re.I))

        print("has_inline_img=", has_inline_img)
        print("has_inline_alt=", has_inline_alt)
        print("featured_src_head=", str(featured.get("src") or "")[:120])
        print("featured_alt=", str(featured.get("alt") or "")[:120])
        # Verify SEO via metafields (Shopify stores article SEO in global.title_tag/description_tag)
        try:
            mfs = await pub._rate_limited_get(
                f"{pub._base_url}/articles/{article_id}/metafields.json"
            )
            mfs.raise_for_status()
            metafields = (mfs.json() or {}).get("metafields", [])
            desc = ""
            title_tag = ""
            for mf in metafields if isinstance(metafields, list) else []:
                if not isinstance(mf, dict):
                    continue
                if mf.get("namespace") == "global" and mf.get("key") == "description_tag":
                    desc = str(mf.get("value") or "")
                if mf.get("namespace") == "global" and mf.get("key") == "title_tag":
                    title_tag = str(mf.get("value") or "")
            print("seo_title_len=", len(title_tag))
            print("seo_title_head=", title_tag[:180])
            print("meta_len=", len(desc))
            print("meta_head=", desc[:180])
        except Exception as e:
            print("seo_metafields_check_error=", str(e)[:200])

        return 0
    finally:
        await pub.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
