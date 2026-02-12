"""
E2E Test: Share one Shopify blog article via Sendible → TikTok + Pinterest
Non-interactive version — picks the first article and tries to share.
"""
import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

# Show logs from our integrations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


async def run():
    print("=" * 60)
    print("🔥 E2E Test: Shopify → Sendible (TikTok + Pinterest)")
    print("=" * 60)

    # ── 1) Fetch one article from Shopify ──
    print("\n[1/3] Fetching articles from Shopify...")
    from integrations.shopify_blog_watcher import ShopifyBlogWatcher
    watcher = ShopifyBlogWatcher()
    ok = await watcher.connect()
    if not ok:
        print("❌ Shopify connection failed!")
        return

    # Pick first blog handle
    first_handle = list(watcher._blog_map.keys())[0]
    articles = await watcher.get_recent_articles(first_handle, limit=1)
    await watcher.close()

    if not articles:
        print("❌ No articles found!")
        return

    article = articles[0]
    print(f"  📰 Title: {article['title']}")
    print(f"  🔗 URL: {article['url']}")
    img = article.get("featured_image", "")
    print(f"  🖼️  Image: {img[:80]}..." if img else "  🖼️  Image: none")
    print(f"  🏷️  Tags: {article.get('tags', [])}")

    # ── 2) Initialize ShopifyAutoShare (connects Sendible) ──
    print("\n[2/3] Initializing ShopifyAutoShare (connects Sendible)...")
    from integrations.shopify_auto_share import ShopifyAutoShare
    auto = ShopifyAutoShare()
    init = await auto.initialize()
    print(f"  Init result: {init}")

    if not init.get("initialized"):
        print("❌ ShopifyAutoShare init failed!")
        await auto.close()
        return

    if not init.get("sendible_connected"):
        print("⚠️  Sendible not connected — will try API fallback if available")

    # ── 3) Share the article ──
    print("\n[3/3] Sharing article...")
    print(f"  TikTok via: {init.get('tiktok_via', 'none')}")
    print(f"  Pinterest via: {init.get('pinterest_via', 'none')}")

    result = await auto._share_article(article)

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    # TikTok results
    tiktok_results = result.get("tiktok", [])
    if tiktok_results:
        for tr in tiktok_results:
            status = "✅" if tr.get("success") else "❌"
            print(f"  TikTok ({tr.get('account', '?')}): {status}")
            if tr.get("success"):
                print(f"    post_id: {tr.get('post_id', '-')}")
                print(f"    method: {tr.get('method', '-')}")
            else:
                print(f"    error: {tr.get('error', 'unknown')}")
    else:
        print("  TikTok: not attempted (no publisher available)")

    # Pinterest results
    pin = result.get("pinterest")
    if pin:
        status = "✅" if pin.get("success") else "❌"
        print(f"  Pinterest: {status}")
        if pin.get("success"):
            print(f"    post_id: {pin.get('post_id', '-')}")
        else:
            print(f"    error: {pin.get('error', 'unknown')}")
    else:
        print("  Pinterest: not attempted")

    await auto.close()
    print("\n🏁 Done!")


if __name__ == "__main__":
    asyncio.run(run())
