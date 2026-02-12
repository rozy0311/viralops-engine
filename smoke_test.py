"""
Smoke Test — ViralOps Blog Auto-Share Pipeline
Step 1: Shopify API → fetch articles
Step 2: Sendible login → verify connection
Step 3: Share 1 real article → TikTok + Pinterest via Sendible
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()


async def step1_shopify():
    """Test Shopify Admin API — fetch latest articles."""
    print("\n" + "=" * 60)
    print("STEP 1: Shopify Admin API")
    print("=" * 60)

    from integrations.shopify_blog_watcher import ShopifyBlogWatcher
    w = ShopifyBlogWatcher()
    print(f"  Shop: {w._shop}")
    print(f"  Token: {w._token[:12]}..." if w._token else "  Token: ❌ MISSING")
    print(f"  Blog handles: {w._blog_handles}")

    ok = await w.connect()
    print(f"  Connected: {ok}")
    if not ok:
        print("  ❌ FAILED to connect to Shopify")
        return None

    print(f"  Blog map: {w._blog_map}")

    articles = []
    for handle in w._blog_map:
        fetched = await w.get_recent_articles(handle, limit=2)
        for t in fetched:
            articles.append(t)
            print(f"\n  📰 [{handle}] {t['title']}")
            print(f"     URL:   {t['url']}")
            img = t.get("featured_image", "")
            print(f"     Image: {img[:80] if img else 'none'}")
            print(f"     Tags:  {t.get('tags', [])}")

    await w.close()
    print(f"\n  ✅ Found {len(articles)} articles total")
    return articles


async def step2_sendible():
    """Test Sendible UI login via Playwright stealth."""
    print("\n" + "=" * 60)
    print("STEP 2: Sendible Connection (Playwright stealth)")
    print("=" * 60)

    email = os.getenv("SENDIBLE_EMAIL", "")
    has_pw = bool(os.getenv("SENDIBLE_PASSWORD"))
    headless = os.getenv("SENDIBLE_HEADLESS", "false")
    print(f"  Email: {email}")
    print(f"  Password: {'✅ set' if has_pw else '❌ missing'}")
    print(f"  Headless: {headless}")

    if not email or not has_pw:
        print("  ❌ SENDIBLE_EMAIL or SENDIBLE_PASSWORD missing")
        return False

    from integrations.sendible_ui_publisher import SendibleUIPublisher
    pub = SendibleUIPublisher()
    print("  Connecting (browser will open)...")

    try:
        connected = await pub.connect()
        print(f"  Connected: {connected}")

        if connected:
            services = await pub.get_services()
            print(f"  Services found: {len(services)}")
            for s in services:
                name = s.get("name", s.get("service_name", "?"))
                stype = s.get("type", s.get("service_type", "?"))
                print(f"    🔗 {name} ({stype})")
            print("  ✅ Sendible ready!")
        else:
            print("  ❌ Login failed")

        await pub.close()
        return connected
    except Exception as e:
        print(f"  ❌ Error: {e}")
        try:
            await pub.close()
        except:
            pass
        return False


async def step3_share(article):
    """Share a real article via Sendible to TikTok + Pinterest."""
    print("\n" + "=" * 60)
    print("STEP 3: Share Article via Sendible")
    print("=" * 60)

    title = article.get("title", "Untitled")
    url = article.get("url", "")
    image = article.get("featured_image", "")
    tags = article.get("tags", [])

    print(f"  Article: {title}")
    print(f"  URL:     {url}")
    print(f"  Image:   {image[:80] if image else 'none'}")

    from integrations.shopify_auto_share import ShopifyAutoShare
    auto = ShopifyAutoShare()

    print("  Initializing pipeline...")
    init_result = await auto.initialize()
    print(f"  Init: {init_result}")

    if not auto._sendible:
        print("  ❌ Sendible not connected — cannot share")
        await auto.close()
        return None

    print(f"  TikTok via: {auto._tiktok_via}")
    print(f"  Pinterest via: {auto._pinterest_via}")

    print("\n  🚀 Sharing article...")
    result = await auto._share_article(article)

    print(f"\n  📊 Result:")
    tiktok_results = result.get("tiktok", [])
    for tr in tiktok_results:
        status = "✅" if tr.get("success") else "❌"
        print(f"    TikTok: {status} (via {tr.get('account', '?')})")
        if not tr.get("success"):
            print(f"      Error: {tr.get('error', 'unknown')}")

    pin = result.get("pinterest")
    if pin:
        status = "✅" if pin.get("success") else "❌"
        print(f"    Pinterest: {status}")
        if not pin.get("success"):
            print(f"      Error: {pin.get('error', 'unknown')}")
    else:
        print(f"    Pinterest: skipped")

    await auto.close()
    return result


async def main():
    print("🔥 ViralOps Engine — Smoke Test")
    print("================================")

    # Step 1
    articles = await step1_shopify()
    if not articles:
        print("\n💀 Shopify failed — stopping.")
        return

    input("\n⏸️  Press Enter to test Sendible login (browser will open)...")

    # Step 2
    sendible_ok = await step2_sendible()
    if not sendible_ok:
        print("\n💀 Sendible failed — stopping.")
        return

    # Step 3
    print(f"\n📋 Available articles:")
    for i, a in enumerate(articles):
        print(f"  {i+1}. [{a.get('blog_handle')}] {a['title']}")

    choice = input(f"\nPick article to share (1-{len(articles)}, or 'skip'): ").strip()
    if choice.lower() == "skip":
        print("Skipped sharing. Smoke test done! ✅")
        return

    try:
        idx = int(choice) - 1
        article = articles[idx]
    except (ValueError, IndexError):
        article = articles[0]
        print(f"Using first article: {article['title']}")

    result = await step3_share(article)
    if result:
        print("\n🎉 Smoke test complete!")
    else:
        print("\n⚠️ Share had issues — check logs above.")


if __name__ == "__main__":
    asyncio.run(main())
