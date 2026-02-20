from web.app import _fallback_tiktok_clumpproof


def test_tiktok_fallback_strict_title_has_no_dash_after_title():
    title = "DIY grow-your-own mycelium lampshades using GIY kits with agricultural waste for indoor herb lighting"
    content = "Some body content with 🌿 sections\n\n✅ Tips here"
    caption = _fallback_tiktok_clumpproof(title, content, ["#one"], strict_title=True)
    assert caption.startswith(title + " ")
    assert not caption.startswith(title + " — ")
