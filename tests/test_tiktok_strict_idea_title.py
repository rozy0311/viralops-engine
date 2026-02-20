from web.app import _fallback_tiktok_clumpproof
from web.app import _strip_tiktok_title_label


def test_tiktok_reformat_strips_leading_title_label_variants():
    raw = "TITLE: This should be the first sentence.\n\nMore content."
    cleaned = _strip_tiktok_title_label(raw)
    assert not cleaned.upper().startswith("TITLE:")

    raw2 = "TITTLE: Another caption body.\nText."
    cleaned2 = _strip_tiktok_title_label(raw2)
    assert not cleaned2.upper().startswith("TITTLE:")


def test_tiktok_fallback_strict_title_has_no_dash_after_title():
    title = "DIY grow-your-own mycelium lampshades using GIY kits with agricultural waste for indoor herb lighting"
    content = "Some body content with 🌿 sections\n\n✅ Tips here"
    caption = _fallback_tiktok_clumpproof(title, content, ["#one"], strict_title=True)
    assert caption.startswith(title + " ")
    assert not caption.startswith(title + " — ")
