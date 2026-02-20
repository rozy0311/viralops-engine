import re

import pytest


@pytest.mark.asyncio
async def test_prepare_facebook_dedupes_and_avoids_double_hashtag_tail():
    # long_content includes a hashtag token with punctuation (#TagOne,), and the input list contains
    # duplicates/casing variants. We must not re-append the same hashtag again.
    from web.app import _prepare_facebook_content

    content_pack = {
        "title": "Test Title",
        "universal_caption_block": "🌿 Test Title\n\nBody line with #TagOne, inside.\n\n#TagTwo",
        "hashtags": ["TagOne", "TagTwo", "tagone", "TagThree", "TAGTHREE"],
    }

    result = await _prepare_facebook_content(content_pack)
    assert result["platforms"] == ["facebook"]
    assert result["hashtags"] == []  # hashtags are embedded in caption

    caption = result["caption"]
    tags = [t.lower() for t in re.findall(r"#[^\s#]+", caption)]

    # Each tag appears only once across the entire caption.
    assert tags.count("#tagone") == 1
    assert tags.count("#tagtwo") == 1
    assert tags.count("#tagthree") == 1
