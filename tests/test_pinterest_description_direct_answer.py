import os
import tempfile

import pytest


@pytest.mark.asyncio
async def test_pinterest_description_starts_with_direct_answer_sentence(tmp_path):
    # Avoid calling the Pinterest master prompt / LLM in this unit test.
    os.environ["VIRALOPS_PINTEREST_MASTER_PROMPT"] = "0"
    os.environ["VIRALOPS_DISABLE_HASHTAGS"] = "1"

    from web.app import _prepare_pinterest_content

    # Create a tiny local image so Pinterest doesn't skip.
    img_path = tmp_path / "pin.jpg"
    try:
        from PIL import Image

        Image.new("RGB", (1000, 1500), (250, 250, 250)).save(str(img_path), format="JPEG", quality=90)
    except Exception:
        img_path.write_bytes(b"fake")

    content_pack = {
        "title": "Mycelium-hemp brick walls for 4x8 raised beds",
        "universal_caption_block": "Mycelium-hemp brick walls for 4x8 raised beds\n\nStart by soaking hemp hurd overnight, then mix with mycelium spawn and pack into forms. Let it colonize 10-14 days in a warm closet.\n\nNo links here.",
        "content_formatted": "",
        "hashtags": ["RaisedBedGarden"],
        "_ai_image_path": str(img_path),
        "destination_url": "https://example.com/blog/mycelium-hemp",
    }

    res = await _prepare_pinterest_content(content_pack)
    assert res and isinstance(res, dict)

    caption = str(res.get("caption") or "")
    assert caption

    # Hashtags must be stripped when globally disabled.
    assert "#" not in caption

    first_line = caption.splitlines()[0].strip()
    # Must start with a direct answer sentence (actionable), not the title.
    assert first_line.lower().startswith((
        "use ", "start ", "mix ", "build ", "try ", "add ", "plant ", "layer ", "fill ",
        "grow ", "soak ", "hang ", "place ", "cut ", "make ", "set ", "keep ", "apply ",
        "quick answer:", "do this:", "start by ",
    ))
    assert first_line.lower() != content_pack["title"].lower()
