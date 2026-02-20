from llm_content import build_image_prompt


def test_build_image_prompt_prefers_idea_line_subject():
    pack = {
        "title": "Stop wasting water — hugelkultur mounds hold moisture like a sponge!",
        "pain_point": "Wasting water in your garden",
        "image_title": "Hugelkultur Edges",
        "_idea_line": "Acid-loving herb edges (lavender, borage) around fruit bushes in hugelkultur mounds for water retention",
        "content_formatted": "Acid-loving herb edges (lavender, borage) around fruit bushes...\n\nFull answer...",
    }
    prompt = build_image_prompt(pack)
    # Subject should be derived from the idea line, not the generic image_title.
    assert "lavender" in prompt.lower()
    assert "borage" in prompt.lower()
    assert "hugelkultur" in prompt.lower()
