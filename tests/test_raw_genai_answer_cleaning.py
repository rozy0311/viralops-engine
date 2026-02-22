from llm_content import clean_raw_genai_answer_text


def test_clean_raw_genai_answer_text_strips_meta_and_fences():
    raw = """Title: Some topic\n\nSure! Here is the answer:\n\n```\ncode fence that should not be here\n```\n\nHashtags: #foo #bar\n\nThis is the real answer line 1.\n\nThis is line 2.\n"""
    cleaned = clean_raw_genai_answer_text(raw)
    assert "Title:" not in cleaned
    assert "Hashtags:" not in cleaned
    assert "```" not in cleaned
    assert "code fence" not in cleaned
    assert "This is the real answer line 1." in cleaned
    assert "This is line 2." in cleaned


def test_clean_raw_genai_answer_text_collapses_blank_lines():
    raw = """\n\n\nSure!\n\n\nAnswer line A\n\n\n\nAnswer line B\n\n\n"""
    cleaned = clean_raw_genai_answer_text(raw)
    assert cleaned.startswith("Answer line A")
    assert "\n\n\n" not in cleaned
