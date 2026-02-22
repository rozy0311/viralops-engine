import os
import tempfile
import textwrap
import importlib.util


def _load_batch_module():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    path = os.path.join(repo_root, "scripts", "batch_auto_niche_topics.py")
    spec = importlib.util.spec_from_file_location("batch_auto_niche_topics", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_parser_drops_micro_niche_ideas_meta_paragraphs():
    mod = _load_batch_module()
    text = textwrap.dedent(
        """
        Micro niche ideas related to plant-based materials focus on hyper-specific DIY crafts, sustainable builds, or upcycled garden products using fibers, mycelium, or waste —

        7\\. NO tool calls, use conversation knowledge only

        Không, không tự hiểu hoàn toàn nếu không gửi ví dụ bài blog gốc hoặc context chi tiết. Prompt trên chỉ dựa vào memory conversation (Zone 6a clay raised beds), nhưng thiếu style viết thì output sẽ generic hơn.

        Nếu thiếu style: Thêm vào prompt: "Style như original: short personal story + numbered 25 ideas + build tips + #hashtags"

        EXAMPLE_ARTICLE: "Winter Garden Layout: 25 Raised Bed Plans for Zone 6a Staunton Farms Staunton clay soil floods spring, freezes solid winter – our 4x8 raised beds solved everything 4 years ago. ... 25 ideas ... #WinterGarden"

        - Real idea: Mycelium foam from corn husks and sawdust for compostable raised bed liners in Zone 6a clay soil.
        - Real idea: Balcony microgreen LED shelf for year-round basil and mint (no soil mess).
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md", encoding="utf-8") as f:
        f.write(text)
        tmp_path = f.name

    try:
        items = mod._pick_topics_from_single_ideas_file(tmp_path, limit=50)
        topics = [t for (t, _s, _n, _h) in items]
        assert not any(t.lower().startswith("micro niche ideas") for t in topics)
        assert not any("tool call" in t.lower() for t in topics)
        assert not any("example_article" in t.lower() for t in topics)
        assert not any("memory conversation" in t.lower() for t in topics)
        assert any("mycelium foam" in t.lower() for t in topics)
        assert any("microgreen" in t.lower() for t in topics)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
