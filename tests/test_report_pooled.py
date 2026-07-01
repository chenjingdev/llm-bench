from bench import axes as axis_mod


def test_creativity_is_pooled():
    assert "creativity" in axis_mod.POOLED


def test_non_pooled_axes_unchanged():
    assert "density" not in axis_mod.POOLED
    assert "sycophancy" not in axis_mod.POOLED


def test_score_pool_dispatch(monkeypatch):
    monkeypatch.setattr(axis_mod.creativity.embed, "available", lambda: False)
    res = axis_mod.score_pool("creativity", {"A": [], "B": []})
    assert set(res) == {"A", "B"}
