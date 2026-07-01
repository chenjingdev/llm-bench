from bench.axes import creativity as cr


def _sample(model, pid, sub, text):
    return {"probe_id": pid, "model": model, "text": text, "ok": True,
            "meta": {"subtype": sub, "prompt": f"brainstorm ideas about {sub}"}}


# --- 코어 헬퍼 ---
def test_parse_items_numbered_and_bullet():
    text = "1. first idea\n2) second idea\n- third idea"
    assert cr.parse_items(text) == ["first idea", "second idea", "third idea"]


def test_percentile_rank_bounds():
    pop = [0.0, 1.0, 2.0, 3.0]
    assert cr.percentile_rank(-1, pop) == 0.0
    assert cr.percentile_rank(4, pop) == 1.0
    assert 0.0 < cr.percentile_rank(1.5, pop) < 1.0


def test_percentile_rank_empty_pop():
    assert cr.percentile_rank(5.0, []) == 0.5


def test_ontopic_gate_passes_typical_penalizes_low_tail():
    cos_all = [0.7, 0.72, 0.68, 0.71, 0.30]   # 마지막이 딴소리
    assert cr.ontopic_gate(0.71, cos_all) > 0.8
    assert cr.ontopic_gate(0.30, cos_all) < cr.ontopic_gate(0.71, cos_all)
    assert 0.0 <= cr.ontopic_gate(0.30, cos_all) <= 1.0


# --- 풀-인지 스코어러 (임베딩 결정론 모킹) ---
def test_score_pool_ranks_original_above_cliche(monkeypatch):
    def fake_embed(texts, prefix=True):
        vocab = ["doorstop", "paperweight", "pigment", "amplifier", "chess",
                 "brainstorm", "ideas", "about", "humor", "tech"]
        return [[float(t.lower().count(w)) for w in vocab] + [float(len(t) % 7)]
                for t in texts]
    monkeypatch.setattr(cr.embed, "embed", fake_embed)
    monkeypatch.setattr(cr.embed, "available", lambda: True)
    cliche = [_sample("A", "creativity-tech-0", "tech",
                      "1. brainstorm ideas about tech\n2. brainstorm ideas about tech")]
    orig = [_sample("B", "creativity-tech-0", "tech",
                    "1. grind it into red pigment amplifier\n2. carve a chess pigment")]
    res = cr.score_pool({"A": cliche, "B": orig})
    assert set(res) == {"A", "B"}
    assert res["B"].score >= res["A"].score


def test_score_pool_embed_unavailable(monkeypatch):
    monkeypatch.setattr(cr.embed, "available", lambda: False)
    res = cr.score_pool({"A": [_sample("A", "p", "tech", "x")]})
    assert res["A"].n == 0


# --- 단일모델 CDAT 폴백 ---
def test_score_fallback_single_model_no_crash(monkeypatch):
    monkeypatch.setattr(cr.embed, "available", lambda: True)
    monkeypatch.setattr(cr.embed, "embed",
                        lambda texts, prefix=True: [[float(len(t)), 1.0] for t in texts])
    r = cr.score([_sample("A", "creativity-tech-0", "tech",
                          "1. alpha idea\n2. beta notion\n3. gamma plan")])
    assert 0.0 <= r.score <= 100.0
    assert "폴백" in (r.note or "")
