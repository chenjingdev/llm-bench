import math

from bench.axes import creativity as cr
from bench.axes import _textmetrics as tm


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


# --- 어휘 접지(lexical grounding) ---
def test_content_words_strips_stopwords_and_short_tokens():
    cw = tm.content_words("The quick fox and a big red dog ran to it")
    # 불용어(the/and/a/to/it)·길이<3(it) 제외, 내용어만 남아야
    assert cw == {"quick", "fox", "big", "red", "dog", "ran"}
    assert "the" not in cw and "and" not in cw and "it" not in cw


def test_lexical_grounding_short_item_exempt():
    # 내용어 2개 미만(신조 단일 단어 등)은 ref_vocab과 무관하게 항상 1.0
    assert tm.lexical_grounding("Quietude", {"desk", "office"}) == 1.0
    assert tm.lexical_grounding("", {"desk"}) == 1.0


def test_lexical_grounding_overlap_fraction():
    ref = {"cache", "context", "query"}
    # 내용어 4개 중 2개(cache, context)가 ref_vocab에 있음 → 0.5
    assert tm.lexical_grounding("cache the context somehow please", ref) == 0.5
    # 전부 겹치면 1.0
    assert tm.lexical_grounding("cache the context query", ref) == 1.0
    # 하나도 안 겹치면 0.0
    assert tm.lexical_grounding("banana quantum sock helix", ref) == 0.0


def test_has_function_word_distinguishes_salad_from_sentence():
    assert tm.has_function_word("treat citations as a debt the answer must repay")
    assert not tm.has_function_word("purple monday elephant sqrt tractor")


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


# --- LOF 국소성(adaptive k)이 실제로 순위를 좌우하는지 ---
# 기하: 공유 "온토픽" 축(D, 압도적, 모든 아이템 동일) + 별도 3차원 소축(위치,
# 노름 M으로 고정) → prompt-코사인은 모든 아이템에서 상수(ontopic이 승부에
# 개입 못 하게 분리). A의 12개 항목은 소축의 한 방향(e_a) 근방에 약한 지터로
# 모여 있는 조밀한 컨센서스 클러스터(서로 거의 동일한 벡터). B의 1개 항목은
# 완전히 직교인 방향(e_b)에 홀로 위치한 진짜 국소 이상치.
# 실측(k=20, 수정 전): 풀 크기 n=13 → k가 n-1=12로 클램프되어 사실상
# "전역" 이웃이 되고 밀도차가 뭉개짐 — A=91.59 > B=7.33 (역전, discrimination 상실).
# 실측(adaptive k=4, 수정 후): A=84.27 < B=91.59 — LOF가 B를 올바르게 1위로.
_D = 1000.0
_M = 10.0
_PROMPT = "brainstorm ideas about onboarding"
_E_A = (0.0, 1.0, 0.0)
_E_B = (0.0, 0.0, 1.0)
_A_ITEMS = [
    "streamline signup with social login options",
    "add a progress bar during onboarding steps",
    "send a friendly welcome email after signup",
    "offer a guided product tour on first visit",
    "provide sample data to explore features quickly",
    "highlight quick wins in the first session",
    "let users skip optional profile fields",
    "show contextual tooltips on hover",
    "enable a dark mode toggle in settings",
    "add a keyboard shortcuts cheat sheet",
    "provide inline validation on form fields",
    "offer a checklist for onboarding tasks",
]
_B_ITEM = ("a bioluminescent onboarding guide that rewrites its narrative "
           "tone from real-time mood telemetry")


def _normalize3(v, mag):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n * mag for x in v]


def _jitter3(base, spread, i):
    # 결정론적(비-random) 소섭동: 클러스터 내 아이템을 서로 살짝 다르게(중복
    # 벡터로 인한 LOF 0/inf 방지) 하되 e_base 근방에 조밀하게 유지.
    off = [base[0] + spread * math.sin(i * 0.7 + 0.1),
           base[1] + spread * math.cos(i * 0.9 + 0.3),
           base[2] + spread * math.sin(i * 1.3 + 0.5)]
    return _normalize3(off, _M)


def _lof_probe_vecmap():
    vecmap = {_PROMPT: [_D, 0.0, 0.0, 0.0]}
    for i, text in enumerate(_A_ITEMS):
        vecmap[text] = [_D] + _jitter3(_E_A, 0.10, i)
    vecmap[_B_ITEM] = [_D] + _normalize3(list(_E_B), _M)
    return vecmap


def _lof_probe_samples():
    a_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(_A_ITEMS))
    b_text = f"1. {_B_ITEM}"
    pid, sub = "creativity-lof-0", "onboarding"
    return {
        "A": [_sample("A", pid, sub, a_text)],
        "B": [_sample("B", pid, sub, b_text)],
    }, a_text, b_text


def test_score_pool_lof_discriminates(monkeypatch):
    vecmap = _lof_probe_vecmap()
    monkeypatch.setattr(cr.embed, "embed", lambda texts, prefix=True: [vecmap[t] for t in texts])
    monkeypatch.setattr(cr.embed, "available", lambda: True)
    samples, a_text, b_text = _lof_probe_samples()

    # 검증 게이트가 아니라 LOF가 승부를 가름을 보장: 둘 다 비중복(validity≈1).
    assert tm.validity_gate(a_text, _A_ITEMS) > 0.99
    assert tm.validity_gate(b_text, [_B_ITEM]) > 0.99

    res = cr.score_pool(samples)
    assert res["B"].score > res["A"].score


def test_lof_component_is_load_bearing(monkeypatch):
    vecmap = _lof_probe_vecmap()
    monkeypatch.setattr(cr.embed, "embed", lambda texts, prefix=True: [vecmap[t] for t in texts])
    monkeypatch.setattr(cr.embed, "available", lambda: True)
    # LOF를 무력화(전원 동률) → novelty가 더 이상 판별력을 갖지 않아야 함.
    monkeypatch.setattr(cr.embed, "lof", lambda vecs, k=2: [1.0] * len(vecs))
    samples, _a_text, _b_text = _lof_probe_samples()

    res = cr.score_pool(samples)
    assert res["B"].score <= res["A"].score + 1e-6


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


# --- 형식 계약 응답의 Body 추출 (metaphor 보일러플레이트 오염 방지) ---
def test_metaphor_item_uses_body_section_when_contracted():
    text = ("## Brief\nExplain embeddings via an analogy.\n"
            "## Body\nA vector embedding is a sommelier's palate: each dimension a "
            "tasting note, so two wines land close when they taste alike.\n"
            "## Signal\nThe palate framing is the unconventional part.")
    recs = cr._item_records([_sample("A", "creativity-metaphor-0", "metaphor", text)])
    assert len(recs) == 1
    assert recs[0]["item"].startswith("A vector embedding is a sommelier")
    assert "##" not in recs[0]["item"]


def test_metaphor_item_full_text_without_contract():
    text = "A vector embedding is a sommelier's palate for meaning."
    recs = cr._item_records([_sample("A", "creativity-metaphor-0", "metaphor", text)])
    assert recs[0]["item"] == text
