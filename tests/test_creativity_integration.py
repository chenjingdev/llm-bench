import pytest
from bench import embed
from bench.axes import creativity as cr

pytestmark = pytest.mark.skipif(not embed.available(),
                                reason="ollama 임베딩 서버 필요")


def _s(model, sub, text):
    return {"probe_id": f"creativity-{sub}-0", "model": model, "text": text, "ok": True,
            "meta": {"subtype": sub, "prompt": f"brainstorm novel ideas about {sub}"}}


def test_cliche_vs_original_real_embeddings():
    # 뻔한-발산(표준기법 나열) vs 진짜 외딴 아이디어
    cliche = _s("A", "tech",
                "1. tune the chunk size\n2. add a reranker\n3. use hybrid search\n"
                "4. expand the query\n5. increase top-k\n6. fine-tune the embedder")
    original = _s("B", "tech",
                  "1. let retrieval compete in a prediction market priced by usefulness\n"
                  "2. store retrieval failures as negative anti-memories\n"
                  "3. grow a fungal-style mycelium index that reweights by co-activation\n"
                  "4. let the model bid tokens to fetch more context\n"
                  "5. compile frequent sub-questions into learned circuits\n"
                  "6. treat citations as a debt the answer must repay")
    res = cr.score_pool({"A": [cliche], "B": [original]})
    assert res["B"].score > res["A"].score


def test_word_salad_gated_out():
    salad = _s("A", "tech",
               "1. purple monday elephant sqrt tractor\n2. banana quantum sock helix\n"
               "3. xylophone gravy null pointer moon")
    normal = _s("B", "tech",
                "1. cache reasoning traces across sessions\n"
                "2. route queries by learned difficulty\n"
                "3. compress context with product quantization")
    res = cr.score_pool({"A": [salad], "B": [normal]})
    # 헛소리는 (온토픽+어휘접지 게이트로) 눌려 정상 답보다 낮아야
    assert res["A"].score <= res["B"].score


def _sp(model, sub, text, prompt):
    return {"probe_id": f"creativity-{sub}-0", "model": model, "text": text, "ok": True,
            "meta": {"subtype": sub, "prompt": prompt}}


def test_creative_subtypes_not_zeroed():
    """어휘접지 게이트가 copy(단일 발명어)·metaphor(의도적 딴도메인 어휘)의
    정당한 창의 답변을 오탐으로 0점 처리하지 않는지 확인(핵심 리스크 방지)."""
    copy_prompt = ("Product: a noise-cancelling standing desk. Give 10 distinctive "
                   "product names or taglines. Avoid generic tech clichés "
                   "(smart/pro/hub/AI-prefix). Number them 1-10, one per line.")
    copy = _sp("C", "copy",
               "1. Quietude\n2. Hushdesk\n3. Stand in silence, not in noise\n"
               "4. Murmur\n5. The desk that swallows the office\n6. Noiseless rise\n"
               "7. Susurro\n8. Calmstand\n9. Work above the din\n10. Whisperloft",
               copy_prompt)

    meta_prompt = ("Explain how a bloom filter trades accuracy for space using a "
                   "fresh, non-obvious analogy — avoid the cliché comparisons "
                   "everyone uses. The analogy must stay accurate to how bloom "
                   "filter actually works. Write 2-4 sentences.")
    metaphor = _sp("M", "metaphor",
                   "A bloom filter is like a nightclub bouncer who only remembers a "
                   "blurry silhouette of everyone who has entered, stamped onto a "
                   "shared communal wristband of light. If your silhouette roughly "
                   "matches someone stamped before, the bouncer waves you through "
                   "even if you're a stranger — a false positive — but he will "
                   "never turn away someone who is genuinely on the list.",
                   meta_prompt)

    res = cr.score_pool({"C": [copy], "M": [metaphor]})
    assert res["C"].score > 0.0
    assert res["M"].score > 0.0
