"""축 4 — 창의·발산 (독창성 중심, judge-free, 순수 기계 채점).

독창 = LOF 국소밀도 희소성(풀 임베딩, MAX 1개) × 온토픽 게이트(probe별 상대)
       × 검증 게이트(gzip·긴n-gram·self-bleu·MTLD, _textmetrics).
풀 없으면 CDAT 폴백(내부 발산 × 온토픽). 정규화는 풀 상대 백분위.

⚠️ 정직 경고: 임베딩 독창성은 인간 상관 ~0.2–0.3, 임베딩 키워도 안 오름
(Organisciak 2023). 점수는 정밀 등급이 아니라 거친 순위로 해석.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean, pstdev

from .. import embed
from . import _textmetrics as tm
from .base import AxisResult, Sample

_LINE_ITEM = re.compile(r"^\s*(?:\d+[.):\]]|[-*•·])\s*(.+)$")

CEILING_NOTE = "임베딩 독창성 인간상관 ~0.2–0.3; 거친 순위로만 해석."


def parse_items(text: str) -> list[str]:
    items = []
    for line in (text or "").splitlines():
        m = _LINE_ITEM.match(line)
        if m:
            it = re.sub(r"\*+", "", m.group(1)).strip().rstrip(".")
            if it:
                items.append(it)
    if not items:
        items = [p.strip(" .*") for p in re.split(r"[,;\n]", text or "") if p.strip(" .*")]
    return items


def percentile_rank(value: float, population: list[float]) -> float:
    """population 대비 value의 백분위(0..1). 자의적 상수 없는 상대 정규화."""
    if not population:
        return 0.5
    below = sum(1 for p in population if p < value)
    equal = sum(1 for p in population if p == value)
    return (below + 0.5 * equal) / len(population)


def ontopic_gate(cos_i: float, cos_all: list[float]) -> float:
    """probe 내 프롬프트-코사인 분포 기준, 저-꼬리(딴소리)만 감점(0..1).

    딥리서치: 절대 커트라인 금지(Rossi 2024) → probe 내 z-score 시그모이드.
    전형 항목(평균 이상)은 ~1 통과, 저-꼬리 아웃라이어만 게이트.
    """
    if len(cos_all) < 2:
        return 1.0
    mu = mean(cos_all)
    sd = pstdev(cos_all) or 1e-6
    z = (cos_i - mu) / sd
    return 1.0 / (1.0 + math.exp(-(z + 1.5) * 2.0))


def _prompt_of(s: Sample) -> str:
    return (s.get("meta") or {}).get("prompt") or s.get("prompt") or ""


def _sub_of(s: Sample) -> str:
    return (s.get("meta") or {}).get("subtype", "open")


def _item_records(samples: list[Sample]) -> list[dict]:
    """샘플 → 아이템 단위 레코드(모델/서브/프롬프트/원문). metaphor는 응답 전체=1항목."""
    recs = []
    for s in samples:
        if not s.get("ok", True):
            continue
        sub = _sub_of(s)
        text = s.get("text", "") or ""
        items = parse_items(text) if sub != "metaphor" else [text.strip()]
        for it in items:
            if it:
                recs.append({"model": s["model"], "sub": sub,
                             "probe_id": s.get("probe_id"), "prompt": _prompt_of(s),
                             "item": it, "resp_text": text, "resp_items": items})
    return recs


def _score_from_recs(recs: list[dict]) -> dict[str, dict]:
    """(probe,sub) 풀별 LOF 희소성 × 온토픽 × 검증 → 모델별 서브점수 누적."""
    by_pool = defaultdict(list)
    for r in recs:
        by_pool[(r["probe_id"], r["sub"])].append(r)

    acc: dict[str, dict] = defaultdict(lambda: {"subs": defaultdict(list), "detail": []})
    for (pid, sub), pool in by_pool.items():
        texts = [r["item"] for r in pool]
        vecs = embed.embed(texts)
        prompt_vec = embed.embed([pool[0]["prompt"]])[0] if pool[0]["prompt"] else None
        lofs = embed.lof(vecs, k=20)
        cos_all = ([embed.cosine(v, prompt_vec) for v in vecs]
                   if prompt_vec is not None else [1.0] * len(vecs))
        for r, lf, cos_i in zip(pool, lofs, cos_all):
            novelty = percentile_rank(lf, lofs)
            ot = ontopic_gate(cos_i, cos_all)
            val = tm.validity_gate(r["resp_text"], r["resp_items"])
            item_score = novelty * ot * val
            acc[r["model"]]["subs"][sub].append(item_score)
            acc[r["model"]]["detail"].append(
                {"probe_id": pid, "sub": sub, "item": r["item"][:80],
                 "lof": round(lf, 3), "novelty": round(novelty, 3),
                 "ontopic": round(ot, 3), "validity": round(val, 3),
                 "item_score": round(item_score, 3)})
    return acc


def score_pool(per_model: dict[str, list[Sample]]) -> dict[str, AxisResult]:
    """풀-인지 채점: 한 probe의 전 모델 아이디어를 한 공간에서 LOF 희소성."""
    models = list(per_model)
    if not embed.available():
        return {m: AxisResult(axis="creativity", score=0.0, n=0,
                              note="ollama 임베딩 서버 불가 — 채점 생략") for m in models}
    all_recs = []
    for m in models:
        all_recs += _item_records(per_model[m])
    acc = _score_from_recs(all_recs)

    results = {}
    for m in models:
        a = acc.get(m)
        if not a or not a["subs"]:
            results[m] = AxisResult(axis="creativity", score=0.0, n=0, note="no samples")
            continue
        sub_max = {sub: max(scores) for sub, scores in a["subs"].items() if scores}
        model_score = mean(sub_max.values()) if sub_max else 0.0
        n_items = sum(len(s) for s in a["subs"].values())
        results[m] = AxisResult(
            axis="creativity", score=round(model_score * 100, 2), n=n_items,
            subscores={f"max_{sub}": round(v * 100, 1) for sub, v in sub_max.items()},
            detail=a["detail"], note="독창=LOF희소성(MAX)×온토픽×검증. " + CEILING_NOTE)
    return results


def score(samples: list[Sample]) -> AxisResult:
    """단일모델 CDAT 폴백: 내부 발산 × 온토픽 × 검증(풀/상대비교 아님)."""
    if not embed.available():
        return AxisResult(axis="creativity", score=0.0, n=0,
                          note="ollama 임베딩 서버 불가 — 채점 생략")
    recs = _item_records(samples)
    if not recs:
        return AxisResult(axis="creativity", score=0.0, n=0, note="no samples")
    by_pool = defaultdict(list)
    for r in recs:
        by_pool[(r["probe_id"], r["sub"])].append(r)
    sub_scores: dict[str, list[float]] = defaultdict(list)
    for (pid, sub), pool in by_pool.items():
        texts = [r["item"] for r in pool]
        vecs = embed.embed(texts)
        internal = embed.mean_pairwise_distance(vecs)
        prompt_vec = embed.embed([pool[0]["prompt"]])[0] if pool[0]["prompt"] else None
        cos_all = ([embed.cosine(v, prompt_vec) for v in vecs]
                   if prompt_vec is not None else [1.0] * len(vecs))
        ot = mean(ontopic_gate(c, cos_all) for c in cos_all) if cos_all else 1.0
        val = mean(tm.validity_gate(r["resp_text"], r["resp_items"]) for r in pool)
        sub_scores[sub].append(internal * ot * val)
    model_score = mean(mean(v) for v in sub_scores.values()) if sub_scores else 0.0
    return AxisResult(
        axis="creativity", score=round(min(model_score, 1.0) * 100, 2),
        n=len(recs), note="CDAT 폴백(내부발산×온토픽×검증, 상대비교 아님). " + CEILING_NOTE)
