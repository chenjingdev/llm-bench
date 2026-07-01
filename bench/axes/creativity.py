"""축 4 — 창의·발산 (judge-free + oracle-free, 로컬 임베딩).

연구 결론대로 **다양성 × 품질을 함께** 잰다(다양성 단독은 횡설수설이 만점).
세 probe 유형:
  · DAT  (Divergent Association): 무관한 단어 10개 → 평균 쌍별 의미거리(발산). r≈0.5(인간).
  · AUT  (Alternative Uses): 한 사물의 창의적 용도 → 의미 발산 × 유창성(중복 제거).
  · open (발산 오프닝): 서로 다른 6개 첫문장 → 상호 의미거리.
보조(어휘): MTLD(어휘다양성). 임베딩은 ollama nomic-embed-text(로컬, 무과금).

정규화: nomic-embed의 common-word 발산 관측범위[LO,HI]로 0–100 매핑(측정기 baseline).
"""

from __future__ import annotations

import re
from statistics import mean

from .. import embed
from .base import AxisResult, Sample

# nomic-embed-text의 실측 발산 범위(근사어 바닥 ~ 무관어 천장). 측정기 보정상수.
LO, HI = 0.15, 0.42

_LINE_ITEM = re.compile(r"^\s*(?:\d+[.):\]]|[-*•·])\s*(.+)$")


def _norm(mpd: float) -> float:
    return max(0.0, min(1.0, (mpd - LO) / (HI - LO)))


def parse_items(text: str) -> list[str]:
    items = []
    for line in (text or "").splitlines():
        m = _LINE_ITEM.match(line)
        if m:
            it = re.sub(r"\*+", "", m.group(1)).strip().rstrip(".")
            if it:
                items.append(it)
    if not items:  # fallback: 콤마/줄 구분(DAT가 한 줄로 올 때)
        items = [p.strip(" .*") for p in re.split(r"[,;\n]", text or "") if p.strip(" .*")]
    return items


def _dat_words(items: list[str]) -> list[str]:
    words = []
    for it in items:
        toks = re.findall(r"[A-Za-z]+", it)
        if toks and len(toks[0]) > 1:
            words.append(toks[0].lower())
    # 중복 제거(순서 유지)
    seen, uniq = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); uniq.append(w)
    return uniq[:10]


def _dedup_by_embedding(items: list[str], vecs: list[list[float]], thr: float = 0.92):
    keep_items, keep_vecs = [], []
    for it, v in zip(items, vecs):
        if all(embed.cosine(v, kv) < thr for kv in keep_vecs):
            keep_items.append(it); keep_vecs.append(v)
    return keep_items, keep_vecs


def mtld(text: str, threshold: float = 0.72) -> float:
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text or "")]
    if len(words) < 50:   # MTLD는 짧은 글에서 불안정 → 측정 안 함
        return 0.0

    def factors(seq):
        f, types, cnt = 0, set(), 0
        for w in seq:
            types.add(w); cnt += 1
            if len(types) / cnt <= threshold:
                f += 1; types, cnt = set(), 0
        if cnt > 0:
            f += (1 - len(types) / cnt) / (1 - threshold)
        return max(f, 1.0)   # 부분팩터 바닥 1.0 → 오버플로 방지

    v = (len(words) / factors(words) + len(words) / factors(list(reversed(words)))) / 2
    return min(v, 200.0)


def grade_one(sample: Sample) -> dict:
    sub = (sample.get("meta") or {}).get("subtype", "open")
    text = sample.get("text", "") or ""
    items = parse_items(text)
    row = {"probe_id": sample.get("probe_id"), "sub": sub, "mtld": round(mtld(text), 1)}

    if sub == "dat":
        words = _dat_words(items)
        vecs = embed.embed(words) if len(words) >= 2 else []
        mpd = embed.mean_pairwise_distance(vecs) if vecs else 0.0
        validity = len(words) / 10.0
        sc = _norm(mpd) * min(1.0, validity)
        row.update(n_items=len(words), mpd=round(mpd, 3), validity=round(validity, 2))

    elif sub == "aut":
        vecs0 = embed.embed(items) if len(items) >= 2 else []
        kept, kv = _dedup_by_embedding(items, vecs0) if vecs0 else ([], [])
        mpd = embed.mean_pairwise_distance(kv) if len(kv) >= 2 else 0.0
        fluency = min(1.0, len(kept) / 12.0)         # 12개 이상 유효 아이디어면 만점
        sc = _norm(mpd) * (0.6 + 0.4 * fluency)       # 발산 위주 + 유창성 보정
        row.update(n_raw=len(items), n_unique=len(kept), mpd=round(mpd, 3),
                   fluency=round(fluency, 2))

    else:  # open
        vecs = embed.embed(items) if len(items) >= 2 else []
        mpd = embed.mean_pairwise_distance(vecs) if vecs else 0.0
        sc = _norm(mpd)
        row.update(n_items=len(items), mpd=round(mpd, 3))

    row["score"] = round(sc * 100, 1)
    return row


def score(samples: list[Sample]) -> AxisResult:
    if not embed.available():
        return AxisResult(axis="creativity", score=0.0, n=0,
                          note="ollama 임베딩 서버 불가 — 채점 생략")
    rows = [grade_one(s) for s in samples]
    if not rows:
        return AxisResult(axis="creativity", score=0.0, n=0, note="no samples")

    def by_sub(sub, key):
        vals = [r[key] for r in rows if r["sub"] == sub and key in r]
        return round(mean(vals), 3) if vals else None

    subs = {
        "dat_mpd": by_sub("dat", "mpd"),
        "aut_mpd": by_sub("aut", "mpd"),
        "open_mpd": by_sub("open", "mpd"),
        "avg_mtld": round(mean(r["mtld"] for r in rows), 1),
        "aut_unique": by_sub("aut", "n_unique"),
    }
    return AxisResult(
        axis="creativity",
        score=round(mean(r["score"] for r in rows), 2),
        n=len(rows),
        subscores={k: v for k, v in subs.items() if v is not None},
        detail=rows,
        note="발산(평균 쌍별 의미거리, 로컬 임베딩) × 품질(유효/유창). nomic 범위[0.15,0.42]→0–100.",
    )
