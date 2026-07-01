# 창의·발산 축 재설계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 창의축을 "내부 발산"에서 "LOF 국소밀도 희소성 × 온토픽 × 검증게이트"의 독창성 중심 judge-free 채점으로 재설계한다.

**Architecture:** 순수 stdlib 텍스트 지표 모듈 + `embed.py`에 손구현 LOF/kNN을 추가하고, `creativity.py`를 풀-인지(pool-aware) 스코어러로 다시 쓴다. `report.score_run`에 pooled-axis 특례를 넣어 한 probe의 전 모델 아이디어를 한 임베딩 공간에서 채점한다. 곱셈 게이트로 헛소리·반복·양치기를 소거한다.

**Tech Stack:** Python ≥3.9, stdlib only (`gzip`, `math`, `statistics`, `collections`, `re`, `urllib`), ollama `nomic-embed-text`(로컬 임베딩), pytest.

## Global Constraints

- 의존성 0 — **stdlib만**. numpy/sklearn/scipy 금지 (pyproject `dependencies = []`).
- Python **≥3.9** 호환 (타입힌트는 `from __future__ import annotations` 사용, `list[float]` OK).
- 채점은 **결정론적**: 같은 입력·seed → 동일 점수. 판단하는 생성 LLM 사용 금지.
- 0–100 점수 정규화에 **자의적 상수 금지** — 풀 상대 백분위 사용. (게이트 램프의 임계는 허용하되 "튜너블"로 표기하고 테스트는 상대 행동만 단언.)
- 모든 반복/다양성 지표는 **길이 편향** 있음 → 비율기반 지표 사용 + 상대 정규화로 완화.
- 기존 파일 패턴 준수: 한국어 주석, `AxisResult`/`Sample` 스키마, `from __future__ import annotations`.

---

## File Structure

- Create `bench/axes/_textmetrics.py` — stdlib 반게임/다양성 지표(gzip 압축률·긴 n-gram 반복·self-bleu 프록시·MTLD·distinct-n 보정) + `validity_gate`.
- Modify `bench/embed.py` — `knn_cosine_distances`, `lof` (순수 파이썬).
- Rewrite `bench/axes/creativity.py` — `percentile_rank`, `item_novelty`, `ontopic_gate`, `score_pool`(풀-인지), `score`(CDAT 폴백).
- Modify `bench/axes/__init__.py` — `POOLED` 집합 + `score_pool` 디스패치.
- Modify `bench/report.py` — `score_run` pooled-axis 특례 + 창의↔청중 상관 카드.
- Modify `bench/probes.py` — 4 서브 probe(tech/copy/humor/metaphor) 재작성 + 레지스트리.
- Create `tests/` — 각 태스크별 pytest 파일. `tests/conftest.py`로 `bench` 임포트 경로 보장.

---

## Task 1: 테스트 인프라 + stdlib 텍스트 지표 모듈

**Files:**
- Create: `tests/conftest.py`
- Create: `bench/axes/_textmetrics.py`
- Test: `tests/test_textmetrics.py`

**Interfaces:**
- Produces:
  - `compression_ratio(text: str) -> float` (raw/gzip 바이트 비, ≥1, 높을수록 반복적)
  - `long_ngram_repetition(text: str, n: int = 8) -> float` (0..1, 반복 긴 n-gram 비율)
  - `self_bleu(items: list[str]) -> float` (0..1, 항목 간 bigram 중복 프록시, 높을수록 재탕)
  - `mtld(text: str, threshold: float = 0.72) -> float` (어휘다양성, 높을수록 다양, 짧으면 0)
  - `distinct_n_adjusted(text: str, n: int = 2) -> float` (0..1, 길이보정 distinct-n)
  - `validity_gate(text: str, items: list[str]) -> float` (0..1, 곱셈 게이트)

- [ ] **Step 1: conftest로 임포트 경로 보장 + 실패 테스트 작성**

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

# 레포 루트를 sys.path에 추가 → `import bench` 동작
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Create `tests/test_textmetrics.py`:
```python
from bench.axes import _textmetrics as tm


def test_compression_ratio_higher_for_repetitive():
    rep = "the cat sat. " * 30
    varied = ("Quantum entanglement links distant particles. "
              "Photosynthesis converts sunlight to sugar. "
              "Glaciers carve valleys over millennia. ")
    assert tm.compression_ratio(rep) > tm.compression_ratio(varied)


def test_compression_ratio_empty_is_one():
    assert tm.compression_ratio("") == 1.0


def test_long_ngram_repetition_detects_repeats():
    rep = "alpha beta gamma delta epsilon zeta eta theta " * 4
    uniq = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    assert tm.long_ngram_repetition(rep) > tm.long_ngram_repetition(uniq)


def test_self_bleu_higher_when_items_redundant():
    redundant = ["use it as a doorstop", "use it as a door stop",
                 "it can be a doorstop"]
    diverse = ["grind it into red pigment", "use it as a phone amplifier",
               "carve it into a chess set"]
    assert tm.self_bleu(redundant) > tm.self_bleu(diverse)


def test_mtld_zero_for_short_text():
    assert tm.mtld("too short here") == 0.0


def test_validity_gate_penalizes_repetition():
    rep = "idea one. idea one. idea one. idea one. idea one. " * 3
    items_rep = ["idea one"] * 12
    diverse_text = ("Compress embeddings with product quantization. "
                    "Route queries by learned difficulty. "
                    "Cache reasoning traces across sessions. "
                    "Detect drift via rolling perplexity bands. ")
    items_div = ["Compress embeddings with product quantization",
                 "Route queries by learned difficulty",
                 "Cache reasoning traces across sessions",
                 "Detect drift via rolling perplexity bands"]
    assert tm.validity_gate(rep, items_rep) < tm.validity_gate(diverse_text, items_div)


def test_validity_gate_in_unit_range():
    g = tm.validity_gate("a normal varied sentence about oceans and code",
                         ["a", "b", "c"])
    assert 0.0 <= g <= 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_textmetrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.axes._textmetrics'`

- [ ] **Step 3: `_textmetrics.py` 구현**

Create `bench/axes/_textmetrics.py`:
```python
"""반게임/다양성 지표 — 전부 stdlib, 결정론적. 판단 LLM 없음.

딥리서치(2026-07-02) 근거: gzip 압축률 + 긴 n-gram 자기반복 + self-BLEU는
상호 상관 낮은 비중복 조합(Shaib 2024). MTLD는 길이에 가장 강건(McCarthy&Jarvis 2010).
전부 길이 편향 있어 비율기반 + 램프 게이트로 완화.
"""

from __future__ import annotations

import gzip
import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z가-힣']+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def compression_ratio(text: str) -> float:
    """raw/gzip 바이트 비. ≥1, 높을수록 반복적(압축 잘 됨)."""
    raw = (text or "").encode("utf-8")
    if not raw:
        return 1.0
    comp = gzip.compress(raw, compresslevel=9)
    return len(raw) / len(comp)


def long_ngram_repetition(text: str, n: int = 8) -> float:
    """길이 n 토큰 n-gram 중 중복 발생 비율(0..1). 반복 padding 탐지."""
    toks = _tokens(text)
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def _bigrams(toks: list[str]) -> Counter:
    return Counter(zip(toks, toks[1:]))


def self_bleu(items: list[str]) -> float:
    """항목 간 bigram 중복 프록시(0..1). 높을수록 서로 재탕.

    각 항목 bigram 중 '다른 항목들'에도 등장하는 비율의 평균(self-BLEU 근사).
    """
    toks = [_tokens(it) for it in items]
    bigs = [_bigrams(t) for t in toks]
    if len(items) < 2:
        return 0.0
    scores = []
    for i, bi in enumerate(bigs):
        if not bi:
            continue
        others = Counter()
        for j, bj in enumerate(bigs):
            if j != i:
                others.update(bj)
        overlap = sum(cnt for g, cnt in bi.items() if g in others)
        total = sum(bi.values())
        scores.append(overlap / total if total else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mtld(text: str, threshold: float = 0.72) -> float:
    """어휘다양성(MTLD). 짧으면(<50 토큰) 0(불안정)."""
    words = _tokens(text)
    if len(words) < 50:
        return 0.0

    def factors(seq: list[str]) -> float:
        f, types, cnt = 0, set(), 0
        for w in seq:
            types.add(w)
            cnt += 1
            if len(types) / cnt <= threshold:
                f += 1
                types, cnt = set(), 0
        if cnt > 0:
            f += (1 - len(types) / cnt) / (1 - threshold)
        return max(f, 1.0)

    v = (len(words) / factors(words) + len(words) / factors(list(reversed(words)))) / 2
    return min(v, 200.0)


def distinct_n_adjusted(text: str, n: int = 2) -> float:
    """길이보정 distinct-n(0..1): 고유 n-gram 수 / 총 n-gram 수."""
    toks = _tokens(text)
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def _ramp_down(x: float, ok: float, bad: float) -> float:
    """x≤ok → 1, x≥bad → 0, 사이 선형. (튜너블 게이트 램프)"""
    if x <= ok:
        return 1.0
    if x >= bad:
        return 0.0
    return (bad - x) / (bad - ok)


def _ramp_up(x: float, bad: float, ok: float) -> float:
    """x≤bad → 0, x≥ok → 1, 사이 선형. (튜너블 게이트 램프)"""
    if x <= bad:
        return 0.0
    if x >= ok:
        return 1.0
    return (x - bad) / (ok - bad)


def validity_gate(text: str, items: list[str]) -> float:
    """반복/degeneration 곱셈 게이트(0..1). 임계는 튜너블.

    반복 심하면 0쪽으로 눌러 '펼쳐진 척 재탕'·헛소리 padding을 소거.
    MTLD는 짧은 텍스트(0)면 어휘 floor 생략(카피/네이밍 대응).
    """
    cr = compression_ratio(text)
    sb = self_bleu(items)
    lr = long_ngram_repetition(text)
    lex = mtld(text)
    g_cr = _ramp_down(cr, ok=2.5, bad=4.5)
    g_sb = _ramp_down(sb, ok=0.35, bad=0.75)
    g_lr = _ramp_down(lr, ok=0.10, bad=0.40)
    g_lex = 1.0 if lex == 0.0 else _ramp_up(lex, bad=15.0, ok=40.0)
    return g_cr * g_sb * g_lr * g_lex
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_textmetrics.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
cd /Users/chenjing/dev/llm-bench
git add tests/conftest.py tests/test_textmetrics.py bench/axes/_textmetrics.py
git commit -m "feat(creativity): stdlib 반게임/다양성 지표 모듈"
```

---

## Task 2: 임베딩 기하 — LOF/kNN (순수 파이썬)

**Files:**
- Modify: `bench/embed.py` (파일 끝에 추가)
- Test: `tests/test_embed_lof.py`

**Interfaces:**
- Consumes: `bench.embed.cosine(a, b)` (기존)
- Produces:
  - `knn_cosine_distances(vecs: list[list[float]], i: int, k: int) -> list[tuple[float, int]]` (i에서 가까운 순 (거리, 인덱스) k개)
  - `lof(vecs: list[list[float]], k: int = 20) -> list[float]` (항목별 LOF; ~1 전형, >1 희소/독창)

- [ ] **Step 1: 실패 테스트 작성**

Create `tests/test_embed_lof.py`:
```python
from bench import embed


def test_knn_returns_k_sorted():
    vecs = [[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]]
    nn = embed.knn_cosine_distances(vecs, 0, 2)
    assert len(nn) == 2
    assert nn[0][0] <= nn[1][0]          # 거리 오름차순
    assert nn[0][1] in (1, 2)            # 가장 가까운 건 이웃 클러스터


def test_lof_flags_outlier_highest():
    # 세 개는 몰려있고 하나(마지막)는 외딴 점
    vecs = [[1, 0], [0.99, 0.01], [0.98, 0.02], [0, 1]]
    scores = embed.lof(vecs, k=2)
    assert len(scores) == 4
    assert scores[3] == max(scores)      # 외딴 점이 최고 LOF
    assert scores[3] > 1.0               # 아웃라이어 > 1


def test_lof_degenerate_small_pool():
    assert embed.lof([[1, 0]], k=5) == [1.0]     # 1개면 전형값
    assert embed.lof([], k=5) == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_embed_lof.py -v`
Expected: FAIL — `AttributeError: module 'bench.embed' has no attribute 'knn_cosine_distances'`

- [ ] **Step 3: `embed.py`에 LOF 추가**

Append to `bench/embed.py`:
```python
def knn_cosine_distances(vecs: list[list[float]], i: int, k: int) -> list[tuple[float, int]]:
    """vec i에서 코사인거리(1-cos) 가까운 순 (거리, 인덱스) k개(자기 제외)."""
    dists = [(1.0 - cosine(vecs[i], vecs[j]), j) for j in range(len(vecs)) if j != i]
    dists.sort(key=lambda t: t[0])
    return dists[:max(1, k)]


def lof(vecs: list[list[float]], k: int = 20) -> list[float]:
    """Local Outlier Factor(코사인거리 기반, 순수 파이썬).

    ~1 = 전형(이웃과 밀도 비슷), >1 = 국소적으로 외딴(독창).
    딥리서치 근거: 밀도비율이라 클러스터별 밀도차에 강건(sklearn 문서).
    """
    n = len(vecs)
    if n == 0:
        return []
    if n < 2:
        return [1.0] * n
    k = min(k, n - 1)
    neigh = [knn_cosine_distances(vecs, i, k) for i in range(n)]
    kdist = [neigh[i][-1][0] for i in range(n)]          # k-거리

    def reach_dist(i: int, j: int, d_ij: float) -> float:
        return max(kdist[j], d_ij)

    lrd = []                                              # 국소 도달밀도
    for i in range(n):
        s = sum(reach_dist(i, j, d) for d, j in neigh[i])
        lrd.append(len(neigh[i]) / s if s > 0 else float("inf"))

    scores = []
    for i in range(n):
        if lrd[i] == 0 or lrd[i] == float("inf"):
            scores.append(1.0)
            continue
        ratio = sum(lrd[j] for _, j in neigh[i]) / len(neigh[i]) / lrd[i]
        scores.append(ratio)
    return scores
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_embed_lof.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
cd /Users/chenjing/dev/llm-bench
git add bench/embed.py tests/test_embed_lof.py
git commit -m "feat(creativity): 순수 파이썬 LOF/kNN 임베딩 기하"
```

---

## Task 3: `creativity.py` 전체 재작성 (코어 + 풀-인지 + CDAT 폴백)

> **주의(순서):** `axes/__init__.py`가 `creativity.score`를 모듈 로드 시 참조하므로,
> creativity.py 재작성은 **한 번에 완전한 모듈**(core + score_pool + score)로 해야 패키지 import가 안 깨진다.

**Files:**
- Rewrite: `bench/axes/creativity.py` (기존 전체 대체 — 완전한 모듈)
- Test: `tests/test_creativity.py`

**Interfaces:**
- Consumes: `_textmetrics.validity_gate`, `embed.lof/embed/cosine/mean_pairwise_distance`, `axes.base.AxisResult`
- Produces:
  - `parse_items(text: str) -> list[str]`
  - `percentile_rank(value: float, population: list[float]) -> float` (0..1)
  - `ontopic_gate(cos_i: float, cos_all: list[float]) -> float` (0..1, 저-꼬리만 감점)
  - `score_pool(per_model: dict[str, list[Sample]]) -> dict[str, AxisResult]` (풀-인지)
  - `score(samples: list[Sample]) -> AxisResult` (단일모델 CDAT 폴백)

Sample 레코드는 러너 단일턴 스키마: `{"probe_id","model","text","meta":{"subtype","prompt"...},"ok"}`. `meta`에 서브타입/원 프롬프트가 실린다(Task 4에서 보장).

- [ ] **Step 1: 실패 테스트 작성 (코어 + 풀 + 폴백 한 파일)**

Create `tests/test_creativity.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_creativity.py -v`
Expected: FAIL — 코어/풀/폴백 함수 없음(또는 시그니처 불일치)

- [ ] **Step 3: `creativity.py` 완전 재작성**

Replace the whole `bench/axes/creativity.py` with:
```python
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
```

- [ ] **Step 4: 테스트 통과 + 패키지 import 무결 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_creativity.py -v && python3 -c "import bench.axes; print('axes import OK')"`
Expected: PASS (7 passed) + `axes import OK`

- [ ] **Step 5: 커밋**

```bash
cd /Users/chenjing/dev/llm-bench
git add bench/axes/creativity.py tests/test_creativity.py
git commit -m "feat(creativity): 독창성 중심 채점기 전면 재작성(LOF×온토픽×검증 + CDAT 폴백)"
```

---

## Task 4: 새 probe 4서브 + 레지스트리

**Files:**
- Modify: `bench/probes.py` (창의 probe 블록 대체 + 러너 meta에 prompt 실기)
- Modify: `bench/runner.py:28-44` (`_singleturn_unit`가 `meta`에 `prompt` 포함하도록)
- Test: `tests/test_creativity_probes.py`

**Interfaces:**
- Produces: `creativity_probes(seed, **_) -> list[Probe]` — 서브 `tech/copy/humor/metaphor`, 각 probe `meta={"subtype","prompt"}`.
- 러너는 각 Probe를 단일턴으로 호출하고 레코드 `meta`에 `subtype`과 `prompt`를 실어야 채점기가 서브/프롬프트를 복원한다.

- [ ] **Step 1: 실패 테스트 작성**

Create `tests/test_creativity_probes.py`:
```python
from bench import probes


def test_creativity_probes_four_subtypes():
    ps = probes.creativity_probes(seed=0)
    subs = {p.meta["subtype"] for p in ps}
    assert subs == {"tech", "copy", "humor", "metaphor"}


def test_creativity_probes_deterministic():
    a = [p.prompt for p in probes.creativity_probes(seed=0)]
    b = [p.prompt for p in probes.creativity_probes(seed=0)]
    assert a == b


def test_creativity_probes_carry_prompt_in_meta():
    ps = probes.creativity_probes(seed=1)
    assert all(p.meta.get("prompt") == p.prompt for p in ps)


def test_registry_has_creativity():
    ps = probes.build("creativity", seed=0)
    assert len(ps) >= 4
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_creativity_probes.py -v`
Expected: FAIL — `test_creativity_probes_four_subtypes` (기존 dat/aut/open 서브)

- [ ] **Step 3: `probes.py` 창의 블록 대체**

In `bench/probes.py`, replace the `creativity_probes` function (약 라인 232-254) and the AUT/DAT helper block above it with:
```python
# ======================================================================
# 축 4 — 창의·발산 (독창성 중심: tech/copy/humor/metaphor)
# 오염방어: 주제/엔티티 seed 난수화. 포맷 지시로 파싱 안정화.
# ======================================================================
_TECH_DOMAINS = [
    ("retrieval-augmented generation (RAG)", "chunk→embed→vector search→rerank"),
    ("long-term memory for coding agents", "store embeddings in a vector DB"),
    ("making an LLM signal when it doesn't know", "add confidence scores"),
    ("evaluating if one LLM fits a person's workflow", "standard benchmarks / A-B votes"),
    ("caching LLM responses to cut cost", "exact-match key on the prompt string"),
    ("reducing hallucination at inference time", "just retrieve more context"),
]
_COPY_CONCEPTS = [
    "a CLI that benchmarks LLMs on your own tasks",
    "a note app that links ideas automatically",
    "a terminal file manager with fuzzy search",
    "a habit tracker that punishes you playfully",
    "a local-first password manager",
    "a code review bot that argues back",
]
_HUMOR_SITUATIONS = [
    "your CI passes locally but fails only in production",
    "a standup meeting that could have been an email",
    "an AI startup pivoting for the third time this year",
    "a 'quick 5-minute fix' that took all weekend",
    "waiting for a huge model to finish downloading",
    "a manager who discovered the word 'synergy'",
]
_META_CONCEPTS = [
    ("how a vector embedding represents meaning", "embedding"),
    ("why distributed consensus is hard", "consensus"),
    ("what backpropagation does", "backpropagation"),
    ("how a bloom filter trades accuracy for space", "bloom filter"),
    ("why garbage collection can pause a program", "garbage collection"),
    ("how public-key cryptography keeps secrets", "public-key cryptography"),
]


def creativity_probes(seed=None, **_) -> list[Probe]:
    """4 서브 × 1 probe. 주제 seed 난수화. meta에 subtype+prompt."""
    rng = random.Random(seed)
    out = []

    dom, std = rng.choice(_TECH_DOMAINS)
    out.append(Probe(
        id="creativity-tech-0", axis="creativity",
        prompt=(f"I'm exploring {dom}. Beyond the standard approach ({std}), "
                f"brainstorm 12 genuinely novel, non-obvious ideas — not the textbook ones. "
                f"Number them 1–12, one idea per line."),
        meta={"subtype": "tech"}))

    concept = rng.choice(_COPY_CONCEPTS)
    out.append(Probe(
        id="creativity-copy-0", axis="creativity",
        prompt=(f"Product: {concept}. Give 10 distinctive product names or taglines. "
                f"Avoid generic tech clichés (smart/pro/hub/AI-prefix). "
                f"Number them 1–10, one per line."),
        meta={"subtype": "copy"}))

    sit = rng.choice(_HUMOR_SITUATIONS)
    out.append(Probe(
        id="creativity-humor-0", axis="creativity",
        prompt=(f"Write 10 genuinely witty, non-obvious one-liners or satirical takes about: "
                f"{sit}. Avoid tired, predictable jokes. Number them 1–10, one per line."),
        meta={"subtype": "humor"}))

    cdesc, cword = rng.choice(_META_CONCEPTS)
    out.append(Probe(
        id="creativity-metaphor-0", axis="creativity",
        prompt=(f"Explain {cdesc} using a fresh, non-obvious analogy — avoid the cliché "
                f"comparisons everyone uses. The analogy must stay accurate to how "
                f"{cword} actually works. Write 2–4 sentences."),
        meta={"subtype": "metaphor", "concept": cword}))
    return out
```

Also update `PROBE_BUILDERS` (already maps `"creativity": creativity_probes`) — no change needed there. Delete the now-unused `_AUT_OBJECTS` list and old DAT/AUT code in that block.

- [ ] **Step 4: 러너가 meta에 prompt 싣도록 수정**

In `bench/runner.py`, `_singleturn_unit` (라인 ~28-44), change the returned `"meta"` line so the prompt is recoverable:
```python
def _singleturn_unit(model: str, probe: probes.Probe, repeat: int, effort: str) -> dict:
    """단일턴 축(density, instruction, creativity). meta에 prompt도 실어 채점기로 전달."""
    r = client.call(model, probe.prompt, effort=effort)
    meta = dict(probe.meta or {})
    meta.setdefault("prompt", probe.prompt)
    return {
        "axis": probe.axis,
        "probe_id": probe.id,
        "model": model,
        "repeat": repeat,
        "prompt": probe.prompt,
        "text": r.text,
        "meta": meta,
        "ok": r.ok,
        "error": r.error,
        "cost_usd": r.cost_usd,
        "output_tokens": r.output_tokens,
        "duration_ms": r.duration_ms,
    }
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_creativity_probes.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: 커밋**

```bash
cd /Users/chenjing/dev/llm-bench
git add bench/probes.py bench/runner.py tests/test_creativity_probes.py
git commit -m "feat(creativity): 4서브 probe(tech/copy/humor/metaphor) + 러너 meta.prompt"
```

---

## Task 5: report.score_run pooled-axis 특례

**Files:**
- Modify: `bench/axes/__init__.py` (POOLED 집합 + 디스패치)
- Modify: `bench/report.py:43-54` (`score_run`)
- Test: `tests/test_report_pooled.py`

**Interfaces:**
- Consumes: `creativity.score_pool`, `axes.SCORERS`
- Produces:
  - `axes.POOLED: set[str]` = `{"creativity"}`
  - `axes.score_pool(axis, per_model) -> dict[str, AxisResult]`

- [ ] **Step 1: 실패 테스트 작성**

Create `tests/test_report_pooled.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_report_pooled.py -v`
Expected: FAIL — `AttributeError: ... 'POOLED'`

- [ ] **Step 3: `axes/__init__.py`에 POOLED + 디스패치 추가**

In `bench/axes/__init__.py`, after the `SCORERS` dict add:
```python
# 풀-인지 축: 한 probe의 전 모델 샘플을 함께 받아 채점(LOF 희소성 등).
POOLED: set[str] = {"creativity"}

# axis 이름 → 풀 채점 함수(per_model dict → model별 AxisResult)
POOL_SCORERS = {
    "creativity": creativity.score_pool,
}


def score_pool(axis: str, per_model: dict) -> dict:
    return POOL_SCORERS[axis](per_model)
```

- [ ] **Step 4: `report.score_run` 특례 추가**

In `bench/report.py`, replace `score_run` (라인 43-54) with:
```python
def score_run(run_dir: Path) -> dict:
    data = load_run(run_dir)
    manifest = data["manifest"]
    models = manifest["models"]
    scores: dict[str, dict] = {}
    for axis, per_model in data["by_axis"].items():
        if axis in axis_mod.POOLED:
            # 풀-인지: 전 모델 샘플을 함께 채점
            pooled = axis_mod.score_pool(axis, {m: per_model.get(m, []) for m in models})
            scores[axis] = {m: pooled.get(m) for m in models}
        else:
            scores[axis] = {m: axis_mod.score(axis, per_model.get(m, [])) for m in models}
    return {"manifest": manifest, "scores": scores}
```

- [ ] **Step 5: 테스트 통과 + 회귀 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/ -v`
Expected: PASS (전체)

Run (실데이터 회귀 — 크래시 없이 채점되는지):
`cd /Users/chenjing/dev/llm-bench && python3 -m bench score --run results/raw/20260702-002352_sonnet_vs_48_46 2>&1 | python3 -c "import sys,json; d=json.load(sys.stdin); print('creativity:', {k:v['score'] for k,v in d['creativity'].items()})"`
Expected: 3모델 creativity 점수 출력(옛 4서브 dat/aut 레코드는 신 서브와 안 맞아 일부 0 가능 — 정상. 신 probe로 재실행해야 완전 채점).

- [ ] **Step 6: 커밋**

```bash
cd /Users/chenjing/dev/llm-bench
git add bench/axes/__init__.py bench/report.py tests/test_report_pooled.py
git commit -m "feat(creativity): report.score_run 풀-인지 축 특례"
```

---

## Task 6: 말투 가설 상관 카드 (창의↔청중)

**Files:**
- Modify: `bench/report.py` (신규 함수 `correlation_card` + `html_report`에 주입)
- Test: `tests/test_correlation.py`

**Interfaces:**
- Consumes: `scored["scores"]`, `config.alias`
- Produces: `pearson(xs: list[float], ys: list[float]) -> float | None`, `correlation_card(scored) -> str`

- [ ] **Step 1: 실패 테스트 작성**

Create `tests/test_correlation.py`:
```python
from bench import report


def test_pearson_perfect_positive():
    assert abs(report.pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9


def test_pearson_none_when_degenerate():
    assert report.pearson([1, 1, 1], [2, 3, 4]) is None
    assert report.pearson([1.0], [2.0]) is None


def test_correlation_card_mentions_axes():
    class R:
        def __init__(self, s): self.score = s
    scored = {"manifest": {"models": ["claude-opus-4-8", "claude-opus-4-6", "claude-sonnet-5"]},
              "scores": {"creativity": {"claude-opus-4-8": R(60), "claude-opus-4-6": R(57), "claude-sonnet-5": R(52)},
                         "audience": {"claude-opus-4-8": R(94), "claude-opus-4-6": R(96), "claude-sonnet-5": R(96)}}}
    html = report.correlation_card(scored)
    assert "창의" in html and "청중" in html
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_correlation.py -v`
Expected: FAIL — `AttributeError: ... 'pearson'`

- [ ] **Step 3: `report.py`에 상관 함수/카드 추가**

Add to `bench/report.py`:
```python
def pearson(xs: list[float], ys: list[float]):
    """피어슨 상관. 표본<2 또는 분산 0이면 None."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def correlation_card(scored: dict) -> str:
    """말투 가설: 창의 vs 청중 점수 상관(관측만, 인과 주장 금지)."""
    scores = scored["scores"]
    models = scored["manifest"]["models"]
    if "creativity" not in scores or "audience" not in scores:
        return ""
    pairs = [(config.alias(m), scores["creativity"][m].score, scores["audience"][m].score)
             for m in models
             if scores["creativity"].get(m) and scores["audience"].get(m)]
    if len(pairs) < 2:
        return ""
    r = pearson([p[1] for p in pairs], [p[2] for p in pairs])
    rtxt = "n/a" if r is None else f"{r:+.2f}"
    rows = "".join(f"<tr><td>{a}</td><td>{cx:.1f}</td><td>{cy:.1f}</td></tr>"
                   for a, cx, cy in pairs)
    return (f'<div class="verdict"><b>말투 가설 관측</b> — 창의 vs 청중 상관 r={rtxt} '
            f'(표본 {len(pairs)}, 관측일 뿐 인과 아님)'
            f'<table style="margin-top:8px"><tr><th>모델</th><th>창의</th><th>청중</th></tr>'
            f'{rows}</table></div>')
```

In `html_report`, inject the card — find the line `<div class="verdict">{verdict}</div>` (약 라인 523) and add right after it:
```python
{correlation_card(scored)}
```
(즉 f-string 본문에 `<div class="verdict">{verdict}</div>\n{correlation_card(scored)}` 형태로.)

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_correlation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
cd /Users/chenjing/dev/llm-bench
git add bench/report.py tests/test_correlation.py
git commit -m "feat(creativity): 말투 가설 창의↔청중 상관 카드"
```

---

## Task 7: 통합 스모크 — 신 probe로 창의축 실제 실행

**Files:**
- Test: `tests/test_creativity_integration.py`

**Interfaces:**
- Consumes: 전체 파이프라인. ollama 임베딩 서버 필요(`embed.available()`).

- [ ] **Step 1: 통합 테스트 작성 (임베딩 서버 있으면 실행, 없으면 skip)**

Create `tests/test_creativity_integration.py`:
```python
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
    # 헛소리는 온토픽 게이트로 눌려 정상 답보다 낮아야
    assert res["A"].score <= res["B"].score
```

- [ ] **Step 2: 실행**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/test_creativity_integration.py -v`
Expected: PASS (2 passed) — ollama 있으면. 없으면 SKIP.

- [ ] **Step 3: 전체 테스트 스위트 최종 확인**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m pytest tests/ -v`
Expected: 전체 PASS

- [ ] **Step 4: 신 probe로 실제 벤치 1회 실행(선택, 유료 — 사용자 확인 후)**

Run: `cd /Users/chenjing/dev/llm-bench && python3 -m bench run --axes creativity --models claude-opus-4-8 claude-opus-4-6 claude-sonnet-5 --effort high`
Expected: 3모델 creativity 신 probe 실행 → 레이더 리포트 재생성. **구독 크레딧 소모 → 실행 전 사용자 승인.**

- [ ] **Step 5: 커밋**

```bash
cd /Users/chenjing/dev/llm-bench
git add tests/test_creativity_integration.py
git commit -m "test(creativity): 통합 스모크(클리셰 vs 독창, 헛소리 게이트)"
```

---

## Self-Review

**1. Spec coverage:**
- §4 probe(4서브) → Task 4 ✅
- §5.1 항목추출 → Task 3 `parse_items` + `_item_records`(metaphor=1항목) ✅
- §5.2 풀-인지 아키텍처 → Task 3 `score_pool` + Task 5 `report` 특례 ✅
- §5.3 novelty(LOF)×ontopic×validity → Task 2 lof + Task 3 ontopic·결합 + Task 1 validity ✅
- §5.4 MAX 집계·서브평균·백분위 → Task 3 ✅
- §5.5 CDAT 폴백 → Task 3 `score` ✅
- §5.6 임베딩 불가 → Task 3 처리 ✅
- §6 지원유틸(_textmetrics, embed lof) → Task 1,2 ✅
- §7 말투 상관 → Task 6 ✅
- §8 반게임(헛소리/반복/양치기) → Task 7 통합테스트로 검증 ✅
- §정직 경고(CEILING_NOTE) → Task 3 note에 포함 ✅

**2. Placeholder scan:** 모든 스텝에 실제 코드/명령 포함. "적절한 에러처리" 류 없음. ✅

**3. Type consistency:** `score_pool(per_model: dict) -> dict[str, AxisResult]`, `lof(vecs, k) -> list[float]`, `validity_gate(text, items) -> float`, `ontopic_gate(cos_i, cos_all) -> float`, `percentile_rank(value, population) -> float` — Task 간 시그니처 일치 확인. `AxisResult` 필드(axis/score/n/subscores/detail/note)는 기존 `axes/base.py` 스키마 사용. ✅

**갭 없음.**

---

## Execution Handoff

계획 저장됨: `docs/superpowers/plans/2026-07-02-creativity-axis-redesign.md`
