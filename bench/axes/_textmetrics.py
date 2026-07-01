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
