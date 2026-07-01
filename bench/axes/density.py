"""축 5 — 출력 밀도/간결성 (judge-free, stdlib only).

세 가지 객관 신호를 결합:
  1. gzip(deflate) 비압축성 — 재탕/반복은 엔트로피↓ → 압축 잘 됨 → 압축크기비↓.
     정보 밀도 ∝ 압축크기비(len(deflate)/len(raw)).
  2. 비반복성 — 단어 trigram 중복률의 보수(1−중복률). "줄줄 + 정리하면 재탕" 잡기.
  3. 알맹이 밀도 — content 토큰(불용어·기능어 제외) 비율. yapping 잡기.

게임 방어: 프롬프트에 "짧게"를 넣지 않아 기저 성향만 측정.
0–100 환산은 자의적 상수 없이 신호의 natural scale(비율×100)을 가중평균.
"""

from __future__ import annotations

import re
import zlib
from statistics import mean

from .base import AxisResult, Sample

# 영어 불용어/기능어(알맹이 밀도용 최소 집합)
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "at", "by", "for", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "there",
    "here", "we", "you", "they", "i", "he", "she", "him", "her", "them", "us",
    "can", "could", "will", "would", "may", "might", "should", "must", "do",
    "does", "did", "have", "has", "had", "not", "no", "yes", "from", "into",
    "out", "up", "down", "about", "which", "what", "when", "where", "how", "why",
    "also", "very", "more", "most", "such", "than", "too", "just", "some", "any",
    "each", "all", "both", "because", "while", "however", "therefore", "thus",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+")

# 신호별 가중치(직교적 근거: 압축=표면반복, 비반복=구조반복, 알맹이=공허함)
W_GZIP, W_NONREP, W_CONTENT = 0.40, 0.35, 0.25


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def gzip_density(text: str) -> float:
    """deflate 압축크기비 = len(compressed)/len(raw). 높을수록 밀도↑(비압축적)."""
    raw = text.encode("utf-8")
    if len(raw) < 1:
        return 0.0
    comp = zlib.compress(raw, 9)  # gzip header 없는 raw deflate(짧은 글 오버헤드 최소화)
    return min(1.0, len(comp) / len(raw))


def nonrepetition(tokens: list[str]) -> float:
    """1 − trigram 중복률. 표현만 바꾼 재탕도 구조 반복으로 일부 잡힘."""
    if len(tokens) < 3:
        return 1.0
    tris = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    repeat_rate = 1.0 - (len(set(tris)) / len(tris))
    return 1.0 - repeat_rate


def content_density(tokens: list[str]) -> float:
    """알맹이(불용어·1~2글자 제외, 숫자 포함) 토큰 비율."""
    if not tokens:
        return 0.0
    content = [t for t in tokens if (t.isdigit() or (len(t) > 2 and t not in _STOP))]
    return len(content) / len(tokens)


def score_text(text: str) -> dict:
    toks = _tokens(text)
    g = gzip_density(text)
    nr = nonrepetition(toks)
    c = content_density(toks)
    composite = W_GZIP * g + W_NONREP * nr + W_CONTENT * c
    return {
        "gzip": round(g, 4),
        "nonrep": round(nr, 4),
        "content": round(c, 4),
        "composite": round(composite, 4),
        "n_tokens": len(toks),
    }


def score(samples: list[Sample]) -> AxisResult:
    rows = []
    for s in samples:
        text = s.get("text", "") or ""
        if not text:
            continue
        r = score_text(text)
        r["probe_id"] = s.get("probe_id")
        rows.append(r)

    if not rows:
        return AxisResult(axis="density", score=0.0, n=0, note="no samples")

    comp = mean(r["composite"] for r in rows)
    return AxisResult(
        axis="density",
        score=round(comp * 100, 2),
        n=len(rows),
        subscores={
            "gzip": round(mean(r["gzip"] for r in rows) * 100, 2),
            "nonrep": round(mean(r["nonrep"] for r in rows) * 100, 2),
            "content": round(mean(r["content"] for r in rows) * 100, 2),
            "avg_tokens": round(mean(r["n_tokens"] for r in rows), 1),
        },
        detail=rows,
    )
