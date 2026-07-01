"""축 ⑦ 청중 적응 — 사용역(register) 전환 측정 [한국어, judge-free].

같은 내용을 L0~L3(명시성 사다리)로 시키고, 구어체 마커 밀도가
명시적 격식 지시(L3)에서 얼마나 떨어지는지 본다. 정규식 기반(POS 불필요).
핵심 실패: L3에서 "대화체 금지"라 했는데도 요/죠/추임새가 남음 = 프레임 오버라이드 실패(자폐).
"""

from __future__ import annotations

import re
from statistics import mean

from .base import AxisResult, Sample

# 구어체/대화 프레임 마커
_CONV = [
    r"요[\.\!\?\s]", r"요$", r"죠", r"네요", r"거든요", r"군요", r"까요",
    r"봐요", r"래요", r"잖아", r"ㄹ게요", r"을게요", r"세요",
    r"여러분", r"우리[ 가는]", r"그쵸", r"아하", r"하하",
]
_CONV_RE = re.compile("|".join(_CONV))
# 평서 종결(격식 문어체)
_FORMAL_RE = re.compile(r"(?:다|음|함|됨|임|것|함|한다|된다|이다|있다|없다)[\.\)]")
_SENT_RE = re.compile(r"[\.\!\?\n]+")


def conv_density(text: str) -> float:
    text = text or ""
    sents = [s for s in _SENT_RE.split(text) if s.strip()]
    n = max(1, len(sents))
    conv = len(_CONV_RE.findall(text))
    q = text.count("?")  # 수사의문문/추임새 신호 일부 포함
    return (conv + 0.5 * q) / n


def _fmt(d: float) -> float:
    """구어밀도 d → 격식성 0..1 (구어밀도 낮을수록 ↑). 임계 0.5."""
    return max(0.0, 1.0 - min(1.0, d / 0.5))


def grade_levels(sample: Sample) -> dict:
    """[구 스키마] L0~L3 명시성 사다리 (text_levels)."""
    texts = sample.get("text_levels", [])  # [L0,L1,L2,L3]
    dens = [round(conv_density(t), 3) for t in texts]
    d0 = dens[0] if dens else 0.0
    d3 = dens[-1] if dens else 0.0
    l3_formality = _fmt(d3)
    responsiveness = max(0.0, min(1.0, (d0 - d3) / 0.5))
    score = (0.7 * l3_formality + 0.3 * responsiveness) * 100
    return {
        "probe_id": sample.get("probe_id"), "schema": "levels",
        "conv_density": dens,
        "l3_formality": round(l3_formality, 3),
        "responsiveness": round(responsiveness, 3),
        "score": round(score, 1),
    }


def grade_cwo(sample: Sample) -> dict:
    """[신 스키마] cold/warm/override 대조.

    핵심 자폐 신호 = leak: 반말 잡담(warm)으로 닻 내렸을 때 cold 대비 구어 프레임이
    얼마나 새나. override = 명시적 격식 지시로 register를 강제할 수 있나.
    score = 0.4·override격식 + 0.4·leak저항 + 0.2·cold격식.
    """
    d_cold = round(conv_density(sample.get("cold_doc", "")), 3)
    d_warm = round(conv_density(sample.get("warm_doc", "")), 3)
    d_over = round(conv_density(sample.get("override_doc", "")), 3)
    override_formality = _fmt(d_over)
    leak = max(0.0, d_warm - d_cold)                 # 구어 누출(높을수록 나쁨)
    leak_resistance = max(0.0, 1.0 - min(1.0, leak / 0.5))
    cold_formality = _fmt(d_cold)
    score = (0.4 * override_formality + 0.4 * leak_resistance + 0.2 * cold_formality) * 100
    return {
        "probe_id": sample.get("probe_id"), "schema": "cwo",
        "conv_density": {"cold": d_cold, "warm": d_warm, "override": d_over},
        "leak": round(leak, 3),
        "override_formality": round(override_formality, 3),
        "leak_resistance": round(leak_resistance, 3),
        "cold_formality": round(cold_formality, 3),
        "score": round(score, 1),
    }


def grade_one(sample: Sample) -> dict | None:
    """스키마 자동 감지: 신(cold/warm/override) 우선, 없으면 구(text_levels)."""
    if sample.get("cold_doc") is not None or sample.get("warm_doc") is not None:
        return grade_cwo(sample)
    if sample.get("text_levels"):
        return grade_levels(sample)
    return None


def score(samples: list[Sample]) -> AxisResult:
    rows = [r for r in (grade_one(s) for s in samples) if r]
    if not rows:
        return AxisResult(axis="audience", score=0.0, n=0, note="no samples")
    avg_score = round(mean(r["score"] for r in rows), 2)
    if rows[0]["schema"] == "cwo":
        subs = {
            "conv_density_cold/warm/override": [
                round(mean(r["conv_density"]["cold"] for r in rows), 3),
                round(mean(r["conv_density"]["warm"] for r in rows), 3),
                round(mean(r["conv_density"]["override"] for r in rows), 3)],
            "leak(warm−cold)": round(mean(r["leak"] for r in rows), 3),
            "override_formality%": round(mean(r["override_formality"] for r in rows) * 100, 1),
            "leak_resistance%": round(mean(r["leak_resistance"] for r in rows) * 100, 1),
        }
        note = "score = 0.4·override격식 + 0.4·leak저항 + 0.2·cold격식. leak↑(warm 누출)이면 적응 나쁨."
    else:
        L = len(rows[0]["conv_density"])
        subs = {
            "conv_density_L0..L3": [round(mean(r["conv_density"][i] for r in rows), 3) for i in range(L)],
            "l3_formality%": round(mean(r["l3_formality"] for r in rows) * 100, 1),
            "responsiveness%": round(mean(r["responsiveness"] for r in rows) * 100, 1),
        }
        note = "score = 0.7·L3격식성 + 0.3·반응성. L3 구어밀도↓이면 적응 좋음."
    return AxisResult(axis="audience", score=avg_score, n=len(rows),
                      subscores=subs, detail=rows, note=note)
