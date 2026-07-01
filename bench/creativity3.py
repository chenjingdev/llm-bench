"""creativity3 — '아이디어 발산' 측정 (사용자 정의 창의력).

creativity2(AUT/DAT + 타당성 게이트)의 문제: 타당성 필터가 '허황되지만 참신한'
아이디어를 깎아 — 사용자가 발산 단계에서 가치 있다고 보는 대담함을 페널티로 만듦.

creativity3:
  · 실제 기술 도메인 브레인스토밍(새 RAG 방식 등) — toy(벽돌 용도) 아님.
  · **참신성(judge-free)** 3각도:
      novelty_obvious = 아이디어가 '교과서적 표준 답'(gpt-oss가 생성한 obvious 앵커)에서 떨어진 거리
      peer_rarity     = 다른 모델 대부분이 안 떠올린 정도(near-match 없음 비율)
      self_divergence = 자기 아이디어들끼리 의미 발산 폭(mpd)
  · **타당성(plausibility)은 거르지 않고 *별도 축*** — gpt-oss가 각 아이디어 실현가능성 0~100.
    → '참신 vs 허황' 트레이드오프를 2D로 표시(발산 단계에선 참신축이 우선).
정규화 = 9모델 peer min-max. 임베딩 = 로컬 nomic(creativity2 재사용).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean

from . import audience2 as A2
from . import creativity2 as C2

OLLAMA = "http://localhost:11434"
JUDGE = "gpt-oss:20b"
NOMIC = "nomic-embed-text"
NEAR_THR = 0.82      # 다른 모델이 '사실상 같은 아이디어'를 냈다고 볼 유사도
W = {"novelty": 0.4, "rarity": 0.35, "divergence": 0.25}

# 도메인 → obvious 앵커 생성용 설명
_DOMAIN_DESC = {
    "rag": "improving retrieval-augmented generation (RAG) systems",
    "uncertainty": "making an LLM recognize and express when it does not know something",
    "agent_memory": "giving an AI agent long-term memory across sessions",
    "llm_eval": "evaluating or benchmarking which LLM is better for someone",
}


def _post(path, obj, t=180):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read())


# --- obvious 앵커(교과서적 표준 답) — gpt-oss 생성, 디스크 캐시 -----------
_ANCHOR_PATH = Path("results/reports/.divergence_anchors.json")
_anchors: dict = {}


def obvious_anchor(domain: str) -> list[str]:
    if not _anchors and _ANCHOR_PATH.exists():
        try:
            _anchors.update(json.loads(_ANCHOR_PATH.read_text()))
        except Exception:
            pass
    if domain in _anchors:
        return _anchors[domain]
    desc = _DOMAIN_DESC.get(domain, domain)
    prompt = (f"List the 12 most standard, well-known, commonly-recommended, textbook approaches to "
              f"{desc}. Only the conventional, obvious ones that everyone already knows. "
              f"Numbered list 1-12, one short approach per line.")
    try:
        r = _post("/api/generate", {"model": JUDGE, "prompt": prompt, "stream": False,
                                    "options": {"temperature": 0}})
        items = C2.parse_items(r["response"])
    except Exception:
        items = []
    _anchors[domain] = items
    try:
        _ANCHOR_PATH.write_text(json.dumps(_anchors, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return items


# --- 타당성(별도 축) — gpt-oss가 각 아이디어 실현가능성 0~100 -------------
_PL_PATH = Path("results/reports/.divergence_plausibility.json")
_pl: dict = {}


def plausibility(domain: str, ideas: list[str]) -> list[int]:
    if not _pl and _PL_PATH.exists():
        try:
            _pl.update(json.loads(_PL_PATH.read_text()))
        except Exception:
            pass
    key = domain + "||" + "|".join(ideas)
    if key in _pl:
        return _pl[key]
    desc = _DOMAIN_DESC.get(domain, domain)
    numbered = "\n".join(f"{i+1}. {it}" for i, it in enumerate(ideas))
    prompt = (f"For {desc}, rate each numbered idea's technical PLAUSIBILITY/feasibility from 0 "
              f"(absurd, could never work) to 100 (sound, clearly workable). Judge feasibility only, "
              f"NOT novelty. Reply ONLY a JSON array like [{{\"i\":1,\"p\":70}}].\n{numbered}")
    try:
        r = _post("/api/generate", {"model": JUDGE, "prompt": prompt, "stream": False,
                                    "options": {"temperature": 0}})
        m = re.search(r"\[.*\]", r["response"], re.S)
        arr = json.loads(m.group(0))
        vd = {d["i"]: int(d.get("p", 50)) for d in arr if isinstance(d, dict) and "i" in d}
        res = [max(0, min(100, vd.get(i + 1, 50))) for i in range(len(ideas))]
    except Exception:
        res = [50] * len(ideas)
    _pl[key] = res
    try:
        _PL_PATH.write_text(json.dumps(_pl, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return res


def score_run(run_dir: Path, use_gpt: bool = False):
    """pure-vector 기본(use_gpt=False): 참신성 기준점을 모델 합의 centroid로 정의.
    use_gpt=True 일 때만 obvious 앵커·타당성에 gpt-oss 사용(비권장, 오염)."""
    rows = [json.loads(l) for l in (run_dir / "divergence.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    models = sorted({r["model"] for r in rows})
    domains = sorted({r["meta"]["domain"] for r in rows})

    # (model, domain) → ideas, vecs
    ideas = defaultdict(dict); vecs = defaultdict(dict)
    for r in rows:
        d = r["meta"]["domain"]
        its = C2.parse_items(r["text"])[:14]
        ideas[r["model"]][d] = its
        vecs[r["model"]][d] = C2.embed(its, NOMIC) if its else []

    # '뻔한 영역' = 도메인별 전체 아이디어의 합의 중심(centroid). gpt 무관, 순수 vector.
    centroid = {}
    for d in domains:
        allv = [v for m in models for v in vecs[m].get(d, [])]
        if allv:
            dim = len(allv[0])
            centroid[d] = [sum(x[i] for x in allv) / len(allv) for i in range(dim)]

    anchor_vecs = {}
    if use_gpt:
        for d in domains:
            anchor_vecs[d] = C2.embed(obvious_anchor(d), NOMIC) or []

    comp = {}; examples = {}
    for m in models:
        nov = []; rar = []; div = []; pls = []; best = (-1, "", "")
        for d in domains:
            vi = ideas[m].get(d, []); vv = vecs[m].get(d, [])
            if not vv:
                continue
            # 자기발산 = 군집수(12개가 몇 개의 distinct 테마로 갈리나). 평균거리는 고차원에서
            # 다 ~0.25로 평탄화돼 변별 못함(변별 12%) → 군집수가 훨씬 잘 가름(변별 79%). 순수 vector.
            div.append(C2.n_clusters(vv))
            c = centroid.get(d); av = anchor_vecs.get(d, [])
            others = [x for x in models if x != m]
            pl = plausibility(d, vi) if use_gpt else [None] * len(vi)
            if use_gpt:
                pls.extend([p for p in pl if p is not None])
            for it, v, p in zip(vi, vv, pl):
                if use_gpt and av:                            # (비권장) gpt 앵커 거리
                    nd = min(1 - C2.cos(v, b) for b in av)
                else:                                         # 기본: 합의 중심에서의 거리
                    nd = (1 - C2.cos(v, c)) if c else 0.5
                nov.append(nd)
                # peer 희소성: 다른 모델 중 near-match 없으면 희소(순수 vector)
                near = any(C2.cos(v, w) > NEAR_THR for om in others for w in vecs[om].get(d, []))
                rar.append(0.0 if near else 1.0)
                if nd > best[0]:
                    best = (nd, it, d)
        comp[m] = {"novelty": mean(nov) if nov else 0, "rarity": mean(rar) if rar else 0,
                   "divergence": mean(div) if div else 0,
                   "plausibility": (mean(pls) if pls else None), "n_ideas": len(nov)}
        examples[m] = best

    def norm(key):
        vals = {m: comp[m][key] for m in models}
        lo, hi = min(vals.values()), max(vals.values())
        return {m: (50.0 if hi == lo else (vals[m] - lo) / (hi - lo) * 100) for m in models}

    nn, nr, nd = norm("novelty"), norm("rarity"), norm("divergence")
    out = {}
    for m in models:
        diverge = W["novelty"] * nn[m] + W["rarity"] * nr[m] + W["divergence"] * nd[m]
        out[m] = {"score": round(diverge, 1),
                  "norm": {"novelty": round(nn[m], 1), "rarity": round(nr[m], 1),
                           "divergence": round(nd[m], 1)},
                  "raw": {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in comp[m].items()},
                  "example": examples[m]}
    return out, models, domains


# --- 리포트 ------------------------------------------------------------
def _scatter(out, models):
    """참신성(x, 합의중심 거리) × 자기발산(y, 아이디어 간 spread) 2D SVG — 둘 다 순수 vector."""
    W_, H_, pad = 540, 380, 48
    pts = []
    for m in models:
        a = A2.config_alias(m)
        x = out[m]["norm"]["novelty"]                 # 0~100
        y = out[m]["norm"]["divergence"]              # 0~100 (순수 vector)
        px = pad + x / 100 * (W_ - 2 * pad)
        py = H_ - pad - y / 100 * (H_ - 2 * pad)
        c = A2._COLOR.get(a, "#ccc")
        pts.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="6" fill="{c}"/>'
                   f'<text x="{px+9:.0f}" y="{py+4:.0f}" fill="{c}" font-size="11">{A2.disp(a)}</text>')
    grid = (f'<line x1="{pad}" y1="{H_-pad}" x2="{W_-pad}" y2="{H_-pad}" stroke="#2a3642"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H_-pad}" stroke="#2a3642"/>'
            f'<text x="{W_-pad}" y="{H_-pad+30}" fill="#8aa0b4" font-size="11" text-anchor="end">참신성(합의중심 거리) →</text>'
            f'<text x="{pad-34}" y="{pad-16}" fill="#8aa0b4" font-size="11">↑ 자기발산(테마 군집수)</text>')
    return (f'<svg width="{W_}" height="{H_}" style="background:#0e141b;border:1px solid #1e2730;'
            f'border-radius:8px;margin-top:8px">{grid}{"".join(pts)}</svg>')


def html_report(out, models, domains):
    disp, COL = A2.disp, A2._COLOR
    al = [A2.config_alias(m) for m in models]
    R = {A2.config_alias(m): out[m] for m in models}
    rank = sorted(al, key=lambda a: -R[a]["score"])
    col = lambda a: COL.get(a, "#ccc")
    best = rank[0]
    most_novel = max(al, key=lambda a: R[a]["norm"]["novelty"])
    most_rare = max(al, key=lambda a: R[a]["norm"]["rarity"])

    def barrow(a):
        c = col(a); s = R[a]["score"]
        return (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td>'
                f'<td><div style="display:flex;align-items:center;gap:8px"><div style="width:{round(s*1.9)+6}px;'
                f'height:14px;background:{c};border-radius:3px"></div><span style="color:#cfe0ee;font-size:12px">{s}</span></div></td></tr>')
    rows = "".join(barrow(a) for a in al)

    raw = ""
    for a in al:
        n = R[a]["norm"]; c = col(a)
        raw += (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td><td><b>{R[a]["score"]}</b></td>'
                f'<td>{n["novelty"]}</td><td>{n["rarity"]}</td><td>{n["divergence"]}</td></tr>')

    ex = ""
    for a in rank:
        m = next(mm for mm in models if A2.config_alias(mm) == a)
        nd, it, dom = R[a]["example"]
        ex += (f'<div class="ex"><b style="color:{col(a)}">{disp(a)}</b> '
               f'<span style="color:#6b7a8a">[{dom}]</span> <span class="q">{it}</span></div>')

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>아이디어 발산</title><style>
body{{background:#0b0f14;color:#e5eef7;font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:28px;line-height:1.6}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;color:#cfe0ee;border-bottom:1px solid #1e2730;padding-bottom:6px;margin-top:28px}}
.meta{{color:#8aa0b4;font-size:13px}} table{{border-collapse:collapse;font-size:13.5px;margin-top:10px}}
th,td{{border:1px solid #1e2730;padding:7px 12px;text-align:center}} td.ax{{text-align:left;font-weight:600}}
th{{background:#121922;color:#9fb0c0}} .verdict{{background:#11161d;border:1px solid #1e2730;border-radius:10px;padding:14px 18px;margin:14px 0;max-width:1060px}}
.q{{color:#9fe0a0}} .ex{{font-size:13px;color:#8aa0b4;margin:5px 0;max-width:1060px}} b{{color:#e5eef7}}
.note{{color:#8aa0b4;font-size:12.5px;max-width:1060px}}
</style></head><body>
<h1>아이디어 발산 — 실제 도메인 브레인스토밍 · <b>순수 vector(gpt 배제)</b></h1>
<div class="meta">100% 임베딩 기반(gpt 판정 없음): 참신성=합의중심 거리 · 희소성=peer near-match 없음 · 자기발산=아이디어 간 거리 · 9모델 × 도메인 4개(RAG·불확실성·에이전트 메모리·LLM 평가) · 12 아이디어/프롬프트 · nomic 임베딩</div>
<div class="verdict">
<b>발산 1위 <span style="color:{col(best)}">{disp(best)}</span></b> · <b>참신성 최고(합의에서 가장 먼) <span style="color:{col(most_novel)}">{disp(most_novel)}</span></b>
· <b>희소성 최고 <span style="color:{col(most_rare)}">{disp(most_rare)}</span></b>.<br>
<b>gpt 완전 배제</b> — '뻔한 영역'은 gpt가 아니라 <u>모든 모델 아이디어의 합의 중심(centroid)</u>으로 정의(다들 몰리는 곳에서 멀수록 참신).
아래 2D: <b>우측=합의에서 멀다(참신), 상단=자기 아이디어가 더 많은 테마로 갈린다</b>.
</div>
<h2>참신성 × 자기발산 2D (둘 다 순수 vector)</h2>
{_scatter(out, models)}
<h2>발산 종합 점수 (참신 0.4 · 희소 0.35 · 자기발산 0.25)</h2>
<table><tr><th>모델</th><th>발산 종합</th></tr>{rows}</table>
<h2>구성요소 (peer-relative 0–100, 전부 임베딩)</h2>
<table><tr><th>모델</th><th>발산종합</th><th>참신성<br>(합의중심거리)</th><th>희소성<br>(peer)</th><th>자기발산<br>(테마 군집수)</th></tr>{raw}</table>
<h2>각 모델의 가장 참신했던(합의에서 가장 먼) 아이디어</h2>
{ex}
<h2>방법론 · 한계</h2>
<div class="note">
<b>전부 임베딩 거리 계산 — gpt 판정 0.</b> <b>참신성</b>=각 아이디어가 9모델 전체 아이디어의 합의 중심에서 떨어진 코사인 거리(다들 몰리는 '뻔한 영역'에서 멀수록↑). <b>희소성</b>=다른 8모델이 cos&gt;0.82로 비슷한 걸 안 낸 비율. <b>자기발산</b>=한 모델의 12개 아이디어가 몇 개의 distinct 테마로 갈리나(cos&gt;0.72 군집수). 평균거리는 고차원에서 평탄화돼 못 가르므로 군집수 사용.<br><br>
<b>한계</b>: (1) 도메인 4개·repeat=1 — 소표본. (2) nomic 임베딩 1개에 의존 — 거리 척도가 <b>어휘적 생소함</b>을 <b>개념적 참신함</b>으로 오인할 수 있음(표준 용어를 쓰면서 개념이 참신한 아이디어는 저평가 가능). (3) 합의중심은 '다수가 평범'을 전제 — 모두가 같은 참신한 방향으로 가면 그게 평범으로 잡힘. (4) Codex·Gemini는 agentic 페르소나(시스템프롬프트 비대칭). <b>(과거 gpt-oss 의존분(obvious 앵커·타당성)은 제거함.)</b>
</div>
</body></html>"""


def main():
    run_dir = Path(sys.argv[1])
    use_gpt = "--use-gpt" in sys.argv      # 기본 pure-vector. gpt는 명시적으로만(비권장)
    out, models, domains = score_run(run_dir, use_gpt=use_gpt)
    print(json.dumps({A2.config_alias(m): {"score": out[m]["score"], **out[m]["norm"]}
                      for m in models}, ensure_ascii=False, indent=2))
    Path("results/reports/creativity3.html").write_text(html_report(out, models, domains), encoding="utf-8")
    print("\n[report] results/reports/creativity3.html", file=sys.stderr)


if __name__ == "__main__":
    main()
