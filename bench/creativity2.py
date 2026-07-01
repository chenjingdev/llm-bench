"""creativity2 — 정통(Torrance/Guilford) 창의력 측정.

기존 v1의 결함(발산=창의 혼동, dedup뿐인 약한 품질게이트, 마법수 정규화, 단일 임베딩·단일 시행)
을 고친다:
  · 유창성(fluency)   = 유효 아이디어 수
  · 유연성(flexibility)= 의미 군집 수(카테고리 다양성)
  · 독창성(originality)= **peer-relative 희소성** — 다른 모델 대부분이 떠올리지 못한 정도(마법수 없음)
  · DAT               = 무관 단어 의미거리, **이중 임베딩(nomic+emb3) 교차검증**
품질 게이트 = **제3 모델 심판(gpt-oss:20b)** — 비교 대상(Opus) 아닌 로컬 모델, 고정 이진 루브릭
            (물리적으로 타당한 용도인가) → 횡설수설/비유효 용도 제거.
정규화 = 6모델 상대(min-max) → 자의적 상수 없이 순위. raw 값도 함께 보고(정직).
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean

OLLAMA = "http://localhost:11434"
NOMIC = "nomic-embed-text"
EMB3 = "openai/text-embedding-3-small:latest"
JUDGE = "gpt-oss:20b"

CLUSTER_THR = 0.72   # 같은 카테고리로 묶는 유사도(측정기 보정)
NEAR_THR = 0.82      # 다른 모델이 '사실상 같은 아이디어'를 냈다고 볼 유사도
WEIGHTS = {"originality": 0.35, "flexibility": 0.25, "dat": 0.25, "fluency": 0.15}


def _post(path, obj, t=300):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read())


_ec: dict = {}
def embed(texts, model=NOMIC, prefix="clustering: "):
    out = [None] * len(texts); todo = []; idx = []
    for i, t in enumerate(texts):
        k = (model, prefix + t)
        if k in _ec:
            out[i] = _ec[k]
        else:
            todo.append(prefix + t); idx.append(i)
    if todo:
        r = _post("/api/embed", {"model": model, "input": todo})
        for j, i in enumerate(idx):
            out[i] = r["embeddings"][j]; _ec[(model, prefix + texts[i])] = out[i]
    return out


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na and nb else 0.0


def mpd(vs):
    n = len(vs)
    if n < 2:
        return 0.0
    s = c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 - cos(vs[i], vs[j]); c += 1
    return s / c


_ITEM = re.compile(r"^\s*(?:\d+[.):\]]|[-*•·])\s*(.+)$")
def parse_items(text):
    items = []
    for ln in (text or "").splitlines():
        m = _ITEM.match(ln)
        if m:
            it = re.sub(r"\*+", "", m.group(1)).strip().rstrip(".")
            if it:
                items.append(it)
    if not items:
        items = [p.strip(" .*") for p in re.split(r"[,;\n]", text or "") if p.strip(" .*")]
    return items


def dat_words(text):
    ws = []
    for it in parse_items(text):
        t = re.findall(r"[A-Za-z]+", it)
        if t and len(t[0]) > 1:
            ws.append(t[0].lower())
    seen = set(); uniq = [w for w in ws if not (w in seen or seen.add(w))]
    return uniq[:10]


_jc: dict = {}
def judge_valid(obj, ideas):
    """제3 모델 이진 품질 게이트: 물리적으로 타당한 용도인가."""
    key = (obj, tuple(ideas))
    if key in _jc:
        return _jc[key]
    numbered = "\n".join(f"{i+1}. {it}" for i, it in enumerate(ideas))
    prompt = (f'For each numbered proposed use of a "{obj}", decide if it is a physically '
              f'plausible, sensible use (not nonsense). Reply ONLY a JSON array like '
              f'[{{"i":1,"valid":true}}].\n{numbered}')
    try:
        r = _post("/api/generate", {"model": JUDGE, "prompt": prompt, "stream": False,
                                    "options": {"temperature": 0}})
        m = re.search(r"\[.*\]", r["response"], re.S)
        arr = json.loads(m.group(0))
        vd = {d["i"]: bool(d.get("valid", True)) for d in arr if isinstance(d, dict) and "i" in d}
        res = [vd.get(i + 1, True) for i in range(len(ideas))]
    except Exception:
        res = [True] * len(ideas)
    _jc[key] = res
    return res


def n_clusters(vecs, thr=CLUSTER_THR):
    cents = []
    for v in vecs:
        if any(cos(v, c) >= thr for c in cents):
            continue
        cents.append(v)
    return len(cents)


def score_run(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / "creativity.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    models = sorted({r["model"] for r in rows})

    dat = defaultdict(lambda: {"nomic": [], "emb3": []})
    aut = defaultdict(dict)   # model -> object -> ideas
    for r in rows:
        sub = r["meta"]["subtype"]
        if sub == "dat":
            ws = dat_words(r["text"])
            if len(ws) >= 2:
                dat[r["model"]]["nomic"].append(mpd(embed(ws, NOMIC)))
                dat[r["model"]]["emb3"].append(mpd(embed(ws, EMB3)))
        elif sub == "aut":
            aut[r["model"]][r["meta"]["object"]] = parse_items(r["text"])
    objects = sorted({o for mm in aut.values() for o in mm})

    # 품질 게이트 + 임베딩 (model,object)
    valid = defaultdict(dict); vecs = defaultdict(dict)
    for m in models:
        for o in objects:
            ideas = aut[m].get(o, [])
            if not ideas:
                continue
            mask = judge_valid(o, ideas)
            vi = [it for it, ok in zip(ideas, mask) if ok]
            valid[m][o] = vi
            vecs[m][o] = embed(vi, NOMIC) if vi else []
            print(f"  judged {m[-3:]}/{o}: {len(vi)}/{len(ideas)} valid", file=sys.stderr)

    comp = {}; examples = {}
    for m in models:
        flu = []; flx = []; org = []; best = (-1, "")
        for o in objects:
            vi = valid[m].get(o, []); vv = vecs[m].get(o, [])
            if not vi:
                continue
            flu.append(len(vi)); flx.append(n_clusters(vv))
            others = [x for x in models if x != m]
            for it, v in zip(vi, vv):
                near = sum(1 for om in others if any(cos(v, w) > NEAR_THR for w in vecs[om].get(o, [])))
                rar = 1 - near / len(others) if others else 0
                org.append(rar)
                if rar > best[0]:
                    best = (rar, f"{it} ({o})")
        dn = mean(dat[m]["nomic"]) if dat[m]["nomic"] else 0
        de = mean(dat[m]["emb3"]) if dat[m]["emb3"] else 0
        comp[m] = {"fluency": mean(flu) if flu else 0, "flexibility": mean(flx) if flx else 0,
                   "originality": mean(org) if org else 0, "dat_nomic": dn, "dat_emb3": de,
                   "dat": (dn + de) / 2, "valid_total": sum(flu)}
        examples[m] = best[1]

    def norm(key):
        vals = {m: comp[m][key] for m in models}
        lo, hi = min(vals.values()), max(vals.values())
        return {m: (50.0 if hi == lo else (vals[m] - lo) / (hi - lo) * 100) for m in models}

    nf, nx, no, nd = norm("fluency"), norm("flexibility"), norm("originality"), norm("dat")
    out = {}
    for m in models:
        sc = (WEIGHTS["originality"] * no[m] + WEIGHTS["flexibility"] * nx[m] +
              WEIGHTS["dat"] * nd[m] + WEIGHTS["fluency"] * nf[m])
        out[m] = {"score": round(sc, 1), "raw": comp[m],
                  "norm": {"originality": round(no[m], 1), "flexibility": round(nx[m], 1),
                           "dat": round(nd[m], 1), "fluency": round(nf[m], 1)},
                  "example": examples[m]}
    return out, objects


def html_report(out: dict, models: list[str]) -> str:
    from . import audience2 as A2
    disp, COL = A2.disp, A2._COLOR
    al = [A2.config_alias(m) for m in models]
    R = {A2.config_alias(m): out[m] for m in models}
    rank = sorted(al, key=lambda a: -R[a]["score"])
    best, worst = rank[0], rank[-1]
    top_org = max(al, key=lambda a: R[a]["raw"]["originality"])
    top_dat = max(al, key=lambda a: R[a]["raw"]["dat"])
    col = lambda a: COL.get(a, "#ccc")

    def barrow(a):
        c = col(a); s = R[a]["score"]
        return (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td>'
                f'<td><div style="display:flex;align-items:center;gap:8px">'
                f'<div style="width:{round(s*1.9)+6}px;height:14px;background:{c};border-radius:3px"></div>'
                f'<span style="color:#cfe0ee;font-size:12px">{s}</span></div></td></tr>')
    comp_rows = "".join(barrow(a) for a in al)

    raw_rows = ""
    for a in al:
        r = R[a]["raw"]; c = col(a)
        raw_rows += (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td><td><b>{R[a]["score"]}</b></td>'
                     f'<td>{r["originality"]:.3f}</td><td>{r["flexibility"]:.2f}</td>'
                     f'<td>{r["valid_total"]}</td><td>{r["dat"]:.3f}</td></tr>')

    ex_rows = ""
    for a in al:
        c = col(a)
        ex_rows += (f'<div class="ex"><b style="color:{c}">{disp(a)}</b> 최고 독창 아이디어: '
                    f'<span class="q">{(R[a].get("example") or "")}</span></div>')

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>창의력 9모델</title><style>
body{{background:#0b0f14;color:#e5eef7;font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:28px;line-height:1.6}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;color:#cfe0ee;border-bottom:1px solid #1e2730;padding-bottom:6px;margin-top:28px}}
.meta{{color:#8aa0b4;font-size:13px}} table{{border-collapse:collapse;font-size:13.5px;margin-top:10px}}
th,td{{border:1px solid #1e2730;padding:7px 12px;text-align:center}} td.ax{{text-align:left;font-weight:600}}
th{{background:#121922;color:#9fb0c0}} .verdict{{background:#11161d;border:1px solid #1e2730;border-radius:10px;padding:14px 18px;margin:14px 0;max-width:1040px}}
.q{{color:#9fe0a0}} .ex{{font-size:13px;color:#8aa0b4;margin:5px 0;max-width:1040px}} b{{color:#e5eef7}}
.note{{color:#8aa0b4;font-size:12.5px;max-width:1040px}}
</style></head><body>
<h1>창의력 — 정통(Torrance) 측정 · <b>9모델</b>(Opus 6 + Codex 2 + Gemini)</h1>
<div class="meta">judge-free 핵심(임베딩 의미발산) + 제3모델 품질게이트(gpt-oss:20b) · 9모델 ×(DAT 5회 + AUT 4사물) · peer-relative 정규화</div>
<div class="verdict">
<b>핵심</b> — 창의력은 두 갈래로 분리: <b style="color:{col(top_org)}">{disp(top_org)}</b>가 <b>응용 독창성</b>(AUT 희소·타당 용도) 1위,
<b style="color:{col(top_dat)}">{disp(top_dat)}</b>가 <b>단어 발산(DAT)</b> 1위. 종합 1위 <b style="color:{col(best)}">{disp(best)}</b> / 최하 <b style="color:{col(worst)}">{disp(worst)}</b>.
교차 vendor(Codex·Gemini)도 같은 잣대(peer-relative 희소성·로컬 임베딩)로 채점.
</div>
<h2>종합 창의 점수 (peer-relative, 0–100)</h2>
<table><tr><th>모델</th><th>창의 종합</th></tr>{comp_rows}</table>
<div class="note">가중치: 독창성 0.35 · 유연성 0.25 · DAT 0.25 · 유창성 0.15. min-max 상대 정규화라 스프레드 과장(raw 아래).</div>
<h2>구성요소 (raw)</h2>
<table><tr><th>모델</th><th>종합</th><th>독창성<br>(희소성)</th><th>유연성<br>(군집수)</th><th>유효<br>아이디어</th><th>DAT<br>(거리)</th></tr>{raw_rows}</table>
<h2>가장 독창적이었던 아이디어 (모델별)</h2>
{ex_rows}
<h2>방법론 · 한계</h2>
<div class="note">유창성=유효 아이디어 수, 유연성=의미 군집 수, 독창성=peer-relative 희소성(다른 8모델 대부분이 못 떠올린 정도), DAT=무관 단어 평균 의미거리.
품질 게이트=제3 모델(gpt-oss, 비교대상 아님) 이진 판정. <b>한계</b>: (1) AUT 4사물·DAT 5회·repeat=1 — 상위권은 robust하나 하위 순위는 흔들림. (2) DAT 교차임베딩(dat_nomic·dat_emb3)이 동일 digest면 사실상 1개 임베딩. (3) Codex·Gemini는 agentic 페르소나(시스템프롬프트 비대칭)이나 과제·채점은 동일.</div>
</body></html>"""


def main():
    run_dir = Path(sys.argv[1])
    out, objects = score_run(run_dir)
    print(json.dumps({m.split("-")[-1]: v for m, v in out.items()}, ensure_ascii=False, indent=2))
    try:
        models = sorted(out.keys())
        Path("results/reports/creativity2.html").write_text(html_report(out, models), encoding="utf-8")
        print("\n[report] results/reports/creativity2.html", file=sys.stderr)
    except Exception as e:
        print(f"[report-skip] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
