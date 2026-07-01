"""audience3 — 청중 적응, **문서 본문** 기준 재측정.

audience2의 결함(사용자 지적): 응답 전체의 구어 마커를 세어 ① 친근한 채팅 래퍼
("오 ㅋㅋ 여기!")와 ② 문서 *뒤* 마무리 멘트, ③ 표의 기능 이모지(🟢🟡 상태범례)까지
'문서 말투'로 오판했다. 실제 문서 본문은 격식체였는데도 페널티.

수정:
  · 자기지식만으로 완성 가능한 주제(probes) → 모델이 '되묻기' 대신 실제 문서를 쓰게.
  · **문서 본문만** 채점 — 제3모델(gpt-oss)이 "앞뒤 잡담 무시, 문서 본문 격식 0~100,
    본문 없으면 -1" 으로 평가(주 지표, 절대값이라 peer 정규화 불필요).
  · 교차검증(부): 정규식으로 본문 추출(첫 헤더~, 끝 채팅줄 제거) + **이모지 제외** 후 구어밀도.
  · '문서를 썼나'(doc_produced)는 register와 분리해 별도 보고.

핵심 질문: 반말 잡담 직후에도 **문서 본문**은 격식을 유지하나(leak_doc = 격식cold − 격식warm)?
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

OLLAMA = "http://localhost:11434"
JUDGE = "gpt-oss:20b"


# --- 문서 본문 추출(정규식, 교차검증용) --------------------------------
def extract_body(text: str) -> str:
    """첫 구조 요소(헤더/볼드제목/번호목록)부터 본문으로 보고, 끝의 채팅 마무리줄 제거."""
    lines = (text or "").split("\n")
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*#{1,6}\s", ln):
            start = i
            break
    if start is None:
        for i, ln in enumerate(lines):
            s = ln.strip()
            if re.match(r"^\*\*.+\*\*$", s) or re.match(r"^\d+[.)]\s", s):
                start = i
                break
    if start is None:
        return ""                    # 문서 구조 없음(잡담/되묻기만)
    body = lines[start:]
    # 뒤에서부터: 구조요소가 아니면서 대화체 신호가 있는 평문 = 마무리 래퍼 → 절단
    while body:
        last = body[-1].strip()
        if not last:
            body.pop(); continue
        is_struct = bool(re.match(r"^(#|\||>|-|\*|\d+[.)])", last))
        if (not is_struct and (A2._INFORMAL_RE.search(last) or last.endswith("!")
                               or last.endswith("~") or "?" in last)):
            body.pop(); continue
        break
    return "\n".join(body)


def body_informal(body: str) -> float:
    """본문 평문(표 제외) + 이모지 제외 후 구어밀도 — 기능 이모지/표를 register에서 배제."""
    prose = "\n".join(ln for ln in (body or "").split("\n") if not ln.strip().startswith("|"))
    prose = A2._EMOJI_RE.sub("", prose)
    return A2.informal_rate(prose)


# --- 제3모델: 문서 본문만 격식 평가(주 지표) ----------------------------
_jc: dict = {}
_JC_PATH = Path("results/reports/.audience3_judge_cache.json")


def _load():
    if _JC_PATH.exists():
        try:
            _jc.update(json.loads(_JC_PATH.read_text()))
        except Exception:
            pass


def _save():
    try:
        _JC_PATH.write_text(json.dumps(_jc, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def judge_doc(text: str):
    """문서 본문만의 격식 0~100. 인사/잡담은 무시. 본문 없으면 -1. 실패 시 None."""
    snippet = (text or "")[:1700]
    if A2.kor_chars(snippet) < 5:
        return -1
    if snippet in _jc:
        return _jc[snippet]
    prompt = (
        "다음은 사용자 요청에 대한 AI의 응답이다. 앞뒤에 친근한 인사·잡담(예: '오 ㅋㅋ 여기!', "
        "'이대로 붙여넣으면 됨 ㅎㅎ')이 섞여 있을 수 있다. 그 잡담은 **무시**하고, 응답 안에 들어있는 "
        "**문서 본문 자체의 격식 수준**만 평가하라.\n"
        "0 = 문서 본문이 반말·구어체·농담조, 50 = 가벼운 사내 메모체, "
        "100 = 공식 보고서/규정 문서체(다/음 종결, 객관 문어체).\n"
        "표의 상태 이모지(🟢🟡)는 기능 요소이니 격식 판단에서 제외하라.\n"
        "문서라고 할 본문이 아예 없고 인사/되묻기/질문만 있으면 -1.\n"
        'JSON만 출력: {"doc":<0~100 정수 또는 -1>}\n\n---\n' + snippet)
    try:
        req = urllib.request.Request(
            OLLAMA + "/api/generate",
            data=json.dumps({"model": JUDGE, "prompt": prompt, "stream": False,
                             "options": {"temperature": 0}}).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        m = re.search(r'"doc"\s*:\s*(-?\d+)', r["response"])
        v = int(m.group(1)) if m else None
        val = None if v is None else (-1 if v < 0 else max(0, min(100, v)))
    except Exception:
        val = None
    _jc[snippet] = val
    return val


# ----------------------------------------------------------------------
def score_run(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / "audience.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    models = sorted({r["model"] for r in rows})
    _load()

    per = defaultdict(list)
    for r in rows:
        cold, warm, ovr = r.get("cold_doc", ""), r.get("warm_doc", ""), r.get("override_doc", "")
        rec = {"topic": r.get("topic")}
        for cond, t in [("cold", cold), ("warm", warm), ("ovr", ovr)]:
            jd = judge_doc(t)
            body = extract_body(t)
            rec[f"{cond}_judge"] = jd                       # 문서 격식(주), -1=문서없음
            rec[f"{cond}_has_doc"] = (body != "") and (jd is None or jd >= 0)
            rec[f"{cond}_body_inf"] = body_informal(body) if body else None
            rec[f"{cond}_txt"] = t
        per[r["model"]].append(rec)
        print(f"  {A2.config_alias(r['model'])}/{rec['topic'][:12]}: "
              f"cold={rec['cold_judge']} warm={rec['warm_judge']} ovr={rec['ovr_judge']}",
              file=sys.stderr)
    _save()

    out = {}
    for m in models:
        rs = per[m]
        def jvals(cond):  # 문서가 있는(>=0) 표본만
            return [x[f"{cond}_judge"] for x in rs
                    if isinstance(x[f"{cond}_judge"], int) and x[f"{cond}_judge"] >= 0]
        jc, jw, jo = jvals("cold"), jvals("warm"), jvals("ovr")
        bc = [x["cold_body_inf"] for x in rs if x["cold_body_inf"] is not None]
        bw = [x["warm_body_inf"] for x in rs if x["warm_body_inf"] is not None]
        cold_f = mean(jc) if jc else None
        warm_f = mean(jw) if jw else None
        ovr_f = mean(jo) if jo else None
        # 누출: 같은 topic에서 cold·warm 둘 다 문서 생산된 경우만 짝지어
        pairs = [(x["cold_judge"], x["warm_judge"]) for x in rs
                 if isinstance(x["cold_judge"], int) and x["cold_judge"] >= 0
                 and isinstance(x["warm_judge"], int) and x["warm_judge"] >= 0]
        leak = mean(c - w for c, w in pairs) if pairs else None
        # warm에서 가장 격식 낮은(=가장 샌) 문서 예시
        wdocs = [x for x in rs if isinstance(x["warm_judge"], int) and x["warm_judge"] >= 0]
        worst = min(wdocs, key=lambda x: x["warm_judge"]) if wdocs else None
        out[m] = {
            "cold_f": cold_f, "warm_f": warm_f, "ovr_f": ovr_f, "leak": leak,
            "body_inf_cold": mean(bc) if bc else None, "body_inf_warm": mean(bw) if bw else None,
            "doc_cold": sum(1 for x in rs if x["cold_has_doc"]), "n": len(rs),
            "doc_warm": sum(1 for x in rs if x["warm_has_doc"]),
            "n_pairs": len(pairs),
            "worst_warm": ({"topic": worst["topic"], "f": worst["warm_judge"],
                            "txt": worst["warm_txt"]} if worst else None),
        }
    return out, models


def _fmt(v, p=1):
    return "—" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def html_report(out: dict, models: list[str]) -> str:
    al = [A2.config_alias(m) for m in models]
    R = {A2.config_alias(m): out[m] for m in models}
    # 랭킹: warm 문서 격식(주 질문 = 잡담 직후에도 문서가 격식 유지되나) 내림차순
    rank = sorted([a for a in al if R[a]["warm_f"] is not None],
                  key=lambda a: -R[a]["warm_f"])
    col = lambda a: A2._COLOR.get(a, "#ccc")
    disp = A2.disp

    rows = ""
    for a in al:
        r = R[a]; c = col(a)
        lk = r["leak"]
        lks = "—" if lk is None else f"{lk:+.0f}"
        lkc = "#9fb0c0" if lk is None else ("#ef4444" if lk > 8 else ("#10b981" if lk < 3 else "#e5c07b"))
        rows += (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td>'
                 f'<td>{r["doc_cold"]}/{r["n"]} · {r["doc_warm"]}/{r["n"]}</td>'
                 f'<td><b>{_fmt(r["cold_f"],0)}</b></td><td><b>{_fmt(r["warm_f"],0)}</b></td>'
                 f'<td style="color:{lkc}"><b>{lks}</b></td><td>{_fmt(r["ovr_f"],0)}</td>'
                 f'<td>{_fmt(r["body_inf_cold"],2)} → {_fmt(r["body_inf_warm"],2)}</td></tr>')

    # warm에서 가장 격식 낮았던 문서(있으면) — 진짜 누출 사례 후보
    worst_a = min([a for a in al if R[a]["warm_f"] is not None],
                  key=lambda a: R[a]["warm_f"]) if rank else None
    smoke = ""
    if worst_a and R[worst_a]["worst_warm"]:
        ww = R[worst_a]["worst_warm"]
        smoke = (f'<div class="ex"><b style="color:{col(worst_a)}">{disp(worst_a)}</b> '
                 f'— warm 문서 격식 최저(judge {ww["f"]}/100), 주제 「{ww["topic"]}」'
                 f'<div class="q">{A2._hl(ww["txt"], 700)}</div></div>')

    best_a = rank[0] if rank else None
    head = ""
    if best_a:
        b = R[best_a]
        head = (f'잡담 직후에도 <b>문서 본문</b> 격식을 가장 잘 지킨 모델: '
                f'<b style="color:{col(best_a)}">{disp(best_a)}</b> '
                f'(warm 문서 격식 {b["warm_f"]:.0f}/100, 누출 {_fmt(b["leak"],0)}).')

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>청중 적응 v3 — 문서 본문 기준</title><style>
body{{background:#0b0f14;color:#e5eef7;font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:28px;line-height:1.6}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;color:#cfe0ee;border-bottom:1px solid #1e2730;padding-bottom:6px;margin-top:28px}}
.meta{{color:#8aa0b4;font-size:13px}} table{{border-collapse:collapse;font-size:13.5px;margin-top:10px}}
th,td{{border:1px solid #1e2730;padding:7px 11px;text-align:center}} td.ax{{text-align:left;font-weight:600}}
th{{background:#121922;color:#9fb0c0}} .verdict{{background:#11161d;border:1px solid #1e2730;border-radius:10px;padding:14px 18px;margin:14px 0;max-width:1080px}}
.q{{color:#cfe0ee;background:#0e141b;border-left:3px solid #2a3642;padding:9px 12px;margin:7px 0;font-size:12.5px;border-radius:4px;max-width:1040px;white-space:pre-wrap}}
.q mark{{background:#7f1d1d;color:#fecaca;padding:0 1px;border-radius:2px}}
.ex{{font-size:13px;color:#8aa0b4;margin:12px 0;max-width:1080px}} b{{color:#e5eef7}}
.note{{color:#8aa0b4;font-size:12.5px;max-width:1080px}}
</style></head><body>
<h1>청중 적응 v3 — <b>문서 본문</b> 기준 (래퍼·기능 이모지 제외)</h1>
<div class="meta">제3모델(gpt-oss:20b)이 "앞뒤 잡담 무시, <b>문서 본문</b> 격식 0~100" 평가(주) +
정규식 본문추출·이모지제외 구어밀도(부) · 9모델 × 자기완결 주제 × (cold·warm·override)</div>

<div class="verdict">
<b>왜 v2가 틀렸나</b> — v2는 응답 전체의 구어 마커를 세어 ① 친근한 채팅 래퍼("오 ㅋㅋ 여기!")
② 문서 뒤 마무리 멘트 ③ 표의 상태 이모지(🟢🟡)까지 '문서 말투'로 오판했다. 실제 문서 본문은 격식체였다.<br>
<b>v3</b>는 <u>문서 본문만</u> 본다(잡담 래퍼·기능 이모지 제외). {head}
</div>

<h2>문서 본문 격식 — cold vs warm (핵심)</h2>
<table><tr><th>모델</th><th>문서 작성<br>cold·warm</th><th>문서 격식<br>cold</th><th>문서 격식<br>warm</th>
<th>누출<br>(cold−warm)</th><th>override<br>격식</th><th>본문 구어밀도(정규식)<br>cold→warm</th></tr>{rows}</table>
<div class="note"><b>문서 격식</b>=제3모델이 본문만 0~100(높을수록 격식). <b>누출</b>=cold−warm(+면 잡담 직후 문서가 덜 격식적 = 진짜 누출). <b>문서 작성</b>=되묻기/잡담만 아니라 실제 문서 본문을 낸 횟수. 본문 구어밀도=정규식 교차검증(이모지·표 제외).</div>

<h2>warm 문서 격식 최저 사례</h2>
{smoke or '<div class="note">warm에서 격식이 무너진 문서 본문 사례 없음(모두 문서는 격식 유지).</div>'}

<h2>방법론 · 정직한 한계</h2>
<div class="note">
<b>측정</b>: 자기지식만으로 완성 가능한 주제로 cold/warm/override 호출. <b>문서 본문 격식</b>을 제3모델이
잡담 무시하고 0~100으로 평가(절대값이라 peer 정규화 불필요). 정규식 본문추출(첫 헤더~, 끝 채팅줄 절단)+이모지
제외 구어밀도로 교차검증. '문서를 썼나'는 register와 분리.<br><br>
<b>한계</b>: (1) 문서 격식 주 지표가 <b>gpt-oss 1개 심판</b> — 다수결/루브릭 강화 여지. (2) 본문추출 휴리스틱은
헤더 없는 문서에서 약함. (3) 교차 vendor(Codex·Gemini)는 문서를 파일로 빼면 본문이 채팅에 없어 '문서 미작성'으로
잡힘 — 이들의 'doc 작성률'은 인라인 출력 성향이지 능력이 아님. (4) 주제·repeat 소표본.
</div>
</body></html>"""


def main():
    run_dir = Path(sys.argv[1])
    out, models = score_run(run_dir)
    summ = {A2.config_alias(m): {k: v for k, v in out[m].items() if k != "worst_warm"}
            for m in models}
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    Path("results/reports/audience3.html").write_text(html_report(out, models), encoding="utf-8")
    print("\n[report] results/reports/audience3.html", file=sys.stderr)


if __name__ == "__main__":
    main()
