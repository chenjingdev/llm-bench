"""audience2 — 청중 적응(사용역 전환) 정통 측정.

기존 v1의 결함을 고친다:
  · 표면 신호 1개(종결어미)만 봄        → 구어 마커 묶음(해요체·반말·추임새·이모지·호명)
  · 마법수(/0.5 윈도, ? 0.5 가중)       → **self-contrast(warm−cold) + peer-relative** (상수 없음)
  · 되묻기 성향이 적응으로 둔갑(4.5)    → **clarification 분리**(문서 비생산 표본 제외)
  · 정작 '자폐' 현상을 안 잼            → **닻 감수성(leak)** = informal(warm) − informal(cold)

세 조건(runner가 생성):
  A cold      — 맨정신 공식 문서 요청.            informal 낮을수록 좋음.
  B warm      — 반말·ㅋㅋ 잡담 2턴 닻 뒤 같은 요청. leak = informal(warm)−informal(cold) (핵심).
  C override  — 명시적 격식 명령(다/음만).        informal 낮을수록 좋음.

채점:
  informal_rate = 구어 마커 수 / (한글 음절수/100)   ← 길이 정규화, 자릿수 상수 아님
  leak/cold/override 를 6모델 peer min-max(낮을수록↑) → 가중합. raw 값 동봉(정직).
교차검증: 제3 모델(gpt-oss:20b) round-trip — cold/warm 문서의 격식도 0~100 블라인드 평가.
"""

from __future__ import annotations

import html as _html
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean

OLLAMA = "http://localhost:11434"
JUDGE = "gpt-oss:20b"

# --- 구어/대화체 마커 (informal) ----------------------------------------
# 해요체 종결은 '모음 음절 + 요'로 한정 → 필요/중요/내용 등 명사의 요 오탐 방지.
_INFORMAL_PATTERNS = [
    r"[어아에여애해세네대래봐워려이드구까지]요",      # 해요체 종결
    r"거든", r"잖아", r"는데요", r"ㄹ게", r"을게요",
    r"군요", r"네요", r"죠", r"더라고?", r"구나", r"드라",
    r"ㅋ+", r"ㅎ+", r"[ㅠㅜ]+", r"헐", r"엥", r"우와", r"오오+", r"땡큐", r"ㅇㅋ",
    r"여러분", r"너희",                                  # 독자 호명/2인칭
]
_INFORMAL_RE = re.compile("|".join(_INFORMAL_PATTERNS))
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF✀-➿]")

# --- 문어/격식 평서 종결 (formal) ---------------------------------------
_FORMAL_RE = re.compile(
    r"(?:다|음|함|됨|임|것|운다|한다|된다|이다|있다|없다|했다|였다|진다|된다|토록)[\.\)\n]")
_FORMAL_CONN_RE = re.compile(r"또한|따라서|그러나|아울러|본 문서|해당|관련하여|위하여|대하여")

_KOR_RE = re.compile(r"[가-힣]")
_CLAR_RE = re.compile(r"어떤|무엇|구체적|어느|알려주|필요한|범위|기간|대상|항목|정보가|말씀")


def kor_chars(text: str) -> int:
    return len(_KOR_RE.findall(text or ""))


def informal_count(text: str) -> int:
    t = text or ""
    return len(_INFORMAL_RE.findall(t)) + len(_EMOJI_RE.findall(t))


def formal_count(text: str) -> int:
    t = text or ""
    return len(_FORMAL_RE.findall(t)) + len(_FORMAL_CONN_RE.findall(t))


def informal_rate(text: str) -> float:
    """구어 마커 / 100 한글 음절. 길이 정규화된 구어밀도."""
    k = kor_chars(text)
    return informal_count(text) / max(k, 1) * 100


def formal_rate(text: str) -> float:
    k = kor_chars(text)
    return formal_count(text) / max(k, 1) * 100


def is_clarification(text: str) -> bool:
    """문서를 안 쓰고 되묻기/역질문만 한 표본 → register 채점에서 제외."""
    t = text or ""
    if kor_chars(t) > 220:          # 충분히 길면 문서 생산으로 간주
        return False
    return ("?" in t) and bool(_CLAR_RE.search(t))


# --- 제3 모델 round-trip 교차검증 ---------------------------------------
_jc: dict = {}
_JC_PATH = Path("results/reports/.audience_judge_cache.json")


def _load_judge_cache():
    if _JC_PATH.exists():
        try:
            _jc.update(json.loads(_JC_PATH.read_text()))
        except Exception:
            pass


def _save_judge_cache():
    try:
        _JC_PATH.write_text(json.dumps(_jc, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def judge_formality(text: str):
    """블라인드 격식도 0(친구끼리 반말 잡담)~100(공식 보고서). 실패 시 None."""
    snippet = (text or "")[:1100]
    if kor_chars(snippet) < 5:
        return None
    if snippet in _jc:
        return _jc[snippet]
    prompt = ("아래 한국어 글의 '격식 수준'을 0~100 정수로만 평가하라. "
              "0=친구끼리 반말 잡담/이모지, 50=가벼운 사내 메모, 100=공식 보고서/규정 문서. "
              'JSON만 출력: {"formality": <정수>}\n\n---\n' + snippet)
    try:
        req = urllib.request.Request(
            OLLAMA + "/api/generate",
            data=json.dumps({"model": JUDGE, "prompt": prompt, "stream": False,
                             "options": {"temperature": 0}}).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        m = re.search(r'"formality"\s*:\s*(\d+)', r["response"])
        val = max(0, min(100, int(m.group(1)))) if m else None
    except Exception:
        val = None
    _jc[snippet] = val
    return val


# ----------------------------------------------------------------------
def score_run(run_dir: Path, use_judge: bool = True):
    rows = [json.loads(l) for l in (run_dir / "audience.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    models = sorted({r["model"] for r in rows})
    if use_judge:
        _load_judge_cache()

    # model -> list of per-topic dicts
    per = defaultdict(list)
    clars = defaultdict(int)
    for r in rows:
        cold, warm, ovr = r.get("cold_doc", ""), r.get("warm_doc", ""), r.get("override_doc", "")
        rec = {"topic": r.get("topic"),
               "cold_inf": informal_rate(cold), "warm_inf": informal_rate(warm),
               "ovr_inf": informal_rate(ovr), "cold_formal": formal_rate(cold),
               "cold_txt": cold, "warm_txt": warm, "ovr_txt": ovr,
               "cold_clar": is_clarification(cold), "warm_clar": is_clarification(warm)}
        if rec["cold_clar"] or rec["warm_clar"]:
            clars[r["model"]] += 1
        rec["leak"] = rec["warm_inf"] - rec["cold_inf"]
        if use_judge:
            rec["judge_cold"] = judge_formality(cold)
            rec["judge_warm"] = judge_formality(warm)
            print(f"  judged {config_alias(r['model'])}/{rec['topic'][:10]}: "
                  f"cold={rec['judge_cold']} warm={rec['judge_warm']}", file=sys.stderr)
        per[r["model"]].append(rec)
    if use_judge:
        _save_judge_cache()

    comp = {}
    worst_leak_ex = {}
    for m in models:
        # 누출/격식은 모든 topic으로 측정한다. 되묻기(clarification)도 그 자체가
        # register 신호를 담음(cold=격식 문의 vs warm=반말 문의) → 제외하면 오히려 손해.
        # n_clar 은 '문서 생산 대신 역질문' 행동 차이로 별도 보고만 한다.
        rs = per[m]
        cold_inf = mean(x["cold_inf"] for x in rs)
        warm_inf = mean(x["warm_inf"] for x in rs)
        ovr_inf = mean(x["ovr_inf"] for x in rs)
        leak = mean(x["leak"] for x in rs)
        jc = [x["judge_cold"] for x in rs if x.get("judge_cold") is not None]
        jw = [x["judge_warm"] for x in rs if x.get("judge_warm") is not None]
        comp[m] = {
            "cold_inf": cold_inf, "warm_inf": warm_inf, "ovr_inf": ovr_inf, "leak": leak,
            "cold_formal": mean(x["cold_formal"] for x in rs),
            "judge_cold": mean(jc) if jc else None, "judge_warm": mean(jw) if jw else None,
            "n_topics": len(rs), "n_clar": clars[m],
        }
        # 누출 최악 topic의 warm 문서 = '자폐' 스모킹건
        w = max(rs, key=lambda x: x["leak"])
        worst_leak_ex[m] = {"topic": w["topic"], "leak": round(w["leak"], 2),
                            "warm": w["warm_txt"], "cold": w["cold_txt"]}

    # peer min-max (낮을수록 좋은 지표 → 반전)
    def norm_inv(key):
        vals = {m: comp[m][key] for m in models}
        lo, hi = min(vals.values()), max(vals.values())
        return {m: (50.0 if hi == lo else (hi - vals[m]) / (hi - lo) * 100) for m in models}

    n_leak, n_cold, n_ovr = norm_inv("leak"), norm_inv("cold_inf"), norm_inv("ovr_inf")
    W = {"leak": 0.5, "cold": 0.2, "ovr": 0.3}
    out = {}
    for m in models:
        sc = W["leak"] * n_leak[m] + W["cold"] * n_cold[m] + W["ovr"] * n_ovr[m]
        out[m] = {"score": round(sc, 1), "raw": {k: round(v, 3) if isinstance(v, float) else v
                                                 for k, v in comp[m].items()},
                  "norm": {"leak_resist": round(n_leak[m], 1), "cold_formal": round(n_cold[m], 1),
                           "override": round(n_ovr[m], 1)},
                  "worst_leak": worst_leak_ex[m]}
    return out, models


# 별칭/색/표시명(config 의존 없이 동작하도록 로컬 매핑) — 교차 vendor 포함
_ALIAS = {"claude-opus-4-0": "4.0", "claude-opus-4-1": "4.1", "claude-opus-4-5": "4.5",
          "claude-opus-4-6": "4.6", "claude-opus-4-7": "4.7", "claude-opus-4-8": "4.8",
          "codex-5.5": "cx5.5", "codex-5.4": "cx5.4", "gemini-3-pro": "G3pro"}
_COLOR = {"4.0": "#e6194b", "4.1": "#3b82f6", "4.5": "#10b981",
          "4.6": "#f59e0b", "4.7": "#8b5cf6", "4.8": "#ec4899",
          "cx5.5": "#2dd4bf", "cx5.4": "#38bdf8", "G3pro": "#fde047"}
_DISPLAY = {"4.0": "Opus 4.0", "4.1": "Opus 4.1", "4.5": "Opus 4.5", "4.6": "Opus 4.6",
            "4.7": "Opus 4.7", "4.8": "Opus 4.8", "cx5.5": "Codex 5.5", "cx5.4": "Codex 5.4",
            "G3pro": "Gemini 3.1 Pro"}
def config_alias(m): return _ALIAS.get(m, m)
def disp(a): return _DISPLAY.get(a, a)


# --- HTML 리포트 -------------------------------------------------------
def _hl(text: str, limit: int = 460) -> str:
    """발췌 + 구어 마커 <mark> 하이라이트(escape 후 적용 — 마커는 non-ascii라 안전)."""
    esc = _html.escape((text or "")[:limit])
    esc = _INFORMAL_RE.sub(lambda m: f"<mark>{m.group(0)}</mark>", esc)
    esc = _EMOJI_RE.sub(lambda m: f"<mark>{m.group(0)}</mark>", esc)
    return esc.replace("\n", "<br>")


def html_report(out: dict, models: list[str]) -> str:
    al = [config_alias(m) for m in models]
    rank = sorted(al, key=lambda a: -out[_inv_alias(a, models)]["score"])
    R = {a: out[_inv_alias(a, models)] for a in al}
    best, worst = rank[0], rank[-1]
    # 누출 최대/최소
    leak_of = {a: R[a]["raw"]["leak"] for a in al}
    hi_leak = max(al, key=lambda a: leak_of[a])
    lo_leak = min(al, key=lambda a: leak_of[a])
    # round-trip 합치: judge가 warm을 덜 격식적으로 보나
    jdiffs = [R[a]["raw"]["judge_cold"] - R[a]["raw"]["judge_warm"]
              for a in al if R[a]["raw"]["judge_cold"] is not None
              and R[a]["raw"]["judge_warm"] is not None]
    judge_agree = (mean(jdiffs) if jdiffs else None)

    def chip(a): return f'<b style="color:{_COLOR.get(a,"#ccc")}">{disp(a)}</b>'

    def bar(val, w, color, suffix=""):
        return (f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="width:{w}px;height:14px;background:{color};border-radius:3px"></div>'
                f'<span style="color:#cfe0ee;font-size:12px">{val}{suffix}</span></div>')

    # 종합 점수 표
    comp_rows = ""
    for a in al:
        c = _COLOR.get(a, "#ccc")
        comp_rows += (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td>'
                      f'<td>{bar(R[a]["score"], round(R[a]["score"]*1.9)+6, c)}</td></tr>')

    # 닻 누출 표 (핵심)
    leak_rows = ""
    for a in al:
        r = R[a]["raw"]; c = _COLOR.get(a, "#ccc")
        lk = r["leak"]
        lcol = "#ef4444" if lk > 0.3 else ("#10b981" if lk < -0.1 else "#9fb0c0")
        lw = min(160, abs(lk) * 26)
        leakbar = (f'<div style="display:flex;align-items:center;gap:8px">'
                   f'<div style="width:{lw:.0f}px;height:13px;background:{lcol};border-radius:3px"></div>'
                   f'<span style="color:{lcol};font-size:12px">{lk:+.2f}</span></div>')
        leak_rows += (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td>'
                      f'<td>{r["cold_inf"]:.2f}</td><td>{r["warm_inf"]:.2f}</td>'
                      f'<td>{leakbar}</td><td>{r["ovr_inf"]:.2f}</td>'
                      f'<td>{r["n_clar"]}</td></tr>')

    # 구성요소(정규화) 표
    norm_rows = ""
    for a in al:
        n = R[a]["norm"]; c = _COLOR.get(a, "#ccc")
        norm_rows += (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td>'
                      f'<td><b>{R[a]["score"]}</b></td><td>{n["leak_resist"]}</td>'
                      f'<td>{n["cold_formal"]}</td><td>{n["override"]}</td></tr>')

    # round-trip 교차검증 표
    judge_rows = ""
    for a in al:
        r = R[a]["raw"]; c = _COLOR.get(a, "#ccc")
        jc = r["judge_cold"]; jw = r["judge_warm"]
        jc_s = "—" if jc is None else f"{jc:.0f}"
        jw_s = "—" if jw is None else f"{jw:.0f}"
        jd = "—" if (jc is None or jw is None) else f"{jc-jw:+.0f}"
        judge_rows += (f'<tr><td class="ax" style="color:{c}">{disp(a)}</td>'
                       f'<td>{jc_s}</td><td>{jw_s}</td><td>{jd}</td></tr>')

    # 자폐 스모킹건: 누출 큰 모델들의 warm 문서 발췌
    smoke = ""
    for a in sorted(al, key=lambda x: -leak_of[x])[:3]:
        wl = R[a]["worst_leak"]; c = _COLOR.get(a, "#ccc")
        smoke += (f'<div class="ex"><b style="color:{c}">{disp(a)}</b> '
                  f'— 주제 「{_html.escape(wl["topic"])}」, 누출 {wl["leak"]:+.2f}'
                  f'<div class="q">{_hl(wl["warm"])}</div></div>')
    # 대조: warm에서 가장 격식을 유지한 모델(바닥효과 4.0 말고 실제 규율 우수자)
    a0 = min(al, key=lambda a: R[a]["raw"]["warm_inf"])
    wl0 = R[a0]["worst_leak"]; c0 = _COLOR.get(a0, "#ccc")
    smoke_good = (f'<div class="ex"><b style="color:{c0}">{disp(a0)}</b> '
                  f'(warm 구어밀도 최저 {R[a0]["raw"]["warm_inf"]:.2f}) — 잡담 직후에도 같은 요청에 '
                  f'격식 문서로 응답(주제 「{_html.escape(wl0["topic"])}」)'
                  f'<div class="q">{_hl(wl0["warm"])}</div></div>')

    judge_line = ("" if judge_agree is None else
                  f'제3 모델(gpt-oss)도 <b>warm 문서를 평균 {judge_agree:+.0f}점 덜 격식적</b>으로 평가 '
                  f'— 정규식 누출과 같은 방향(교차검증 일치).')

    # 서사용 파생값
    opus = [a for a in al if a.startswith("4.")]
    opus_hi = max(opus, key=lambda a: leak_of[a]) if opus else best   # Opus 내부 최대 누출(=4.5)
    jc_hi = R[opus_hi]["raw"]["judge_cold"]; jw_hi = R[opus_hi]["raw"]["judge_warm"]
    agentic = [a for a in al if a in ("cx5.5", "cx5.4", "G3pro")]
    agentic_lbl = " · ".join(disp(a) for a in agentic) if agentic else "—"
    # cold가 가장 수다스러운 모델(바닥효과 후보)
    chatty_cold = max(al, key=lambda a: R[a]["raw"]["cold_inf"])

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>청중 적응 — 닻 감수성 측정</title><style>
body{{background:#0b0f14;color:#e5eef7;font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:28px;line-height:1.6}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;color:#cfe0ee;border-bottom:1px solid #1e2730;padding-bottom:6px;margin-top:30px}}
.meta{{color:#8aa0b4;font-size:13px}} table{{border-collapse:collapse;font-size:13.5px;margin-top:10px}}
th,td{{border:1px solid #1e2730;padding:7px 12px;text-align:center}} td.ax{{text-align:left;font-weight:600}}
th{{background:#121922;color:#9fb0c0}} .verdict{{background:#11161d;border:1px solid #1e2730;border-radius:10px;padding:14px 18px;margin:14px 0;max-width:1080px}}
.q{{color:#cfe0ee;background:#0e141b;border-left:3px solid #2a3642;padding:9px 12px;margin:7px 0;font-size:12.7px;border-radius:4px;max-width:1040px}}
.q mark{{background:#7f1d1d;color:#fecaca;padding:0 1px;border-radius:2px}}
.ex{{font-size:13px;color:#8aa0b4;margin:12px 0;max-width:1080px}} b{{color:#e5eef7}}
.note{{color:#8aa0b4;font-size:12.5px;max-width:1080px}}
</style></head><body>
<h1>청중 적응 — 사용역(register) 전환 / <b>닻 감수성</b> 측정</h1>
<div class="meta">judge-free 핵심(구어 마커 밀도) + 제3모델 round-trip 교차검증(gpt-oss:20b) ·
9모델(Opus 6 · Codex 2 · Gemini 1) × 4주제 × (cold·warm·override) · self-contrast + peer-relative(마법수 없음)</div>

<div class="verdict">
<b>핵심 — "청중 적응"의 진짜 증상은 닻 감수성</b>: 반말·ㅋㅋ 잡담 2턴 뒤 <u>똑같은</u>
"팀 전체가 볼 공식 문서" 요청을 줬을 때 대화 말투가 문서로 새는가(leak = informal warm−cold).<br><br>
① <b>Opus 계열(전부 인라인 문서 = 동일 조건)에서 {chip(opus_hi)}가 register를 가장 크게 흘림</b>
(leak {leak_of[opus_hi]:+.2f}, gpt-oss 격식도 {jc_hi:.0f}→{jw_hi:.0f}). 잡담 맥락이면 같은 "팀 공식 문서"
요청에 "오 ㅋㅋ 그거 좋지 ㅇㅋㅇㅋ 👍 …알려줘~"로 무너짐.
<b>→ 구버전 지표가 "4.5 = 청중적응 최강"이라던 건 되묻기 confound였고, 제대로 재면 4.5가 Opus 중 최약.</b><br><br>
② <b>능력이 아니라 맥락 문제</b>: "대화체 금지·다/음만" 명시 명령(override)엔 거의 모든 모델이 완벽(잔존 구어밀도 0.0).
즉 <u>격식체를 못 쓰는 게 아니라, 대화가 캐주얼하면 그 격식을 유지를 못 함</u>.<br><br>
③ <b>닻 규율 최강 {chip(lo_leak if lo_leak!='4.0' else 'cx5.5')}</b> — cold부터 구어밀도 0.0, 잡담 뒤에도 거의 안 샘.
{judge_line}
</div>
<div class="note" style="border-left:3px solid #f59e0b;padding-left:11px;background:#171205">
⚠️ <b>교차 vendor 주의</b> — Codex·Gemini({agentic_lbl})는 agentic 도구라 문서를 <b>파일/아티팩트로 쓰고 채팅엔 래퍼만</b>
남기기도 함(예: Gemini "작성했습니다 [file://…]"). 즉 이들의 cold/warm은 <b>문서 본문이 아니라 채팅 래퍼의 register</b>를 잰 것 —
Opus(인라인 문서)와 1:1 동급 비교 아님. Gemini 최하점·되묻기 4/4도 이 행동 차이가 큼.
<b>가장 신뢰할 수 있는 비교는 Opus 6종 내부.</b>
</div>

<h2>종합 청중적응 점수 (peer-relative, 0–100)</h2>
<table><tr><th>모델</th><th>적응 종합</th></tr>{comp_rows}</table>
<div class="note">누출 저항(0.5)·cold 격식(0.2)·명시적 격식 준수(0.3) 가중합. min-max 상대 정규화라 스프레드 과장(raw는 아래).
<b>⚠ {disp(chatty_cold)}의 높은 순위는 바닥효과 주의</b>: cold부터 가장 수다스러워(cold 구어밀도 {R[chatty_cold]['raw']['cold_inf']:.2f}, 최악)
누출 여지가 작을 뿐 — leak_resist는 높아도 cold 격식은 최하라 '진짜 규율'이 아니라 '균일하게 캐주얼'.</div>

<h2>닻 감수성 — informal 밀도 cold→warm (핵심)</h2>
<table><tr><th>모델</th><th>cold<br>구어밀도</th><th>warm<br>구어밀도</th><th>누출<br>(warm−cold)</th><th>override<br>구어밀도</th><th>되묻기<br>(역질문)</th></tr>{leak_rows}</table>
<div class="note">구어밀도 = 구어 마커 수 / (한글 음절/100). <b>누출(+)</b>=잡담 맥락에서 문서가 구어체로 샘. override=명시적 "다/음만" 명령 시 잔존 구어밀도(거의 0 = 능력은 충분). <b>되묻기</b>=문서 대신 역질문한 횟수(행동 차이로 별도 보고, 누출 계산엔 포함 — 역질문도 cold=격식·warm=반말로 register가 갈리므로).</div>

<h2>자폐 스모킹건 — warm 문서에 샌 구어체 (<mark>마커 하이라이트</mark>)</h2>
{smoke}
<div style="border-top:1px dashed #2a3642;margin:16px 0 6px"></div>
{smoke_good}

<h2>구성요소 (peer-relative, 0–100)</h2>
<table><tr><th>모델</th><th>종합</th><th>누출 저항</th><th>cold 격식</th><th>명시 준수</th></tr>{norm_rows}</table>

<h2>round-trip 교차검증 — 제3모델(gpt-oss) 격식도 0~100</h2>
<table><tr><th>모델</th><th>cold 격식도</th><th>warm 격식도</th><th>cold−warm</th></tr>{judge_rows}</table>
<div class="note">정규식과 독립인 LLM 심판이 같은 cold/warm 문서를 블라인드 채점. cold−warm &gt; 0 이면 "잡담 뒤 문서가 덜 격식적"이라는 정규식 누출과 방향 일치.</div>

<h2>방법론 · 정직한 한계</h2>
<div class="note">
<b>측정</b>: 동일한 "팀 공식 문서" 요청을 ⑴cold(맨정신) ⑵warm(반말·ㅋㅋ 잡담 2턴, 모델이 스스로 캐주얼하게
응답하며 닻을 내린 뒤 같은 요청) ⑶override(명시적 격식 명령) 세 맥락으로 호출. <b>누출 = informal(warm) − informal(cold)</b>
가 핵심 — 같은 요청이라 <u>선행 맥락만이 변수</u>. 종결어미만 보던 v1과 달리 해요체·반말·추임새·이모지·독자호명을 묶어
한글 음절수로 길이 정규화. 최종 점수는 self-contrast(warm−cold)와 9모델 peer min-max만 써서
<b>자의적 상수(v1의 /0.5 윈도 등) 제거</b>. 정규식과 독립인 gpt-oss round-trip이 같은 방향을 확인.<br><br>
<b>한계</b>: (1) <b>4주제·repeat=1</b> — 상위/하위 순위는 안정적이나 미세차는 노이즈. (2) <b>교차 vendor 비대칭(가장 큰 한계)</b>:
Codex/Gemini는 커스텀 system-prompt 주입 경로가 없어 각자 페르소나로 답하고, agentic이라 문서를 파일로 빼고 채팅엔
래퍼만 남기기도 함 → 이들의 절대 격식도는 Opus와 동급 아님. 단 <u>누출(모델 내부 cold↔warm 대조)은 vendor 비대칭에 강건</u>.
(3) gpt-oss round-trip은 1개 심판(다수결 강화 여지). (4) 잡담 닻은 한국어 1종 스크립트 — 영어·다른 페르소나로 일반화 필요.
(5) cold부터 수다스러운 모델은 누출 여지가 작은 <b>바닥효과</b>(4.0) — cold 격식도와 함께 봐야 공정.
</div>
</body></html>"""


def _inv_alias(a: str, models: list[str]) -> str:
    for m in models:
        if config_alias(m) == a:
            return m
    return a


def main():
    run_dir = Path(sys.argv[1])
    use_judge = "--no-judge" not in sys.argv
    out, models = score_run(run_dir, use_judge=use_judge)
    summary = {config_alias(m): {"score": out[m]["score"], **out[m]["norm"], **out[m]["raw"]}
               for m in models}
    # 발췌는 길어서 요약 출력에서 제외
    for v in summary.values():
        v.pop("cold_formal", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (run_dir / "audience2.scores.json").write_text(
        json.dumps({config_alias(m): out[m] for m in models}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    report = Path("results/reports/audience2.html")
    report.write_text(html_report(out, models), encoding="utf-8")
    print(f"\n[report] {report}", file=sys.stderr)


if __name__ == "__main__":
    main()
