"""집계 + 레이더 리포트.

run_dir의 원시 JSONL을 모델·축별로 채점해 0–100 점수표를 만들고,
8축 레이더(다각형) SVG와 HTML 리포트를 생성한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import axes as axis_mod
from . import config


def latest_run() -> Path:
    runs = sorted([p for p in config.RAW.glob("*") if (p / "manifest.json").exists()])
    if not runs:
        raise FileNotFoundError("실행 결과가 없습니다. 먼저 `python -m bench run` 하세요.")
    return runs[-1]


def load_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # axis -> model -> [samples]
    by_axis: dict[str, dict[str, list]] = {}
    for axis in manifest["axes"]:
        f = run_dir / f"{axis}.jsonl"
        if not f.exists():
            continue
        by_axis.setdefault(axis, {})
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            by_axis[axis].setdefault(rec["model"], []).append(rec)
    return {"manifest": manifest, "by_axis": by_axis}


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


def ordered_axes(scores: dict) -> list:
    """레이더 표시 순서(하드↔스타일 교대)로 정렬, 나머지는 뒤에."""
    order = [a for a in axis_mod.RADAR_ORDER if a in scores]
    rest = [a for a in scores if a not in order]
    return order + rest


# ---------------------------------------------------------------------------
# 레이더 SVG
# ---------------------------------------------------------------------------
_COLORS = ["#e6194b", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]


# 클라이언트 인터랙티브 차트 엔진 — 체크박스 토글 + 호버 강조(나머지 회백색 음영).
# 표시된 모델만으로 레이더/세대궤적/가설차트를 다시 그리고, 가설차트 평균도 재계산.
INTERACTIVE_JS = r"""
(function(){
const D = __DATA__;
const DIM = '#39454f';
let visible = new Set(D.models.map(m=>m.alias));
let hover = null;
const $ = id => document.getElementById(id);
const colorOf = a => { const m=D.models.find(x=>x.alias===a); return m?m.color:'#888'; };
const vis = () => D.models.filter(m=>visible.has(m.alias));
const sc = (ax,a) => D.scores[ax][a];
const strokeOf = a => (hover && hover!==a) ? DIM : colorOf(a);
const opacOf = a => (hover && hover!==a) ? 0.28 : 1;
const wOf = a => hover===a ? 3.4 : 2;

// Y축 자동 확대: 값 범위에 맞춰 도메인 계산(좁은 밴드도 높낮이 보이게). [0,100] 클램프.
function domain(vals){
  if(!vals.length) return [0,100];
  let lo=Math.min.apply(null,vals), hi=Math.max.apply(null,vals);
  const span=hi-lo, pad=Math.max(2, span*0.25);
  lo-=pad; hi+=pad;
  if(hi-lo<8){ const c=(lo+hi)/2; lo=c-4; hi=c+4; }
  lo=Math.max(0,Math.floor(lo)); hi=Math.min(100,Math.ceil(hi));
  if(hi-lo<2) hi=Math.min(100,lo+2);
  return [lo,hi];
}

function radar(){
  const size=520, cx=size/2, cy=size/2, R=size*0.32, k=D.axes.length;
  const P=['<svg viewBox="0 0 '+size+' '+size+'" width="'+size+'" height="'+size+'" font-family="ui-sans-serif,system-ui">'];
  P.push('<rect width="'+size+'" height="'+size+'" fill="#0b0f14"/>');
  const pt=(i,v)=>{const a=-Math.PI/2+i*2*Math.PI/k, r=R*Math.max(0,Math.min(100,v))/100; return [cx+r*Math.cos(a), cy+r*Math.sin(a)];};
  [25,50,75,100].forEach(g=>{const d=D.axes.map((_,i)=>pt(i,g).map(n=>n.toFixed(1)).join(',')).join(' '); P.push('<polygon points="'+d+'" fill="none" stroke="#27313c"/>');});
  D.axes.forEach((ax,i)=>{const e=pt(i,100); P.push('<line x1="'+cx+'" y1="'+cy+'" x2="'+e[0].toFixed(1)+'" y2="'+e[1].toFixed(1)+'" stroke="#27313c"/>'); const l=pt(i,117); const anc=l[0]<cx-5?'end':(l[0]>cx+5?'start':'middle'); P.push('<text x="'+l[0].toFixed(1)+'" y="'+l[1].toFixed(1)+'" fill="#9fb0c0" font-size="12" text-anchor="'+anc+'" dominant-baseline="middle">'+ax.label+'</text>');});
  vis().forEach(m=>{
    const pts=D.axes.map((ax,i)=>pt(i, sc(ax.key,m.alias)));
    const d=pts.map(p=>p.map(n=>n.toFixed(1)).join(',')).join(' ');
    const fo=(hover===m.alias)?0.18:0;
    P.push('<polygon class="ser" data-m="'+m.alias+'" points="'+d+'" fill="'+colorOf(m.alias)+'" fill-opacity="'+fo+'" stroke="'+strokeOf(m.alias)+'" stroke-width="'+wOf(m.alias)+'" opacity="'+opacOf(m.alias)+'" style="cursor:pointer"/>');
    pts.forEach(p=>P.push('<circle data-m="'+m.alias+'" cx="'+p[0].toFixed(1)+'" cy="'+p[1].toFixed(1)+'" r="'+(hover===m.alias?4:2.6)+'" fill="'+strokeOf(m.alias)+'" opacity="'+opacOf(m.alias)+'"/>'));
  });
  P.push('</svg>'); return P.join('');
}

function trends(){
  const vm=vis();
  const cards=D.axes.map(ax=>{
    const w=300,h=176,pl=34,pr=14,tp=30,bp=26,pw=w-pl-pr,ph=h-tp-bp,n=vm.length;
    const dvals=vm.map(m=>sc(ax.key,m.alias)), dm=domain(dvals), lo=dm[0], hi=dm[1];
    const X=i=>pl+(n>1?pw*i/(n-1):pw/2), Y=v=>tp+ph*(1-(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo));
    const tier=ax.tier==='hard'?'#e6194b':'#3b82f6';
    const s=['<svg viewBox="0 0 '+w+' '+h+'" width="'+w+'" height="'+h+'" font-family="ui-sans-serif">','<rect width="'+w+'" height="'+h+'" fill="#0e141b" rx="8"/>'];
    [lo,(lo+hi)/2,hi].forEach(g=>{const y=Y(g); s.push('<line x1="'+pl+'" y1="'+y+'" x2="'+(w-pr)+'" y2="'+y+'" stroke="#1e2730"/>'); s.push('<text x="'+(pl-4)+'" y="'+y+'" fill="#5b6b7a" font-size="9" text-anchor="end" dominant-baseline="middle">'+g.toFixed(0)+'</text>');});
    s.push('<text x="'+pl+'" y="14" fill="#cfe0ee" font-size="12.5" font-weight="600">'+ax.label+'</text>');
    s.push('<text x="'+(w-pr)+'" y="14" fill="#5b6b7a" font-size="9.5" text-anchor="end">Y '+lo+'–'+hi+'</text>');
    const pts=vm.map((m,i)=>[X(i),Y(sc(ax.key,m.alias)),m.alias,sc(ax.key,m.alias)]);
    s.push('<polyline points="'+pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ')+'" fill="none" stroke="'+tier+'" stroke-width="2.2" opacity="'+(hover?0.3:1)+'"/>');
    pts.forEach(p=>{const hl=hover===p[2]; const col=(hover&&!hl)?DIM:colorOf(p[2]); s.push('<circle data-m="'+p[2]+'" cx="'+p[0].toFixed(1)+'" cy="'+p[1].toFixed(1)+'" r="'+(hl?5:3)+'" fill="'+col+'" style="cursor:pointer"/>'); if(hl||!hover) s.push('<text x="'+p[0].toFixed(1)+'" y="'+(p[1]-8).toFixed(1)+'" fill="#e5eef7" font-size="9.5" text-anchor="middle" opacity="'+(hl?1:(hover?0:0.8))+'">'+p[3].toFixed(0)+'</text>');});
    vm.forEach((m,i)=>s.push('<text x="'+X(i).toFixed(1)+'" y="'+(h-8)+'" fill="#8aa0b4" font-size="10" text-anchor="middle">'+m.alias+'</text>'));
    s.push('</svg>'); return '<div class="card">'+s.join('')+'</div>';
  });
  return '<div class="grid">'+cards.join('')+'</div>';
}

function hyp(){
  const vm=vis(),w=460,h=220,pl=38,pr=16,tp=34,bp=28,pw=w-pl-pr,ph=h-tp-bp,n=vm.length;
  const hard=D.axes.filter(a=>a.tier==='hard'), style=D.axes.filter(a=>a.tier==='style');
  const avg=(set,a)=>set.length?set.reduce((s,x)=>s+sc(x.key,a),0)/set.length:0;
  const allv=vm.map(m=>avg(hard,m.alias)).concat(vm.map(m=>avg(style,m.alias)));
  const dm=domain(allv), lo=dm[0], hi=dm[1];
  const X=i=>pl+(n>1?pw*i/(n-1):pw/2), Y=v=>tp+ph*(1-(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo));
  const s=['<svg viewBox="0 0 '+w+' '+h+'" width="'+w+'" height="'+h+'" font-family="ui-sans-serif">','<rect width="'+w+'" height="'+h+'" fill="#0e141b" rx="8"/>'];
  [lo,(lo+hi)/2,hi].forEach(g=>{const y=Y(g); s.push('<line x1="'+pl+'" y1="'+y+'" x2="'+(w-pr)+'" y2="'+y+'" stroke="#1e2730"/>'); s.push('<text x="'+(pl-5)+'" y="'+y+'" fill="#5b6b7a" font-size="9" text-anchor="end" dominant-baseline="middle">'+g.toFixed(0)+'</text>');});
  s.push('<text x="38" y="16" fill="#cfe0ee" font-size="13" font-weight="600">가설 검증: 하드 vs 스타일 (표시 모델 평균, Y '+lo+'–'+hi+' 확대)</text>');
  [['하드',hard,'#e6194b'],['스타일',style,'#3b82f6']].forEach(t=>{
    const pts=vm.map((m,i)=>[X(i),Y(avg(t[1],m.alias))]);
    s.push('<polyline points="'+pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ')+'" fill="none" stroke="'+t[2]+'" stroke-width="2.4"/>');
    pts.forEach(p=>s.push('<circle cx="'+p[0].toFixed(1)+'" cy="'+p[1].toFixed(1)+'" r="3.2" fill="'+t[2]+'"/>'));
  });
  vm.forEach((m,i)=>s.push('<text x="'+X(i).toFixed(1)+'" y="212" fill="#8aa0b4" font-size="10.5" text-anchor="middle">'+m.alias+'</text>'));
  s.push('<rect x="300" y="26" width="10" height="10" fill="#e6194b" rx="2"/><text x="314" y="35" fill="#cfe0ee" font-size="10">하드</text>');
  s.push('<rect x="356" y="26" width="10" height="10" fill="#3b82f6" rx="2"/><text x="370" y="35" fill="#cfe0ee" font-size="10">스타일</text>');
  s.push('</svg>'); return s.join('');
}

function controls(){
  return D.models.map(m=>'<label class="ck" data-m="'+m.alias+'"><input type="checkbox" checked data-cm="'+m.alias+'"><span class="sw" style="background:'+m.color+'"></span>Opus '+m.alias+'</label>').join('');
}

function renderAll(){
  $('radarBox').innerHTML=radar();
  $('hypBox').innerHTML=hyp();
  $('trendBox').innerHTML=trends();
  document.querySelectorAll('[data-col]').forEach(el=>{ el.style.display = visible.has(el.getAttribute('data-col'))?'':'none'; });
}
function setHover(a){ if(hover!==a){ hover=a; renderAll(); } }

const ctl=$('controls');
ctl.innerHTML=controls();
ctl.addEventListener('change', e=>{ const a=e.target.getAttribute('data-cm'); if(a){ if(e.target.checked) visible.add(a); else visible.delete(a); renderAll(); }});
ctl.addEventListener('mouseover', e=>{ const l=e.target.closest('.ck'); if(l) setHover(l.getAttribute('data-m')); });
ctl.addEventListener('mouseleave', ()=> setHover(null));
['radarBox','trendBox'].forEach(id=>{ const c=$(id); c.addEventListener('mouseover', e=>{ const el=e.target.closest('[data-m]'); if(el) setHover(el.getAttribute('data-m')); }); c.addEventListener('mouseleave', ()=> setHover(null)); });
renderAll();
})();
"""


def interactive_block(axes_order: list[str], models: list[str], score_of) -> str:
    ms = order_models(models)
    data = {
        "models": [{"alias": config.alias(m), "color": _COLORS[i % len(_COLORS)]}
                   for i, m in enumerate(ms)],
        "axes": [{"key": a, "label": axis_mod.AXIS_META.get(a, {}).get("label", a),
                  "tier": axis_mod.AXIS_META.get(a, {}).get("tier", "style")}
                 for a in axes_order],
        "scores": {a: {config.alias(m): round(score_of(a, m), 2) for m in ms}
                   for a in axes_order},
    }
    js = INTERACTIVE_JS.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    return (
        '<div class="hyp"><div id="hypBox"></div></div>'
        '<div class="chart-h" style="margin-top:14px">축별 세대 궤적 — 체크박스로 토글 · 라인/점에 호버하면 그 모델만 강조</div>'
        '<div id="trendBox"></div>'
        '<div style="margin-top:18px"><div class="chart-h">성격 모양: 전 세대 겹쳐보기 (체크박스 토글 · 호버 강조)</div>'
        '<div id="radarBox"></div></div>'
        f'<script>{js}</script>'
    )


def radar_svg(axes_order: list[str], models: list[str],
              score_of, size: int = 520) -> str:
    """score_of(axis, model) -> 0..100 을 받아 겹친 레이더 다각형 SVG 문자열 반환."""
    cx = cy = size / 2
    R = size * 0.34
    k = len(axes_order)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
             f'viewBox="0 0 {size} {size}" font-family="ui-sans-serif,system-ui,sans-serif">']
    parts.append(f'<rect width="{size}" height="{size}" fill="#0b0f14"/>')

    def pt(i, val):
        ang = -math.pi / 2 + i * 2 * math.pi / k
        r = R * max(0.0, min(100.0, val)) / 100.0
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    # 그리드 링
    for ring in (25, 50, 75, 100):
        pts = [pt(i, ring) for i in range(k)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polygon points="{d}" fill="none" stroke="#27313c" stroke-width="1"/>')
    # 스포크 + 축 라벨
    for i, axis in enumerate(axes_order):
        ex, ey = pt(i, 100)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
                     f'stroke="#27313c" stroke-width="1"/>')
        lx, ly = pt(i, 116)
        label = axis_mod.AXIS_META.get(axis, {}).get("label", axis)
        anchor = "middle"
        if lx < cx - 5:
            anchor = "end"
        elif lx > cx + 5:
            anchor = "start"
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#9fb0c0" font-size="13" '
                     f'text-anchor="{anchor}" dominant-baseline="middle">{label}</text>')

    # 모델별 다각형 — 다수(>3)면 채움 끄고 색선만(뭉침 방지)
    many = len(models) > 3
    fop = 0.0 if many else 0.16
    sw = 1.8 if many else 2.0
    for mi, model in enumerate(models):
        color = _COLORS[mi % len(_COLORS)]
        pts = [pt(i, score_of(axis, model)) for i, axis in enumerate(axes_order)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polygon points="{d}" fill="{color}" fill-opacity="{fop}" '
                     f'stroke="{color}" stroke-width="{sw}"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.4 if many else 3}" fill="{color}"/>')

    # 범례
    ly = 18
    for mi, model in enumerate(models):
        color = _COLORS[mi % len(_COLORS)]
        parts.append(f'<rect x="14" y="{ly-9}" width="12" height="12" fill="{color}" rx="2"/>')
        parts.append(f'<text x="32" y="{ly}" fill="#e5eef7" font-size="13" '
                     f'dominant-baseline="middle">Opus {config.alias(model)}</text>')
        ly += 20

    parts.append("</svg>")
    return "\n".join(parts)


def bars_svg(axes_order: list[str], models: list[str], score_of,
             width: int = 560) -> str:
    """축당 모델 막대를 나란히 그린 그룹 가로 막대 차트.

    레이더는 ≥3축부터 면적이 생기므로, 축이 적을 때의 주 시각화는 막대가 맞다.
    """
    pad_l, pad_r, top, row_h, bar_h = 150, 64, 34, 56, 18
    plot_w = width - pad_l - pad_r
    height = top + len(axes_order) * row_h + 16
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">']
    P.append(f'<rect width="{width}" height="{height}" fill="#0b0f14"/>')
    # 0~100 그리드
    for g in (0, 25, 50, 75, 100):
        x = pad_l + plot_w * g / 100
        P.append(f'<line x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{height-12}" '
                 f'stroke="#1e2730" stroke-width="1"/>')
        P.append(f'<text x="{x:.1f}" y="{top-12}" fill="#5b6b7a" font-size="11" '
                 f'text-anchor="middle">{g}</text>')
    for ai, axis in enumerate(axes_order):
        y0 = top + ai * row_h
        label = axis_mod.AXIS_META.get(axis, {}).get("label", axis)
        P.append(f'<text x="{pad_l-12}" y="{y0+row_h/2-2}" fill="#cfe0ee" font-size="13" '
                 f'text-anchor="end" dominant-baseline="middle">{label}</text>')
        for mi, model in enumerate(models):
            v = max(0.0, min(100.0, score_of(axis, model)))
            bw = plot_w * v / 100
            by = y0 + 4 + mi * (bar_h + 4)
            color = _COLORS[mi % len(_COLORS)]
            P.append(f'<rect x="{pad_l}" y="{by}" width="{max(1,bw):.1f}" height="{bar_h}" '
                     f'rx="3" fill="{color}" fill-opacity="0.85"/>')
            P.append(f'<text x="{pad_l+bw+6:.1f}" y="{by+bar_h/2}" fill="#e5eef7" '
                     f'font-size="12" dominant-baseline="middle">{v:.1f}</text>')
    P.append("</svg>")
    return "\n".join(P)


# ---------------------------------------------------------------------------
# 다모델(세대 사다리) — 궤적 라인차트
# ---------------------------------------------------------------------------
def gen_value(model: str) -> float:
    """세대 순서용 버전 숫자(별칭 '4.8'→4.8). 파싱 실패 시 0."""
    try:
        return float(config.alias(model))
    except (TypeError, ValueError):
        return 0.0


def order_models(models: list[str]) -> list[str]:
    return sorted(models, key=gen_value)


def line_chart_svg(title: str, xlabels: list[str], values: list[float],
                   color: str = "#3b82f6", width: int = 300, height: int = 176,
                   subtitle: str = "") -> str:
    pad_l, pad_r, pad_t, pad_b = 30, 14, 30, 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(values)
    def X(i): return pad_l + (pw * i / (n - 1) if n > 1 else pw / 2)
    def Y(v): return pad_t + ph * (1 - max(0.0, min(100.0, v)) / 100.0)
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">']
    P.append(f'<rect width="{width}" height="{height}" fill="#0e141b" rx="8"/>')
    for g in (0, 50, 100):
        y = Y(g)
        P.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" '
                 f'stroke="#1e2730" stroke-width="1"/>')
        P.append(f'<text x="{pad_l-4}" y="{y:.1f}" fill="#5b6b7a" font-size="9" '
                 f'text-anchor="end" dominant-baseline="middle">{g}</text>')
    P.append(f'<text x="{pad_l}" y="14" fill="#cfe0ee" font-size="12.5" font-weight="600">{title}</text>')
    if subtitle:
        P.append(f'<text x="{width-pad_r}" y="14" fill="#8aa0b4" font-size="10.5" '
                 f'text-anchor="end">{subtitle}</text>')
    pts = [(X(i), Y(v)) for i, v in enumerate(values)]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    P.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    for i, (x, y) in enumerate(pts):
        P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        P.append(f'<text x="{x:.1f}" y="{y-7:.1f}" fill="#e5eef7" font-size="9.5" '
                 f'text-anchor="middle">{values[i]:.0f}</text>')
        P.append(f'<text x="{x:.1f}" y="{height-8}" fill="#8aa0b4" font-size="10" '
                 f'text-anchor="middle">{xlabels[i]}</text>')
    P.append("</svg>")
    return "\n".join(P)


def trend_block(axes_order: list[str], models: list[str], score_of) -> str:
    """세대 궤적: 축별 미니 라인차트(small multiples) + 하드/스타일 가설 차트."""
    ms = order_models(models)
    xl = [config.alias(m) for m in ms]
    charts = []
    for axis in axes_order:
        meta = axis_mod.AXIS_META.get(axis, {})
        label = meta.get("label", axis)
        color = "#e6194b" if meta.get("tier") == "hard" else "#3b82f6"
        vals = [score_of(axis, m) for m in ms]
        slope = vals[-1] - vals[0]
        arrow = "▲" if slope > 3 else ("▼" if slope < -3 else "→")
        sub = f"{xl[0]}→{xl[-1]} {arrow}{slope:+.0f}"
        charts.append(line_chart_svg(label, xl, vals, color=color, subtitle=sub))

    # 가설 차트: 하드축 평균 vs 스타일축 평균(세대별)
    hard = [a for a in axes_order if axis_mod.AXIS_META.get(a, {}).get("tier") == "hard"]
    style = [a for a in axes_order if axis_mod.AXIS_META.get(a, {}).get("tier") == "style"]
    def avg(axset, m):
        vs = [score_of(a, m) for a in axset]
        return sum(vs) / len(vs) if vs else 0.0
    hard_vals = [avg(hard, m) for m in ms]
    style_vals = [avg(style, m) for m in ms]
    hyp = [f'<svg xmlns="http://www.w3.org/2000/svg" width="460" height="220" '
           f'viewBox="0 0 460 220" font-family="ui-sans-serif,system-ui,sans-serif">']
    hyp.append('<rect width="460" height="220" fill="#0e141b" rx="8"/>')
    pad_l, pad_r, pad_t, pad_b = 36, 16, 34, 28
    pw, ph = 460 - pad_l - pad_r, 220 - pad_t - pad_b
    n = len(ms)
    def HX(i): return pad_l + (pw * i / (n - 1) if n > 1 else pw / 2)
    def HY(v): return pad_t + ph * (1 - max(0.0, min(100.0, v)) / 100.0)
    for g in (0, 25, 50, 75, 100):
        y = HY(g)
        hyp.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{460-pad_r}" y2="{y:.1f}" stroke="#1e2730"/>')
        hyp.append(f'<text x="{pad_l-5}" y="{y:.1f}" fill="#5b6b7a" font-size="9" text-anchor="end" dominant-baseline="middle">{g}</text>')
    hyp.append('<text x="36" y="16" fill="#cfe0ee" font-size="13" font-weight="600">가설 검증: 하드(실행) vs 스타일 세대 궤적</text>')
    for vals, color, name in [(hard_vals, "#e6194b", "하드(도구+지시)"),
                              (style_vals, "#3b82f6", "스타일(창의+밀도+청중+아첨)")]:
        pts = [(HX(i), HY(v)) for i, v in enumerate(vals)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        hyp.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for i, (x, y) in enumerate(pts):
            hyp.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"/>')
    for i, m in enumerate(ms):
        hyp.append(f'<text x="{HX(i):.1f}" y="212" fill="#8aa0b4" font-size="10.5" text-anchor="middle">{xl[i]}</text>')
    # 범례
    hyp.append('<rect x="300" y="26" width="10" height="10" fill="#e6194b" rx="2"/><text x="314" y="35" fill="#cfe0ee" font-size="10">하드</text>')
    hyp.append('<rect x="356" y="26" width="10" height="10" fill="#3b82f6" rx="2"/><text x="370" y="35" fill="#cfe0ee" font-size="10">스타일</text>')
    hyp.append("</svg>")

    cards = "".join(f'<div class="card">{c}</div>' for c in charts)
    return (f'<div class="hyp">{"".join(hyp)}</div>'
            f'<div class="chart-h" style="margin-top:14px">축별 세대 궤적 (4.0 → 4.8)</div>'
            f'<div class="grid">{cards}</div>')


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


def html_report(scored: dict, findings_html: str = "") -> str:
    manifest = scored["manifest"]
    scores = scored["scores"]
    models = manifest["models"]
    axes_order = ordered_axes(scores)

    def score_of(axis, model):
        return scores[axis][model].score

    multi = len(models) > 2
    disp = order_models(models) if multi else models  # 표 표시 순서

    # 인터랙티브 차트(체크박스 토글 + 호버 강조). 컨트롤은 #controls(legend 자리)에 JS가 채움.
    charts_section = interactive_block(axes_order, models, score_of)
    legend = '<div id="controls" class="controls"></div>'

    # 판정 요약
    if multi:
        ms = order_models(models)
        def avg_tier(tier, m):
            vs = [score_of(a, m) for a in axes_order
                  if axis_mod.AXIS_META.get(a, {}).get("tier") == tier]
            return sum(vs) / len(vs) if vs else 0.0
        h0, h1 = avg_tier("hard", ms[0]), avg_tier("hard", ms[-1])
        s0, s1 = avg_tier("style", ms[0]), avg_tier("style", ms[-1])
        a0, aN = config.alias(ms[0]), config.alias(ms[-1])
        hyp_holds = (h1 - h0) > 3 and (s1 - s0) < -3
        verdict = (
            f"<b>세대 궤적 판정</b> ({a0}→{aN}) — "
            f"<span class='win48'>하드(실행) {h0:.0f}→{h1:.0f} ({h1-h0:+.0f})</span> · "
            f"<span class='win46'>스타일 {s0:.0f}→{s1:.0f} ({s1-s0:+.0f})</span><br>"
            f"<span class='tie'>가설(세대↑ = 코딩↑·스타일↓ '자폐화'): "
            f"{'성립 경향' if hyp_holds else '미성립 — 두 곡선이 함께 움직이거나 평탄'}. "
            f"각 축 곡선은 아래 small-multiples 참조.</span>")
    else:
        m0, m1 = models[0], models[1]
        lead0, lead1, ties, total_gap = [], [], [], 0.0
        for axis in axes_order:
            gap = scores[axis][m0].score - scores[axis][m1].score
            total_gap += abs(gap)
            lab = axis_mod.AXIS_META.get(axis, {}).get("label", axis)
            if abs(gap) < 3:
                ties.append(lab)
            elif gap > 0:
                lead0.append(f"{lab}(+{gap:.0f})")
            else:
                lead1.append(f"{lab}(+{-gap:.0f})")
        a0, a1 = config.alias(m0), config.alias(m1)
        verdict = (
            f"<b>판정 요약</b> — "
            f"<span class='win48'>Opus {a0} 우위:</span> {', '.join(lead0) or '없음'} · "
            f"<span class='win46'>Opus {a1} 우위:</span> {', '.join(lead1) or '없음'} · "
            f"<span class='tie'>동률:</span> {', '.join(ties) or '없음'}<br>"
            f"<span class='tie'>비대칭 총량 Σ|격차| = {total_gap:.0f}</span>")

    # 점수표(표시 순서). data-col로 체크박스 토글 시 컬럼 숨김.
    head = "".join(f'<th data-col="{config.alias(m)}">Opus {config.alias(m)}</th>' for m in disp)
    rows = []
    for axis in axes_order:
        meta = axis_mod.AXIS_META.get(axis, {})
        label = meta.get("label", axis)
        fav = meta.get("fav", "—")
        cells = []
        vals = [scores[axis][m].score for m in disp]
        best = max(vals) if vals else 0
        for m in disp:
            v = scores[axis][m].score
            n = scores[axis][m].n
            mark = " ★" if v == best and len([x for x in vals if x == best]) == 1 else ""
            cells.append(f'<td data-col="{config.alias(m)}">{v:.1f}<span class="n">(n={n})</span>{mark}</td>')
        rows.append(f"<tr><td class='ax'>{label}<span class='fav'>가설:{fav}</span></td>{''.join(cells)}</tr>")

    # 세부 신호(축별 subscores)
    sub_blocks = []
    for axis in axes_order:
        for m in disp:
            r = scores[axis][m]
            if r.subscores:
                kv = " · ".join(f"{k}={v}" for k, v in r.subscores.items())
                sub_blocks.append(f"<div class='sub'><b>{axis_mod.AXIS_META.get(axis,{}).get('label',axis)} / "
                                  f"Opus {config.alias(m)}</b>: {kv}{(' — ' + r.note) if r.note else ''}</div>")

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>llm-bench {manifest['run_id']}</title>
<style>
  body{{background:#0b0f14;color:#e5eef7;font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:28px;}}
  h1{{font-size:20px;margin:0 0 4px}} .meta{{color:#8aa0b4;font-size:13px;margin-bottom:20px}}
  .wrap{{display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start}}
  table{{border-collapse:collapse;font-size:14px}}
  th,td{{border:1px solid #1e2730;padding:8px 12px;text-align:center}}
  td.ax{{text-align:left;color:#cfe0ee}} .fav{{color:#5b6b7a;font-size:11px;margin-left:8px}}
  .n{{color:#5b6b7a;font-size:11px;margin-left:4px}} th{{background:#121922;color:#9fb0c0}}
  .sub{{color:#8aa0b4;font-size:12px;margin:4px 0}} .subs{{margin-top:18px;max-width:620px}}
  code{{color:#9fe0a0}} .chart-h{{color:#9fb0c0;font-size:13px;margin:0 0 6px}}
  .legend{{display:flex;gap:18px;margin:6px 0 14px;font-size:13px}}
  .legend span{{display:inline-flex;align-items:center;gap:6px}}
  .sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}
  .radar-note{{color:#8aa0b4;font-size:13px;max-width:300px;line-height:1.6;
    border:1px dashed #27313c;border-radius:8px;padding:14px}}
  .verdict{{background:#11161d;border:1px solid #1e2730;border-radius:10px;padding:14px 18px;
    margin:14px 0 6px;max-width:1000px;line-height:1.7;font-size:13.5px}}
  .findings{{margin:26px 0;max-width:1000px;line-height:1.7;font-size:14px}}
  .findings h2{{font-size:17px;color:#cfe0ee;border-bottom:1px solid #1e2730;padding-bottom:6px;margin-top:26px}}
  .findings h3{{font-size:14.5px;color:#9fe0a0;margin:16px 0 4px}}
  .findings b{{color:#e5eef7}} .findings .q{{color:#8aa0b4;border-left:2px solid #27313c;padding-left:10px;margin:6px 0}}
  .win48{{color:#e6194b}} .win46{{color:#3b82f6}} .tie{{color:#8aa0b4}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px;max-width:1000px}}
  .card{{line-height:0}} .hyp{{margin-bottom:4px}}
  .controls{{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 16px}}
  .ck{{display:inline-flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;
    user-select:none;padding:3px 8px;border:1px solid #1e2730;border-radius:7px;background:#11161d}}
  .ck:hover{{border-color:#3a4654;background:#161d27}}
  .ck input{{accent-color:#9fb0c0;cursor:pointer;margin:0}}
  #radarBox svg,#hypBox svg,#trendBox svg{{display:block}}
</style></head><body>
<h1>llm-bench — Opus 세대 사다리 비교 ({len(disp)}모델 × {len(axes_order)}축)</h1>
<div class="meta">run <code>{manifest['run_id']}</code> · effort {manifest['effort']} ·
 repeats {manifest['repeats']} · cost ${manifest['total_cost_usd']} ·
 errors {manifest['n_errors']} · 가설: 세대↑ = 코딩/실행↑, 대화/창의/스타일↓ ("벤치 최적화 → 자폐화")</div>
<div class="verdict">{verdict}</div>
{correlation_card(scored)}
<div class="legend">{legend}</div>
{charts_section}
<div style="margin-top:22px;overflow-x:auto"><table><tr><th>축</th>{head}</tr>{''.join(rows)}</table></div>
<div class="subs"><div style="color:#5b6b7a;font-size:12px;margin:14px 0 6px">세부 신호</div>
{''.join(sub_blocks)}</div>
{findings_html}
<p class="meta" style="margin-top:20px">★ = 단독 우위. 점수는 natural-scale 0–100(자의적 상수 없음).
상대/백분위 정규화는 모델·인간 코퍼스 확장 시 적용 예정.</p>
</body></html>"""


def build(run_dir: Path | None = None) -> dict:
    config.ensure_dirs()
    run_dir = run_dir or latest_run()
    scored = score_run(run_dir)
    manifest = scored["manifest"]

    out_html = config.REPORTS / f"{manifest['run_id']}.html"
    out_svg = config.REPORTS / f"{manifest['run_id']}.svg"
    out_json = config.REPORTS / f"{manifest['run_id']}.scores.json"

    # 분석 섹션(워크플로우가 생성한 findings 조각이 있으면 주입)
    findings_file = config.REPORTS / f"{manifest['run_id']}.findings.html"
    findings_html = findings_file.read_text(encoding="utf-8") if findings_file.exists() else ""

    out_html.write_text(html_report(scored, findings_html), encoding="utf-8")
    axes_order = ordered_axes(scored["scores"])
    out_svg.write_text(
        radar_svg(axes_order, manifest["models"],
                  lambda a, m: scored["scores"][a][m].score),
        encoding="utf-8")
    # 기계 판독용 점수 덤프
    dump = {"manifest": manifest, "scores": {
        a: {m: {"score": scored["scores"][a][m].score,
                "n": scored["scores"][a][m].n,
                "subscores": scored["scores"][a][m].subscores}
            for m in manifest["models"]}
        for a in axes_order}}
    out_json.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[report] {out_html}")
    print(f"[report] {out_svg}")
    print(f"[report] {out_json}")
    return {"html": out_html, "svg": out_svg, "json": out_json, "scored": scored}
