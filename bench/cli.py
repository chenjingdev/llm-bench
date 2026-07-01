"""CLI — python -m bench {run|report|score|smoke}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import axes as axis_mod
from . import config, report, runner

ALL_AXES = list(axis_mod.SCORERS.keys())


def cmd_run(a):
    axes = a.axes or ALL_AXES
    models = a.models or config.DEFAULT_MODELS
    run_dir = runner.run(
        axes, models, effort=a.effort, repeats=a.repeats,
        workers=a.workers, limit=a.limit, seed=a.seed,
    )
    if not a.no_report:
        report.build(run_dir)


def cmd_report(a):
    run_dir = Path(a.run) if a.run else None
    report.build(run_dir)


def cmd_score(a):
    run_dir = Path(a.run) if a.run else report.latest_run()
    scored = report.score_run(run_dir)
    out = {}
    for axis, per in scored["scores"].items():
        out[axis] = {config.alias(m): {"score": r.score, "n": r.n, **r.subscores}
                     for m, r in per.items()}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_smoke(a):
    """최소 비용 스모크: haiku로 축당 probe 1개씩만."""
    run_dir = runner.run(
        a.axes or ALL_AXES, ["claude-haiku-4-5"],
        effort="low", repeats=1, workers=2, limit=1, seed=a.seed,
    )
    report.build(run_dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench", description="llm-bench N-of-1 모델 비교")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="벤치 실행")
    r.add_argument("--axes", nargs="*", choices=ALL_AXES, help=f"기본: {ALL_AXES}")
    r.add_argument("--models", nargs="*", help=f"기본: {config.DEFAULT_MODELS}")
    r.add_argument("--effort", default=config.DEFAULT_EFFORT)
    r.add_argument("--repeats", type=int, default=config.DEFAULT_REPEATS)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--limit", type=int, default=None, help="축당 probe 수 제한(스모크용)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-report", action="store_true")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="최신/지정 run으로 레이더 리포트 생성")
    rp.add_argument("--run", help="results/raw/<run_id> 경로(기본: 최신)")
    rp.set_defaults(func=cmd_report)

    sc = sub.add_parser("score", help="점수만 JSON으로 출력")
    sc.add_argument("--run")
    sc.set_defaults(func=cmd_score)

    sm = sub.add_parser("smoke", help="haiku 최소 호출 스모크 테스트")
    sm.add_argument("--axes", nargs="*", choices=ALL_AXES)
    sm.add_argument("--seed", type=int, default=0)
    sm.set_defaults(func=cmd_smoke)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
