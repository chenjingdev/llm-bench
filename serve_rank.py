#!/usr/bin/env python3
"""blindrank.html 정적 서빙 + 채점 저장 엔드포인트.

기본 http.server 는 POST 를 못 받아 앱의 saveServer()(/save_rank)가 501 로 튕긴다.
이 서버는 POST /save_rank, /dump_ls 를 받아 디스크에 저장 → 채점이 파일로 영속화된다.

  python3 serve_rank.py [port]      # 기본 8777

저장 위치: results/raw/blind_pool_v2/blindrank_result.json (매 저장마다 최신본 덮어씀)
          results/raw/blind_pool_v2/blindrank_ls.json     (localStorage 원본 스냅샷)
"""
from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/raw/blind_pool_v2"
OUT.mkdir(parents=True, exist_ok=True)

SAVE_MAP = {
    "/save_rank": OUT / "blindrank_result.json",
    "/dump_ls": OUT / "blindrank_ls.json",
}


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        dest = SAVE_MAP.get(self.path.split("?")[0])
        if dest is None:
            self.send_error(404, "no such endpoint")
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        try:
            # 유효 JSON 인지 검증(깨진 바디 저장 방지) 후 예쁘게 저장
            obj = json.loads(body.decode("utf-8"))
            dest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[save] {self.path} → {dest.name} ({dest.stat().st_size}B)")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as e:  # noqa: BLE001
            print(f"[err] {self.path}: {e}")
            self.send_error(400, f"bad body: {e}")

    def log_message(self, fmt, *args):  # 소음 감소: GET 200 은 조용히
        if "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    import os
    os.chdir(ROOT)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[serve] http://localhost:{port}/blindrank.html  (POST 저장 활성)")
    print(f"[serve] 저장 → {OUT}/blindrank_result.json")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
