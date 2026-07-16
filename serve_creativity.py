#!/usr/bin/env python3
"""blindrank_creativity.html 정적 서빙 + 채점 저장(창의 전용, 원본과 분리).

기존 serve_rank.py 와 동일하되 저장 파일만 분리해 원본 5질문 별점을 안 건드린다.
  python3 serve_creativity.py [port]   # 기본 8765
저장: results/raw/blind_pool_creativity/blindrank_result.json / blindrank_ls.json
"""
from __future__ import annotations

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/raw/blind_pool_creativity"
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

    def log_message(self, fmt, *args):
        if args and "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.chdir(ROOT)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[serve] http://localhost:{port}/blindrank_creativity.html  (POST 저장 활성)")
    print(f"[serve] 저장 → {OUT}/blindrank_result.json")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
