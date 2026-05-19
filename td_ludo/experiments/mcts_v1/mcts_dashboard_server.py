"""Tiny HTTP server for the MCTS dashboard.

Serves:
  /                   → mcts_dashboard.html
  /api/gen_stats      → reads $BUFFER.stats.json (gen progress)
  /api/sl_stats       → reads training stats (when training is running)

Usage:
    python3 -m experiments.mcts_v1.mcts_dashboard_server \\
        --buffer runs/mcts_v135_buffer_100k.npz \\
        --port 8792
"""
from __future__ import annotations

import argparse
import functools
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, gen_stats=None, sl_stats=None, **kw):
        self._gen_stats = gen_stats
        self._sl_stats = sl_stats
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):
        return  # silence

    def do_GET(self):  # noqa: N802
        if self.path in ("/", ""):
            self.path = "/mcts_dashboard.html"
            return super().do_GET()
        if self.path == "/api/gen_stats":
            return self._serve_file(self._gen_stats)
        if self.path == "/api/sl_stats":
            return self._serve_file(self._sl_stats)
        return super().do_GET()

    def _serve_file(self, path):
        if not path or not os.path.exists(path):
            self.send_response(404)
            self.end_headers()
            return
        try:
            with open(path) as f:
                data = f.read()
        except OSError:
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data.encode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", required=True,
                   help="Buffer .npz path; stats.json is read from <buffer>.stats.json")
    p.add_argument("--sl-stats", default=None,
                   help="Path to training stats.json (optional; auto-discovered "
                        "from checkpoint dir if --run-name is given)")
    p.add_argument("--run-name", default="mcts_v135_step1_distill",
                   help="Training run name (used to find sl_stats.json)")
    p.add_argument("--port", type=int, default=8792)
    args = p.parse_args()

    gen_stats = args.buffer + ".stats.json"
    sl_stats = args.sl_stats
    if sl_stats is None:
        # Default location: checkpoints/<run-name>/sl_stats.json
        repo_root = Path(__file__).resolve().parent.parent.parent
        sl_stats = str(repo_root / "checkpoints" / args.run_name / "sl_stats.json")

    here = Path(__file__).resolve().parent
    handler = functools.partial(_Handler, directory=str(here),
                                gen_stats=gen_stats, sl_stats=sl_stats)
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"[mcts-dashboard] http://localhost:{args.port}/")
    print(f"  gen stats:  {gen_stats}")
    print(f"  sl stats:   {sl_stats}")
    server.serve_forever()


if __name__ == "__main__":
    main()
