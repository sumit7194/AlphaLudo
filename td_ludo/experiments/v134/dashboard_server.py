"""V13.4 chain unified dashboard server.

Serves dashboard.html on / and these JSON endpoints from the run dir:
    /api/chain   → chain_status.json (current phase)
    /api/sl      → sl_stats.json     (SL progress)
    /api/rl      → rl_stats.json     (RL progress)
    /api/h2h     → h2h_results.json  (final tournament results, after Phase 3)

Usage:
    python experiments/v134/dashboard_server.py [--port 8796] [--run v134]
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8796)
    p.add_argument("--run", default="v134")
    return p.parse_args()


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, ckpt_dir=None, html_dir=None, **kw):
        self._ckpt_dir = ckpt_dir
        super().__init__(*args, directory=html_dir, **kw)

    def do_GET(self):
        # API endpoints
        api_map = {
            "/api/chain": "chain_status.json",
            "/api/sl":    "sl_stats.json",
            "/api/rl":    "rl_stats.json",
            "/api/h2h":   "h2h_results.json",
        }
        if self.path in api_map:
            return self._serve_json(os.path.join(self._ckpt_dir, api_map[self.path]))
        # Static file
        if self.path in ("/", ""):
            self.path = "/dashboard.html"
        return super().do_GET()

    def _serve_json(self, path):
        try:
            with open(path) as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"null")

    def log_message(self, *args):
        pass  # silent


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.abspath(os.path.join(here, "..", ".."))
    ckpt_dir = os.path.join(project, "checkpoints", args.run)
    os.makedirs(ckpt_dir, exist_ok=True)
    handler = functools.partial(_Handler, ckpt_dir=ckpt_dir, html_dir=here)
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"[dashboard] http://localhost:{args.port}/  (run={args.run}, ckpt_dir={ckpt_dir})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
