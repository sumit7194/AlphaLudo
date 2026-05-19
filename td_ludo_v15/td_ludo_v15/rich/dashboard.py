"""V15 dashboard HTTP server.

Serves the JSON endpoints `v13_dashboard.html` consumes:
    /api/stats     — live rich stats (loss, GPM, ELO rankings, opp stats)
    /api/metrics   — list of per-eval snapshots
    /api/elo       — ELO rankings + history
    /api/games     — recent games from GameDB
    /api/system    — CPU/memory/PID snapshot
    /api/chain     — pipeline stage status

Falls through to static-file serving for the dashboard HTML/CSS/JS. The
target `dashboard_dir` should contain `v13_dashboard.html` (typically
the legacy ~/td_ludo/ directory).

Stats writers are owned by the main loop; this module just serves whatever
the JSON files contain plus a couple of live endpoints (elo, system).
"""
from __future__ import annotations

import functools
import json
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler


class V15RichDashboardHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler subclass with /api/* endpoints."""

    def __init__(
        self, *args, directory=None,
        stats_path=None, metrics_path=None, chain_path=None,
        elo_tracker=None, game_db=None, landing=None, **kw
    ):
        self._stats_path = stats_path
        self._metrics_path = metrics_path
        self._chain_path = chain_path
        self._elo_tracker = elo_tracker
        self._game_db = game_db
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):
        return  # silence per-request noise

    def do_GET(self):  # noqa: N802
        try:
            if self.path in ("/", ""):
                if self._landing:
                    self.path = "/" + self._landing
                return super().do_GET()
            if self.path == "/api/stats":
                return self._serve_json_file(self._stats_path)
            if self.path == "/api/metrics":
                return self._serve_json_file(self._metrics_path)
            if self.path == "/api/chain":
                return self._serve_json_file(self._chain_path)
            if self.path == "/api/sl_stats":
                # The dashboard occasionally polls this; fall through to stats.
                return self._serve_json_file(self._stats_path)
            if self.path == "/api/elo":
                return self._serve_elo()
            if self.path.startswith("/api/games"):
                return self._serve_games()
            if self.path == "/api/system":
                return self._serve_system()
            return super().do_GET()
        except (ConnectionResetError, BrokenPipeError):
            pass

    # ── helpers ─────────────────────────────────────────────────────────────
    def _serve_json_file(self, path):
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
        self._send_json(data.encode())

    def _serve_elo(self):
        if self._elo_tracker is None:
            self._send_json(b'{"rankings": [], "history": {}}')
            return
        try:
            rankings = self._elo_tracker.get_rankings(top_n=15)
            history = getattr(self._elo_tracker, "history", {})
            # Downsample history to ≤200 points per name
            history_out = {}
            for name, h in dict(history).items():
                if len(h) <= 200:
                    history_out[name] = list(h)
                else:
                    step = max(1, len(h) // 200)
                    history_out[name] = list(h)[::step]
            payload = {
                "rankings": [list(t) for t in rankings],
                "history": history_out,
            }
            self._send_json(json.dumps(payload).encode())
        except Exception as e:
            self._send_json(
                json.dumps({"error": str(e), "rankings": [], "history": {}}).encode()
            )

    def _serve_games(self):
        if self._game_db is None:
            self._send_json(b'{"games": []}')
            return
        try:
            recent = self._game_db.get_recent_games(50)
            self._send_json(json.dumps({"games": list(recent)}).encode())
        except Exception as e:
            self._send_json(
                json.dumps({"error": str(e), "games": []}).encode()
            )

    def _serve_system(self):
        try:
            import psutil
            cpu = float(psutil.cpu_percent(interval=None))
            mem = float(psutil.virtual_memory().percent)
            pid = os.getpid()
            payload = {"cpu_percent": cpu, "memory_percent": mem, "pid": pid}
        except Exception:
            payload = {"cpu_percent": 0.0, "memory_percent": 0.0, "pid": os.getpid()}
        self._send_json(json.dumps(payload).encode())

    def _send_json(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def start_dashboard(
    port: int, dashboard_dir: str, stats_path: str, metrics_path: str,
    chain_path: str, elo_tracker=None, game_db=None,
) -> None:
    """Start the dashboard server in a daemon thread."""
    landing = None
    for cand in ("v13_dashboard.html", "v12_dashboard.html",
                 "rl_dashboard.html", "index.html"):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand
            break
    handler = functools.partial(
        V15RichDashboardHandler, directory=dashboard_dir,
        stats_path=stats_path, metrics_path=metrics_path,
        chain_path=chain_path, elo_tracker=elo_tracker,
        game_db=game_db, landing=landing,
    )
    server = HTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[dashboard] http://0.0.0.0:{port}/{landing or ''}", flush=True)
