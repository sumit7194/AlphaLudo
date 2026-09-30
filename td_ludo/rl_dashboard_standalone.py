"""Standalone RL dashboard — reads the run's stats files and serves them.

Runs as a SEPARATE process from training (own GIL), so it stays responsive
even when the training loop saturates the trainer's in-process dashboard.
Reads live_stats.json + training_metrics.json from the run dir. Port 8791.
"""
from __future__ import annotations
import json, os, re, time
from http.server import BaseHTTPRequestHandler, HTTPServer

RUN_DIR = os.environ.get("RL_RUN_DIR",
    "/home/sumit/AlphaLudo/td_ludo/checkpoints/v136_rl_parity")
PORT = int(os.environ.get("RL_DASH_PORT", "8791"))


def read_json(name):
    try:
        with open(os.path.join(RUN_DIR, name)) as f:
            return json.load(f)
    except Exception:
        return None


def eval_trajectory():
    d = read_json("training_metrics.json")
    if not d:
        return []
    h = d if isinstance(d, list) else d.get("metrics_history", d.get("history", []))
    out = []
    for e in (h if isinstance(h, list) else []):
        g = e.get("total_games", e.get("games"))
        w = e.get("eval_win_rate", e.get("eval_wr", e.get("win_rate")))
        if g is not None and w is not None:
            out.append((g, w * 100 if w <= 1 else w))
    return out


def log_age():
    p = os.path.join(RUN_DIR, "console.log")
    try:
        return time.time() - os.path.getmtime(p)
    except Exception:
        return None


def render():
    s = read_json("live_stats.json") or {}
    evals = eval_trajectory()
    peak = max(evals, key=lambda x: x[1]) if evals else None
    age = log_age()
    running = age is not None and age < 180
    color = "#22c55e" if running else "#f59e0b"
    status = "RUNNING" if running else f"IDLE? (log {int(age)}s)" if age else "?"

    rows = "".join(
        f"<tr><td>{g:,}</td><td>{w:.1f}%</td></tr>" for g, w in evals[-15:]
    )
    g = s.get("total_games", 0)
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=5><title>V13.6 RL — {g:,}g</title>
<style>
body{{background:#0b0f17;color:#e5e7eb;font-family:Menlo,monospace;margin:0;padding:28px}}
.w{{max-width:720px;margin:0 auto}}h1{{font-size:19px;margin:0 0 2px}}
.badge{{padding:3px 12px;border-radius:999px;background:{color}22;color:{color};
border:1px solid {color}55;font-size:13px;font-weight:600}}
.sub{{color:#6b7280;font-size:12px;margin:6px 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:14px}}
.k{{color:#6b7280;font-size:11px;text-transform:uppercase}}.v{{font-size:22px;font-weight:600;margin-top:3px}}
table{{width:100%;border-collapse:collapse;margin-top:18px;font-size:13px}}
td,th{{text-align:left;padding:4px 8px;border-bottom:1px solid #1f2937}}
.peak{{color:#22c55e;font-weight:600}}
</style></head><body><div class=w>
<h1>V13.6 RL (parity) <span class=badge>{status}</span></h1>
<div class=sub>resumed run · dense+bias · v13_5_no_bots · L4 · standalone reader (GIL-independent)</div>
<div class=grid>
<div class=card><div class=k>Games</div><div class=v>{g:,}</div></div>
<div class=card><div class=k>Best eval</div><div class=v class=peak>{s.get('best_eval_win_rate','—')}%</div></div>
<div class=card><div class=k>GPM</div><div class=v>{s.get('games_per_minute','—')}</div></div>
<div class=card><div class=k>WR (last100, vs pool)</div><div class=v>{s.get('win_rate_100','—')}%</div></div>
<div class=card><div class=k>Value loss</div><div class=v>{s.get('avg_value_loss','—') if not isinstance(s.get('avg_value_loss'),(int,float)) else round(s['avg_value_loss'],3)}</div></div>
<div class=card><div class=k>Entropy</div><div class=v>{s.get('policy_entropy','—')}</div></div>
</div>
<div class=sub style=margin-top:18px>Peak eval: <span class=peak>{f'{peak[1]:.1f}% @ {peak[0]:,}g' if peak else '—'}</span> · auto-refresh 5s</div>
<table><tr><th>games</th><th>eval WR</th></tr>{rows}</table>
</div></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        try:
            if self.path.startswith("/api"):
                body = json.dumps(read_json("live_stats.json") or {}).encode()
                ct = "application/json"
            else:
                body = render().encode(); ct = "text/html; charset=utf-8"
        except Exception as e:
            body = f"err: {e}".encode(); ct = "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    print(f"RL standalone dashboard on 0.0.0.0:{PORT}  (reads {RUN_DIR})")
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
