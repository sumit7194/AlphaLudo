"""Tiny standalone dashboard for the SL data-generation job.

Tracks generate_sl_dataset.py progress by reading shards on disk + the log.
No external deps. Auto-refreshes every 4s.

    ./td_env/bin/python gen_dashboard.py
    open http://localhost:8815/
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "checkpoints" / "sl_dataset_v1"
LOG = DATA_DIR / "generate_append_1M.log"
PORT = 8815

# Baseline: shards that existed before this 1M append job.
BASE_SHARDS = 500
TARGET_NEW_GAMES = 1_000_000
GAMES_PER_SHARD = 1000

PROG_RE = re.compile(
    r"\[shard\s+(\d+)\]\s+(\d+)\s+games.*?=\s*([\d.]+)\s*g/s.*?total:\s*([\d,]+)/([\d,]+)"
)


def is_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "generate_sl_dataset.py"],
            capture_output=True, text=True, timeout=4,
        )
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def tail_progress(n: int = 40):
    """Return list of recent (shard, gps) tuples + last full match dict."""
    if not LOG.exists():
        return [], None
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            back = min(size, 200_000)
            f.seek(size - back)
            text = f.read().decode("utf-8", "replace")
    except Exception:
        return [], None
    hist = []
    last = None
    for m in PROG_RE.finditer(text):
        shard = int(m.group(1))
        gps = float(m.group(3))
        done = int(m.group(4).replace(",", ""))
        total = int(m.group(5).replace(",", ""))
        hist.append((shard, gps))
        last = {"shard": shard, "gps": gps, "done": done, "total": total}
    return hist[-n:], last


def gather():
    npz = glob.glob(str(DATA_DIR / "shard_*.npz"))
    n_shards = len(npz)
    new_shards = max(0, n_shards - BASE_SHARDS)
    new_games = new_shards * GAMES_PER_SHARD
    pct = 100.0 * new_games / TARGET_NEW_GAMES if TARGET_NEW_GAMES else 0.0

    hist, last = tail_progress()
    running = is_running()

    # rolling avg g/s over recent history
    recent = [g for _, g in hist[-15:]] if hist else []
    avg_gps = sum(recent) / len(recent) if recent else 0.0
    remaining = max(0, TARGET_NEW_GAMES - new_games)
    eta_min = (remaining / avg_gps / 60.0) if avg_gps > 0 else None

    # log freshness
    log_age = None
    if LOG.exists():
        log_age = time.time() - LOG.stat().st_mtime

    return {
        "n_shards": n_shards, "new_shards": new_shards, "new_games": new_games,
        "pct": pct, "running": running, "avg_gps": avg_gps, "eta_min": eta_min,
        "last": last, "hist": hist, "log_age": log_age, "remaining": remaining,
        "total_games": n_shards * GAMES_PER_SHARD,
    }


def fmt_eta(mins):
    if mins is None:
        return "—"
    h = int(mins // 60)
    m = int(mins % 60)
    return f"{h}h {m}m" if h else f"{m}m"


def sparkline(hist):
    if not hist:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    vals = [g for _, g in hist[-60:]]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(chars[min(7, int((v - lo) / rng * 7))] for v in vals)


def render():
    d = gather()
    last = d["last"] or {}
    status_color = "#22c55e" if d["running"] else "#ef4444"
    status_text = "RUNNING" if d["running"] else "STOPPED"
    if d["running"] and d["log_age"] is not None and d["log_age"] > 240:
        status_color = "#f59e0b"
        status_text = f"STALLED? (log idle {int(d['log_age'])}s)"

    bar_w = d["pct"]
    spark = sparkline(d["hist"])

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="4">
<title>SL Data Gen — {d['pct']:.1f}%</title>
<style>
  body {{ background:#0b0f17; color:#e5e7eb; font-family:-apple-system,Menlo,monospace;
         margin:0; padding:32px; }}
  .wrap {{ max-width:760px; margin:0 auto; }}
  h1 {{ font-size:20px; margin:0 0 4px; font-weight:600; }}
  .sub {{ color:#6b7280; font-size:13px; margin-bottom:24px; }}
  .badge {{ display:inline-block; padding:3px 12px; border-radius:999px;
            background:{status_color}22; color:{status_color}; font-weight:600;
            font-size:13px; border:1px solid {status_color}55; }}
  .bar-bg {{ background:#1f2937; border-radius:10px; height:28px; overflow:hidden;
             margin:20px 0 8px; }}
  .bar {{ background:linear-gradient(90deg,#3b82f6,#22c55e); height:100%;
          width:{bar_w:.2f}%; transition:width .5s; border-radius:10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-top:24px; }}
  .card {{ background:#111827; border:1px solid #1f2937; border-radius:12px; padding:16px; }}
  .k {{ color:#6b7280; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
  .v {{ font-size:24px; font-weight:600; margin-top:4px; }}
  .spark {{ font-size:18px; letter-spacing:1px; color:#3b82f6; margin-top:6px; word-break:break-all; }}
</style></head>
<body><div class="wrap">
  <h1>SL Dataset Generation <span class="badge">{status_text}</span></h1>
  <div class="sub">checkpoints/sl_dataset_v1 · +1M games on top of 500K · Random excluded · 9 workers</div>

  <div class="bar-bg"><div class="bar"></div></div>
  <div style="display:flex;justify-content:space-between;font-size:13px;color:#9ca3af;">
    <span>{d['new_games']:,} new games</span>
    <span><b style="color:#e5e7eb;">{d['pct']:.1f}%</b> of 1,000,000</span>
    <span>{d['remaining']:,} to go</span>
  </div>

  <div class="grid">
    <div class="card"><div class="k">Total shards on disk</div>
      <div class="v">{d['n_shards']:,}</div>
      <div class="sub" style="margin:0">{d['total_games']:,} total games (incl. base 500K)</div></div>
    <div class="card"><div class="k">Current shard</div>
      <div class="v">#{last.get('shard','—')}</div></div>
    <div class="card"><div class="k">Throughput (avg)</div>
      <div class="v">{d['avg_gps']:.1f} <span style="font-size:14px;color:#6b7280">g/s</span></div>
      <div class="spark">{spark}</div></div>
    <div class="card"><div class="k">ETA (remaining)</div>
      <div class="v">{fmt_eta(d['eta_min'])}</div></div>
  </div>
  <div class="sub" style="margin-top:20px;text-align:center;">
    auto-refresh 4s · log idle {int(d['log_age']) if d['log_age'] is not None else '—'}s
  </div>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        try:
            html = render().encode("utf-8")
        except Exception as e:
            html = f"<pre>dashboard error: {e}</pre>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


def main():
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Gen dashboard: http://localhost:{PORT}/  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
