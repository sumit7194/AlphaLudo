"""Tiny local HTML dashboard for the 3 SL trainers.

Reads each model's `train.log` and renders a single-page comparison view
that auto-refreshes every 5 seconds.

Usage:
    /Users/sumit/Github/AlphaLudo/td_ludo/td_env/bin/python sl_dashboard.py
    open http://localhost:8810/

What it shows per model:
  - latest step / total steps
  - last train loss / acc
  - last val_acc / val_loss (from save_every eval, if any)
  - latest samples/sec
  - 60-min rolling chart of loss + acc
  - status: running / stopped (based on log mtime + pgrep)

No external deps beyond stdlib. Just point a browser at the URL.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
PORT = 8810

# (model_label, train.log path, expected model script name pattern for pgrep)
MODELS = [
    # Active run: V13.6 epoch-2 on the full 1.5M dataset (warm-started).
    ("V13.6 ep2 (1.5M)", HERE / "checkpoints" / "v136_sl_ep2" / "train.log", "train_v136_sl.py"),
    # Epoch-1 / sibling runs (kept for reference; stopped).
    ("V15.2",  HERE / "checkpoints" / "v152_sl" / "train.log", "train_v152_sl.py"),
    ("V13.6 ep1", HERE / "checkpoints" / "v136_sl" / "train.log", "train_v136_sl.py"),
    ("V12.3",  HERE / "checkpoints" / "v123_sl" / "train.log", "train_v123_sl.py"),
]

# Compile regexes once.
_STEP_RE = re.compile(
    r"\[(\d\d:\d\d:\d\d)\] step\s+(\d+)\s*\|\s*"
    r"loss\s+(-?[\d.]+)\s+acc\s+([\d.]+)%\s+lr\s+([\d.eE+-]+)\s*"
    r"\(([\d.]+)\s+samples/sec\)"
)
_VAL_RE = re.compile(
    r"\[eval @ step (\d+)\]\s+val_acc=([\d.]+)%\s+val_loss=([\d.]+)"
)
_TOTAL_STEPS_RE = re.compile(r"total_steps=([\d,]+)")
_FINAL_RE = re.compile(r"FINAL\s+val_acc=([\d.]+)%\s+val_loss=([\d.]+)")
_PARAMS_RE = re.compile(r"model:\s+([\d,]+)\s+params")


def is_running(pattern: str) -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, timeout=2,
        )
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def parse_log(path: Path) -> dict:
    """Extract latest stats from a single train.log."""
    if not path.exists():
        return {"status": "no log yet", "rows": []}
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return {"status": "log unreadable", "rows": []}

    total_steps = None
    m = _TOTAL_STEPS_RE.search(text)
    if m:
        total_steps = int(m.group(1).replace(",", ""))

    params = None
    m = _PARAMS_RE.search(text)
    if m:
        params = int(m.group(1).replace(",", ""))

    final_val = None
    m = _FINAL_RE.search(text)
    if m:
        final_val = {"val_acc": float(m.group(1)), "val_loss": float(m.group(2))}

    # All step lines (for chart). Cap at last 200 points for speed.
    rows = []
    for m in _STEP_RE.finditer(text):
        rows.append({
            "ts": m.group(1),
            "step": int(m.group(2)),
            "loss": float(m.group(3)),
            "acc": float(m.group(4)),
            "lr": float(m.group(5)),
            "sps": float(m.group(6)),
        })
    rows = rows[-200:]

    vals = []
    for m in _VAL_RE.finditer(text):
        vals.append({
            "step": int(m.group(1)),
            "val_acc": float(m.group(2)),
            "val_loss": float(m.group(3)),
        })

    return {
        "total_steps": total_steps,
        "params": params,
        "final": final_val,
        "rows": rows,
        "vals": vals,
        "last_mtime": path.stat().st_mtime,
    }


def gather():
    out = []
    now = time.time()
    for label, log_path, pattern in MODELS:
        d = parse_log(log_path)
        running = is_running(pattern)
        if d["rows"]:
            last = d["rows"][-1]
            stale_sec = now - d.get("last_mtime", 0)
        else:
            last = None
            stale_sec = None
        out.append({
            "label": label,
            "running": running,
            "total_steps": d.get("total_steps"),
            "params": d.get("params"),
            "final": d.get("final"),
            "last": last,
            "stale_sec": stale_sec,
            "rows": d["rows"],
            "vals": d.get("vals", []),
            "log_path": str(log_path),
        })
    return out


HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>SL Training — 3-way</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
       background: #0e1320; color: #d0d8e8; margin: 0; padding: 20px; }
h1 { margin: 0 0 8px; font-size: 18px; color: #8b5cf6; }
.subtitle { font-size: 12px; color: #6b7591; margin-bottom: 24px; }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.card { background: #161c2e; border-radius: 8px; padding: 16px; }
.card h2 { margin: 0 0 6px; font-size: 14px; color: #c0c7d6; }
.status-pill { display: inline-block; padding: 2px 8px; border-radius: 4px;
               font-size: 11px; margin-left: 8px; }
.status-running { background: rgba(52, 211, 153, 0.18); color: #34d399; }
.status-stopped { background: rgba(248, 113, 113, 0.18); color: #f87171; }
.status-stale   { background: rgba(251, 191, 36, 0.18); color: #fbbf24; }
.meta-row { font-size: 11px; color: #6b7591; margin: 4px 0; }
.big { font-size: 22px; font-weight: 600; color: #e6e9f2; }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
               margin: 12px 0; }
.metric-cell { background: #1f2638; border-radius: 4px; padding: 8px; }
.metric-label { font-size: 10px; color: #6b7591; text-transform: uppercase; }
.metric-val { font-size: 16px; color: #e6e9f2; font-weight: 600; margin-top: 2px; }
.progress-bar { background: #1f2638; height: 6px; border-radius: 3px; margin-top: 6px; overflow: hidden; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%); }
.chart-wrap { height: 160px; margin-top: 8px; }
.footer { font-size: 10px; color: #4a5268; margin-top: 16px; text-align: center; }
.log-path { font-size: 10px; color: #4a5268; font-family: monospace; word-break: break-all; }
.val-row { font-size: 11px; color: #f59e0b; margin-top: 4px; }
.final { color: #34d399; font-weight: 600; }
</style>
</head><body>

<h1>SL Training — V15.2 / V13.6 / V12.3</h1>
<div class="subtitle">1 epoch over 37.4M rows (bot-vs-bot, Random-filtered). MPS. Auto-refresh 5s.</div>

<div class="grid">__CARDS__</div>

<div class="footer">refreshed at <span id="ts"></span></div>

<script>
document.getElementById("ts").textContent = new Date().toLocaleTimeString();
const _data = __DATA__;
_data.forEach((m, i) => {
  const canvas = document.getElementById(`chart-${i}`);
  if (!canvas) return;
  new Chart(canvas, {
    type: 'line',
    data: {
      labels: m.rows.map(r => r.step),
      datasets: [
        { label: 'loss',      data: m.rows.map(r => r.loss),
          borderColor: '#f87171', backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, yAxisID: 'y' },
        { label: 'acc (%)',   data: m.rows.map(r => r.acc),
          borderColor: '#34d399', backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { labels: { color: '#9aa3bd', font: {size: 10} } } },
      scales: {
        x: { ticks: { color: '#4a5268', font: {size: 9} } },
        y: { position: 'left', ticks: { color: '#f87171', font: {size: 9} }, title: { display: false } },
        y1:{ position: 'right', min: 0, max: 100, ticks: { color: '#34d399', font: {size: 9} }, grid: { display: false } },
      },
    }
  });
});
</script>
</body></html>"""


def render_card(model: dict, idx: int) -> str:
    label = model["label"]
    running = model["running"]
    if running:
        status_html = '<span class="status-pill status-running">running</span>'
    elif model.get("final"):
        status_html = '<span class="status-pill status-running">done</span>'
    else:
        # Check staleness
        stale = model.get("stale_sec") or 0
        if stale and stale > 60:
            status_html = '<span class="status-pill status-stale">stale</span>'
        else:
            status_html = '<span class="status-pill status-stopped">stopped</span>'
    last = model["last"]
    total_steps = model.get("total_steps") or 0
    if last:
        step = last["step"]
        loss = last["loss"]
        acc = last["acc"]
        sps = last["sps"]
        progress_pct = 100.0 * step / max(1, total_steps) if total_steps else 0
        progress = (
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{progress_pct:.1f}%"></div>'
            f'</div>'
            f'<div class="meta-row">step {step:,} / {total_steps:,} '
            f'({progress_pct:.1f}%)  ·  {sps:.0f} samples/sec</div>'
        )
        metrics = (
            f'<div class="metric-grid">'
            f'<div class="metric-cell"><div class="metric-label">loss</div>'
            f'<div class="metric-val">{loss:.4f}</div></div>'
            f'<div class="metric-cell"><div class="metric-label">train acc</div>'
            f'<div class="metric-val">{acc:.1f}%</div></div>'
            f'</div>'
        )
    else:
        progress = '<div class="meta-row">no step log yet…</div>'
        metrics = ''
    vals = model.get("vals", [])
    val_html = ''
    if vals:
        v = vals[-1]
        val_html = (
            f'<div class="val-row">latest val (step {v["step"]:,}): '
            f'acc <b>{v["val_acc"]:.2f}%</b>  ·  loss {v["val_loss"]:.4f}</div>'
        )
    final = model.get("final")
    final_html = ''
    if final:
        final_html = (
            f'<div class="val-row final">FINAL val_acc {final["val_acc"]:.2f}%  '
            f'val_loss {final["val_loss"]:.4f}</div>'
        )
    params = model.get("params")
    params_str = f"{params:,} params" if params else "—"
    return (
        f'<div class="card">'
        f'<h2>{label} {status_html}</h2>'
        f'<div class="meta-row">{params_str}</div>'
        f'{metrics}'
        f'{progress}'
        f'{val_html}'
        f'{final_html}'
        f'<div class="chart-wrap"><canvas id="chart-{idx}"></canvas></div>'
        f'<div class="log-path">{model["log_path"]}</div>'
        f'</div>'
    )


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a, **kw):
        return

    def do_GET(self):
        if self.path == "/api/status":
            data = gather()
            payload = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        # Serve dashboard
        data = gather()
        cards = "".join(render_card(m, i) for i, m in enumerate(data))
        # Compact rows for JS chart payload (just step/loss/acc to keep page small)
        js_data = json.dumps([{
            "label": m["label"],
            "rows": [{"step": r["step"], "loss": r["loss"], "acc": r["acc"]}
                     for r in m["rows"]],
        } for m in data])
        html = HTML_TEMPLATE.replace("__CARDS__", cards).replace("__DATA__", js_data)
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"SL dashboard: http://localhost:{PORT}/  (auto-refresh 5s, Ctrl-C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
