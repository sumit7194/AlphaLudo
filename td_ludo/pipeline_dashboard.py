"""
Tiny HTTP dashboard for run_v10_pipeline.sh progress.

Streams the latest pipeline log file + extracts key metrics into a single page.
Starts on port 8788 (RL dashboard uses 8787 — keep them separate).

Usage:
  python3 pipeline_dashboard.py [--log-dir checkpoints/ac_v10] [--port 8788]
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset='UTF-8'>
<title>V10 Pipeline</title>
<style>
  :root {
    --bg: #0a0a0f; --card: #111117; --border: #222230;
    --text: #e8e8f0; --dim: #8888a0;
    --blue: #3b82f6; --green: #22c55e; --amber: #f59e0b; --red: #ef4444;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 20px; background: var(--bg); color: var(--text);
         font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
  h1 { margin: 0 0 4px 0; font-size: 18px; font-weight: 600; }
  .sub { color: var(--dim); margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 240px 1fr; gap: 20px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 16px; }
  .stage { margin-bottom: 10px; padding: 10px; border-radius: 6px;
           background: #0e0e14; border-left: 3px solid var(--border); }
  .stage.active { border-left-color: var(--amber); background: #1a1512; }
  .stage.done   { border-left-color: var(--green); background: #0e1a10; }
  .stage.failed { border-left-color: var(--red); background: #1a0f10; }
  .stage-name { font-size: 11px; color: var(--dim); text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 4px; }
  .stage-status { font-size: 14px; font-weight: 500; }
  .metric { display: flex; justify-content: space-between; padding: 4px 0;
            border-bottom: 1px dashed var(--border); }
  .metric:last-child { border-bottom: 0; }
  .metric .v { font-weight: 600; color: var(--blue); }
  pre { background: #05050a; border: 1px solid var(--border); border-radius: 6px;
        padding: 12px; margin: 0; overflow-x: auto; max-height: 70vh; overflow-y: auto;
        font-size: 12px; white-space: pre-wrap; word-break: break-all; }
  .log-line-stage { color: var(--amber); font-weight: 600; }
  .log-line-pass { color: var(--green); }
  .log-line-fail { color: var(--red); }
  .log-line-metric { color: var(--blue); }
  .pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           background: var(--amber); margin-right: 6px;
           animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  #updated { color: var(--dim); font-size: 11px; margin-top: 16px; }
</style></head>
<body>
  <h1>V10 Pipeline</h1>
  <div class='sub' id='subtitle'>loading...</div>
  <div class='grid'>
    <div>
      <div class='card'>
        <div id='stages'></div>
      </div>
      <div class='card' style='margin-top:16px'>
        <div style='font-size:11px; color:var(--dim); text-transform:uppercase; margin-bottom:8px'>Metrics</div>
        <div id='metrics'></div>
      </div>
    </div>
    <div class='card'>
      <div style='font-size:11px; color:var(--dim); text-transform:uppercase; margin-bottom:8px'>Log (live tail)</div>
      <pre id='log'>loading...</pre>
    </div>
  </div>
  <div id='updated'></div>

<script>
const STAGES = [
  { id: 'smoke', title: 'Stage 0: Smoke test' },
  { id: 'datagen', title: 'Stage 1: Data generation' },
  { id: 'sl', title: 'Stage 2: SL training' },
  { id: 'eval', title: 'Stage 3: Evaluation' },
];

function colorizeLog(text) {
  return text
    .replace(/^(═+)$/gm, "<span style='color:var(--dim)'>$1</span>")
    .replace(/^(\s*Stage \d+:.*)$/gm, "<span class='log-line-stage'>$1</span>")
    .replace(/^(\s*✓.*)$/gm, "<span class='log-line-pass'>$1</span>")
    .replace(/^(\s*✗.*|.*Error.*|.*Traceback.*)$/gm, "<span class='log-line-fail'>$1</span>")
    .replace(/^(  E\d+\/\d+.*)$/gm, "<span class='log-line-metric'>$1</span>");
}

function parseMetrics(log) {
  const m = {};
  // states/s from data gen
  let rates = [...log.matchAll(/(\d+) states\/s/g)];
  if (rates.length) m['Data gen rate'] = rates[rates.length-1][1] + ' states/s';
  let progress = [...log.matchAll(/(\d[\d,]*)\/(\d[\d,]*) \|/g)];
  if (progress.length) m['Data progress'] = progress[progress.length-1][1].replace(/,/g,'') + '/' + progress[progress.length-1][2].replace(/,/g,'');
  // SL training lines: E{n}/{total} [{s}s] tr: pol=...
  let epochs = [...log.matchAll(/E\s*(\d+)\/\s*(\d+) \[(\d+)s\] tr: pol=([\d.]+) acc=([\d.]+)%/g)];
  if (epochs.length) {
    const e = epochs[epochs.length-1];
    m['Epoch']          = e[1] + '/' + e[2];
    m['Train pol loss'] = e[4];
    m['Train pol acc']  = e[5] + '%';
  }
  let valAcc = [...log.matchAll(/val: pol_acc=([\d.]+)%\s+win_acc=([\d.]+)%\s+mae=([\d.]+)/g)];
  if (valAcc.length) {
    const v = valAcc[valAcc.length-1];
    m['Val pol acc']   = v[1] + '%';
    m['Val win acc']   = v[2] + '%';
    m['Val moves MAE'] = v[3];
  }
  // Eval results
  let wr = log.match(/Win rate:\s+([\d.]+)%/);
  if (wr) m['Eval WR'] = wr[1] + '%';
  let brier = log.match(/Brier score:\s+([\d.]+)/);
  if (brier) m['Brier score'] = brier[1];
  return m;
}

function detectStages(log) {
  const s = { smoke: 'pending', datagen: 'pending', sl: 'pending', eval: 'pending' };
  if (log.includes('Stage 0:')) s.smoke = 'active';
  if (log.includes('ALL CHANNELS VERIFIED')) s.smoke = 'done';
  if (log.includes('Stage 1:')) { s.smoke = 'done'; s.datagen = 'active'; }
  if (log.match(/V10 Data\] Done:/)) s.datagen = 'done';
  if (log.includes('Stage 1: SKIPPED')) s.datagen = 'done';
  if (log.includes('Stage 2:')) { s.datagen = 'done'; s.sl = 'active'; }
  if (log.match(/V10 Train\] Done\./)) s.sl = 'done';
  if (log.includes('Stage 3:')) { s.sl = 'done'; s.eval = 'active'; }
  if (log.includes('V10 Pipeline complete')) s.eval = 'done';
  if (log.match(/Traceback|Error:|exit code [1-9]/)) {
    for (const k of Object.keys(s)) if (s[k] === 'active') s[k] = 'failed';
  }
  return s;
}

async function refresh() {
  try {
    const logResp = await fetch('/log');
    const log = await logResp.text();
    const statusResp = await fetch('/status');
    const status = await statusResp.json();

    // Subtitle
    document.getElementById('subtitle').innerHTML =
      (status.running ? "<span class='pulse'></span>Running: " : "Complete: ") + status.log_file;

    // Stages
    const stages = detectStages(log);
    const html = STAGES.map(st => {
      const cls = stages[st.id];
      const icon = { pending: '○', active: '◉', done: '✓', failed: '✗' }[cls];
      return `<div class='stage ${cls}'>
                <div class='stage-name'>${st.title}</div>
                <div class='stage-status'>${icon} ${cls}</div>
              </div>`;
    }).join('');
    document.getElementById('stages').innerHTML = html;

    // Metrics
    const metrics = parseMetrics(log);
    document.getElementById('metrics').innerHTML = Object.entries(metrics)
      .map(([k, v]) => `<div class='metric'><span>${k}</span><span class='v'>${v}</span></div>`)
      .join('') || "<div style='color:var(--dim)'>waiting for output...</div>";

    // Log (last 200 lines, colorized)
    const lines = log.split('\\n');
    const tail = lines.slice(-200).join('\\n');
    document.getElementById('log').innerHTML = colorizeLog(tail);
    const logEl = document.getElementById('log');
    logEl.scrollTop = logEl.scrollHeight;

    document.getElementById('updated').textContent =
      'updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('updated').textContent = 'fetch error: ' + e.message;
  }
}
refresh();
setInterval(refresh, 2000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    log_dir = None

    def _latest_log(self):
        files = sorted(glob.glob(os.path.join(self.log_dir, 'pipeline_*.log')))
        return files[-1] if files else None

    def _send(self, code, body, ctype='text/plain'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        if isinstance(body, str):
            body = body.encode('utf-8', errors='replace')
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._send(200, HTML, 'text/html; charset=utf-8')
        elif self.path == '/log':
            log = self._latest_log()
            if log and os.path.exists(log):
                with open(log, 'rb') as f:
                    self._send(200, f.read())
            else:
                self._send(200, 'no log yet')
        elif self.path == '/status':
            log = self._latest_log()
            running = False
            if log and os.path.exists(log):
                # "running" = log has no "Pipeline complete" and was modified in last 10 min
                import time as T
                try:
                    with open(log) as f:
                        content = f.read()
                    age = T.time() - os.path.getmtime(log)
                    running = ('Pipeline complete' not in content) and age < 600
                except Exception:
                    pass
            self._send(200, json.dumps({
                'log_file': os.path.basename(log) if log else None,
                'running': running,
            }), 'application/json')
        else:
            self._send(404, 'not found')

    def log_message(self, *args):
        pass  # silence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='checkpoints/ac_v10')
    ap.add_argument('--port', type=int, default=8788)
    args = ap.parse_args()

    Handler.log_dir = args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)
    server = HTTPServer(('0.0.0.0', args.port), Handler)
    print(f"[Dashboard] http://localhost:{args.port}  (log-dir: {args.log_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
