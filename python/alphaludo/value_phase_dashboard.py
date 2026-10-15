"""Live dashboard for the value phase (self-play -> value training -> test).

A standalone, read-only process: it parses the logs / metrics files the jobs
already write, so it never needs the jobs restarted and survives them. Serves
http://localhost:8790/ (page) and /api/status (JSON, recomputed per request).

  nohup td_ludo/td_env/bin/python python/alphaludo/value_phase_dashboard.py &
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# Round configuration (set from --round in main()).
ROUNDS = {
    1: {"sp": "data/selfplay_v152", "val": "checkpoints/value_v1", "chain": "chain_value_phase.sh",
        "title": "Value phase — champion self-play → value net → 1-ply test",
        "tests": [("test.log", "1-ply by new value")]},
    2: {"sp": "data/selfplay_v152_b", "val": "checkpoints/value_v2", "chain": "chain_value_round2.sh",
        "title": "Value round 2 — +200k games → value net on 400k games → tests",
        "tests": [("test_1ply.log", "1-ply by new value"),
                  ("test_combined.log", "policy-guided search, 400 evals")]},
}
CFG = ROUNDS[1]
SP_DIR = ROOT / CFG["sp"]
VAL_DIR = ROOT / CFG["val"]
TARGET_GAMES = 200_000
HISTORY = [("champion value, 1-ply", 0.350), ("value v1, 1-ply", 0.430), ("old value, search 400 by visits", 0.480)]


def alive(pattern: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
    except Exception:
        return False


def read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


_GAMES_CACHE: dict = {}


def shard_games(path: Path) -> int:
    """Complete games in a shard (unique game ids), cached: shards are immutable."""
    key = (path.name, path.stat().st_size)
    if key not in _GAMES_CACHE:
        import numpy as np
        try:
            with open(path, "rb") as fh:
                n = int.from_bytes(fh.read(16)[8:12], "little")
            gid = np.memmap(path, dtype=np.uint32, mode="r", offset=16, shape=(n, 8))[:, 7]
            _GAMES_CACHE[key] = int(len(np.unique(gid))) if n else 0
        except Exception:
            return 5000
    return _GAMES_CACHE[key]


def selfplay_status():
    logs = read(SP_DIR / "console.log") + read(SP_DIR / "console_gpu.log") + read(SP_DIR / "console_cpu.log")
    files = sorted(SP_DIR.glob("shard_*.bin"))
    info = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                recs = int.from_bytes(fh.read(16)[8:12], "little")
            info.append((f.stat().st_mtime, shard_games(f), recs))
        except OSError:
            pass
    info.sort()
    games = sum(g for _, g, _ in info)
    records = sum(r for _, _, r in info)
    # Rate = games written in the last 30 minutes (all workers together).
    now = time.time()
    recent = [(t, g) for t, g, _ in info if now - t < 1800]
    gpm = 0.0
    if len(recent) >= 2:
        span = recent[-1][0] - recent[0][0]
        gpm = sum(g for _, g in recent[1:]) / (span / 60) if span > 0 else 0.0
    # Rolling rate: games written in the trailing 15 minutes at each shard
    # write. (Dividing one shard by the gap to the previous write spiked wildly
    # with two workers whose writes can land seconds or minutes apart.)
    W = 900.0
    per_shard = []
    for i in range(len(info)):
        t_i = info[i][0]
        win = [(t, g) for t, g, _ in info[: i + 1] if t_i - t <= W]
        if len(win) < 2:
            continue
        span = t_i - win[0][0]
        if span < 300:          # need >= 5 min of history for a stable rate
            continue
        per_shard.append({"shard": i, "gpm": round(sum(g for _, g in win[1:]) / (span / 60))})
    workers = {"gpu": alive("gen_selfplay.py.*--backend mlx"), "cpu": alive("gen_selfplay.py.*--device cpu")}
    running = alive("gen_selfplay.py")
    return {
        "running": running, "workers": workers, "games": games, "target": TARGET_GAMES, "records": records,
        "gpm": gpm, "eta_min": (max(0, TARGET_GAMES - games) / gpm) if gpm and running else None,
        "per_shard": per_shard[-80:], "done": len(files) >= TARGET_GAMES // 5000, "errors": "Traceback" in logs,
    }


def training_status():
    log = read(VAL_DIR / "train.log")
    out = {"running": alive("train_value.py"), "started": bool(log), "errors": "Traceback" in log}
    m = re.search(r"\[data\] (\d+) train shards, ([\d,]+) games, ([\d,]+) records; ([\d,]+) samples/epoch, ([\d,]+) steps", log)
    if m:
        out["total_steps"] = int(m.group(5).replace(",", ""))
    steps = [{"step": int(a), "epoch": int(b), "bce": float(d), "sps": float(e.replace(",", ""))}
             for a, b, c, d, e in re.findall(
                 r"step\s+(\d+)/\d+ \| epoch (\d+) \| lr ([\d.e+-]+) \| train BCE ([\d.]+) \| ([\d,]+) samples/s", log)]
    out["steps"] = steps[-400:]
    vals = []
    for line in read(VAL_DIR / "metrics.jsonl").splitlines():
        try:
            vals.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    out["val"] = vals
    if steps:
        last = steps[-1]
        out["step"] = last["step"]
        tot = out.get("total_steps")
        if tot and last["sps"] > 0 and out["running"]:
            out["eta_min"] = (tot - last["step"]) * 512 / last["sps"] / 60
    out["done"] = "[exit] clean" in log
    return out


def _one_test(fname, label):
    log = read(VAL_DIR / fname)
    out = {"label": label, "started": bool(log), "errors": "Traceback" in log, "target_games": 300}
    prog = re.findall(r"^\s+(\d+) games \| ([\d,]+) evals \(([\d,]+)/s", log, re.M)
    if prog:
        g, e, r = prog[-1]
        out.update(games=int(g), evals_per_s=int(r.replace(",", "")))
    m = re.search(r"A win rate ([\d.]+) ±([\d.]+) \((\d+)/(\d+)\)", log)
    if m:
        out.update(win_rate=float(m.group(1)), stderr=float(m.group(2)), wins=int(m.group(3)), total=int(m.group(4)))
    return out


def test_status():
    tests = [_one_test(f, l) for f, l in CFG["tests"]]
    started = [t for t in tests if t["started"]]
    cur = next((t for t in started if t.get("win_rate") is None), started[-1] if started else tests[0])
    out = dict(cur)
    out["running"] = alive("net_farm.py h2h")
    out["all_done"] = all(t.get("win_rate") is not None for t in tests)
    out["tests"] = tests
    return out


def status():
    chain = read(VAL_DIR / "chain.log").strip().splitlines()
    sp, tr, te = selfplay_status(), training_status(), test_status()
    stage = ("complete" if te["all_done"] else "test" if te["started"]
             else "training" if tr["started"] else "selfplay")
    # Baseline for the validation chart = this run's own step-0 measurement.
    ll0 = auc0 = None
    if tr["val"]:
        ll0, auc0 = tr["val"][0].get("val_logloss"), tr["val"][0].get("val_auc")
    sysinfo = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        sysinfo = {"cpu": psutil.cpu_percent(interval=None), "mem": vm.percent,
                   "mem_avail_gb": vm.available / 2**30, "swap": psutil.swap_memory().percent}
    except Exception:
        pass
    return {"time": time.time(), "stage": stage, "chain_alive": alive(CFG["chain"]), "title": CFG["title"],
            "chain_log": chain[-12:], "selfplay": sp, "training": tr, "test": te, "system": sysinfo,
            "history": HISTORY, "baselines": {"target": 0.5, "start_ll": ll0, "start_auc": auc0}}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Value Phase</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#2a313c;--text:#e6edf3;--mut:#8b949e;--acc:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0}.sub{color:var(--mut);font-size:13px;margin-top:2px}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.pipe{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stage.active{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}.stage.done{border-color:#238636}
.stage h3{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.big{font-size:22px;font-weight:600;display:flex;align-items:center}.small{font-size:12px}.mut{color:var(--mut)}
.bar{height:8px;background:#21262d;border-radius:5px;overflow:hidden;margin:8px 0 4px}.bar>i{display:block;height:100%;background:var(--acc);width:0}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;background:var(--mut);flex:none}
.dot.on{background:var(--good);box-shadow:0 0 6px var(--good)}.dot.err{background:var(--bad)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h2{font-size:14px;margin:0 0 8px}.chart{position:relative;height:230px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:13px}.kv b{font-weight:500;color:var(--mut)}
pre{margin:0;font:12px/1.5 ui-monospace,Menlo,monospace;color:#c9d1d9;white-space:pre-wrap;word-break:break-word;max-height:220px;overflow:auto}
.verdict{font-size:30px;font-weight:700}
@media(max-width:820px){.pipe,.grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="top"><div><h1 id="title">Value phase</h1>
<div class="sub">Win rate vs the v15.2 champion · 50% = parity · results so far: <span id="hist"></span></div></div>
<div class="small mut"><span class="dot" id="chain-dot"></span><span id="upd">connecting…</span></div></div>
<div class="pipe">
 <div class="stage" id="st-sp"><h3>1 · Self-play</h3><div class="big"><span class="dot" id="sp-dot"></span><span id="sp-big">–</span></div><div class="bar"><i id="sp-bar"></i></div><div class="small mut" id="sp-sub"></div></div>
 <div class="stage" id="st-tr"><h3>2 · Value training</h3><div class="big"><span class="dot" id="tr-dot"></span><span id="tr-big">waiting for self-play</span></div><div class="bar"><i id="tr-bar"></i></div><div class="small mut" id="tr-sub"></div></div>
 <div class="stage" id="st-te"><h3>3 · Tests vs champion</h3><div class="big"><span class="dot" id="te-dot"></span><span id="te-big">waiting for training</span></div><div class="bar"><i id="te-bar"></i></div><div class="small mut" id="te-sub"></div></div>
</div>
<div class="grid">
 <div class="card"><h2>Self-play speed (games/min, all workers, rolling 15-min window)</h2><div class="chart"><canvas id="c-sp"></canvas></div></div>
 <div class="card"><h2>Value training — train BCE</h2><div class="chart"><canvas id="c-tr"></canvas></div></div>
 <div class="card"><h2>Validation on held-out games — dashed = start of this run</h2><div class="chart"><canvas id="c-val"></canvas></div></div>
 <div class="card"><h2>Test results vs champion</h2><div id="tests" style="margin-bottom:12px"></div>
   <div class="kv"><b>CPU</b><span id="sys-cpu">–</span><b>Memory</b><span id="sys-mem">–</span><b>Swap</b><span id="sys-swap">–</span></div></div>
 <div class="card" style="grid-column:1/-1"><h2>Pipeline log</h2><pre id="chain"></pre></div>
</div></div>
<script>
const css=getComputedStyle(document.documentElement),C=k=>css.getPropertyValue(k).trim();
Chart.defaults.color=C('--mut');Chart.defaults.borderColor=C('--line');Chart.defaults.font.size=11;
const lineOpts=()=>({animation:false,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{boxWidth:10}}}});
const spC=new Chart(document.getElementById('c-sp'),{type:'line',data:{labels:[],datasets:[{label:'games/min',data:[],borderColor:C('--acc'),pointRadius:0,tension:.2}]},options:lineOpts()});
const trC=new Chart(document.getElementById('c-tr'),{type:'line',data:{labels:[],datasets:[{label:'train BCE',data:[],borderColor:C('--acc'),pointRadius:0,tension:.2}]},options:lineOpts()});
const vo=lineOpts();vo.scales={y:{position:'left',title:{display:true,text:'log-loss'}},y2:{position:'right',grid:{drawOnChartArea:false},title:{display:true,text:'AUC'}}};
const valC=new Chart(document.getElementById('c-val'),{type:'line',data:{labels:[],datasets:[
 {label:'val log-loss',data:[],borderColor:C('--acc'),yAxisID:'y',tension:.2},
 {label:'start log-loss',data:[],borderColor:C('--warn'),borderDash:[5,4],pointRadius:0,yAxisID:'y'},
 {label:'val AUC',data:[],borderColor:C('--good'),yAxisID:'y2',tension:.2},
 {label:'start AUC',data:[],borderColor:'#7ee787',borderDash:[5,4],pointRadius:0,yAxisID:'y2'}]},options:vo});
const $=id=>document.getElementById(id),fmt=n=>n==null?'–':Number(n).toLocaleString(),
 eta=m=>m==null?'':' · '+(m>90?(m/60).toFixed(1)+' h':Math.round(m)+' min')+' left',
 txt=(id,t)=>{$(id).textContent=t},
 setDot=(id,on,err)=>{const d=$(id);d.classList.toggle('on',!!on&&!err);d.classList.toggle('err',!!err)},
 stageCls=(id,active,done)=>{$(id).classList.toggle('active',active);$(id).classList.toggle('done',done)};
async function tick(){
 let s;try{s=await (await fetch('/api/status',{cache:'no-store'})).json()}catch(e){txt('upd','dashboard server unreachable');setDot('chain-dot',false,true);return}
 setDot('chain-dot',s.chain_alive,false);
 txt('upd',`pipeline ${s.chain_alive?'running':'not running'} · updated ${new Date(s.time*1000).toLocaleTimeString()}`);
 const sp=s.selfplay,tr=s.training,te=s.test;
 stageCls('st-sp',s.stage==='selfplay',sp.done);stageCls('st-tr',s.stage==='training',!!tr.done);stageCls('st-te',s.stage==='test',!!te.all_done);
 setDot('sp-dot',sp.running,sp.errors);txt('sp-big',`${fmt(sp.games)} / ${fmt(sp.target)} games`);
 $('sp-bar').style.width=(100*sp.games/sp.target)+'%';
 const wk=sp.workers||{};
 txt('sp-sub',`${fmt(sp.records)} positions · ${sp.gpm?Math.round(sp.gpm):'–'} games/min · workers: GPU ${wk.gpu?'on':'off'}, CPU ${wk.cpu?'on':'off'}${eta(sp.eta_min)}${sp.errors?' · ERROR in log':''}`);
 spC.data.labels=sp.per_shard.map(x=>x.shard);spC.data.datasets[0].data=sp.per_shard.map(x=>x.gpm);spC.update();
 if(tr.started){const tot=tr.total_steps||0,st=tr.step||0,last=tr.steps.length?tr.steps[tr.steps.length-1]:null;
  setDot('tr-dot',tr.running,tr.errors);txt('tr-big',`step ${fmt(st)} / ${fmt(tot)}`);$('tr-bar').style.width=(tot?100*st/tot:0)+'%';
  txt('tr-sub',last?`epoch ${last.epoch} · BCE ${last.bce.toFixed(4)} · ${fmt(Math.round(last.sps))} samples/s${eta(tr.eta_min)}${tr.errors?' · ERROR in log':''}`:'loading data…');}
 trC.data.labels=tr.steps.map(x=>x.step);trC.data.datasets[0].data=tr.steps.map(x=>x.bce);trC.update();
 valC.data.labels=tr.val.map(v=>v.step);valC.data.datasets[0].data=tr.val.map(v=>v.val_logloss);
 valC.data.datasets[1].data=tr.val.map(()=>s.baselines.start_ll);valC.data.datasets[2].data=tr.val.map(v=>v.val_auc);
 valC.data.datasets[3].data=tr.val.map(()=>s.baselines.start_auc);valC.update();
 if(te.started){setDot('te-dot',te.running,te.errors);txt('te-big',`${fmt(te.games||0)} / ${te.target_games} games`);
  $('te-bar').style.width=(100*(te.games||0)/te.target_games)+'%';txt('te-sub',`${te.label} · `+(te.evals_per_s?`${fmt(te.evals_per_s)} evals/s`:'starting…'))}
 txt('title',s.title);txt('hist',(s.history||[]).map(h=>`${h[0]} ${(100*h[1]).toFixed(1)}%`).join(' · '));
 const box=$('tests');box.replaceChildren();
 for(const t of te.tests||[]){const row=document.createElement('div');row.style.margin='0 0 10px';
  const big=document.createElement('div');big.className='verdict';const lab=document.createElement('div');lab.className='small mut';
  if(t.win_rate!=null){const ok=t.win_rate-2*t.stderr>0.5,par=t.win_rate+2*t.stderr>=0.5;
   big.textContent=(100*t.win_rate).toFixed(1)+'% ± '+(100*t.stderr).toFixed(1);
   big.style.color=ok?C('--good'):par?C('--warn'):C('--bad');
   lab.textContent=`${t.label} · ${t.wins}/${t.total} · `+(ok?'BEATS the champion':par?'within noise of parity':'below the champion');}
  else{big.textContent=t.started?`${t.games||0} / ${t.target_games} games`:'not started';big.classList.add('mut');big.style.fontSize='20px';lab.textContent=t.label;}
  row.append(big,lab);box.append(row);}
 const sy=s.system||{};txt('sys-cpu',sy.cpu!=null?sy.cpu.toFixed(0)+'%':'–');
 txt('sys-mem',sy.mem!=null?`${sy.mem.toFixed(0)}% used · ${sy.mem_avail_gb.toFixed(1)} GB free`:'–');txt('sys-swap',sy.swap!=null?sy.swap.toFixed(0)+'%':'–');
 txt('chain',(s.chain_log||[]).join('\n')||'(no pipeline log yet)');
}
tick();setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path.startswith("/api/status"):
                self._send(200, json.dumps(status()).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--round", type=int, default=2, choices=sorted(ROUNDS))
    a = ap.parse_args()
    global CFG, SP_DIR, VAL_DIR
    CFG = ROUNDS[a.round]
    SP_DIR, VAL_DIR = ROOT / CFG["sp"], ROOT / CFG["val"]
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"value-phase dashboard on http://localhost:{a.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
