"""Live dashboard for a V17 run (SL stage and RL stage) — read-only, standalone.

Parses the run directory's status / metrics files, so it never touches the
training processes. Serves http://localhost:<port>/ and /api/status.

  python -m v17.dashboard --run-dir checkpoints/v17 --port 8790
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUN = Path("checkpoints/v17")
CHAIN = "run_v17.sh"
TITLE = "V17 — unified 2P/4P network, trained from scratch"


def alive(pattern: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
    except Exception:
        return False


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_jsonl(p: Path, limit: int = 5000):
    out = []
    try:
        for line in p.read_text().splitlines()[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return out


def tail(p: Path, n: int = 14):
    try:
        lines = [l for l in p.read_text(errors="ignore").splitlines() if "TD-Ludo Config" not in l]
        return lines[-n:]
    except OSError:
        return []


def status():
    sl_s = read_json(RUN / "sl_status.json") or {}
    rl_s = read_json(RUN / "rl_status.json") or {}
    sl_checks = read_jsonl(RUN / "sl_metrics.jsonl")
    rl_all = read_jsonl(RUN / "rl_metrics.jsonl")
    rl_updates = [r for r in rl_all if r.get("kind") == "update"][-600:]
    rl_evals = [r for r in rl_all if r.get("kind") == "eval"]
    sysinfo = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        sysinfo = {"cpu": psutil.cpu_percent(interval=None), "mem": vm.percent,
                   "mem_avail_gb": vm.available / 2**30, "swap": psutil.swap_memory().percent}
    except Exception:
        pass
    phase = "rl" if rl_s else ("sl" if sl_s else "not started")
    mixed = any("val4_ce" in c for c in sl_checks) or "rung4" in rl_s
    rd = f"--run-dir {RUN} "     # per-run process match (checkpoints/v17 vs checkpoints/v17_4p)
    return {
        "time": time.time(), "phase": phase, "mixed": mixed, "title": TITLE,
        "procs": {"chain": alive("bash ./" + CHAIN), "sl": alive("v17.sl_train " + rd), "rl": alive("v17.rl_train " + rd)},
        "sl": {"status": sl_s, "checks": sl_checks},
        "rl": {"status": rl_s, "updates": rl_updates, "evals": rl_evals},
        "log": tail(RUN / "console.log"), "system": sysinfo,
    }


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V17 Training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#2a313c;--text:#e6edf3;--mut:#8b949e;--acc:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149;--vio:#bc8cff;--org:#ff9e64}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:1220px;margin:0 auto;padding:20px 16px 40px}h1{font-size:20px;margin:0}.sub{color:var(--mut);font-size:13px;margin-top:2px}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.tile h3{margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.big{font-size:21px;font-weight:600;display:flex;align-items:center}.small{font-size:12px}.mut{color:var(--mut)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;background:var(--mut);flex:none}
.dot.on{background:var(--good);box-shadow:0 0 6px var(--good)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h2{font-size:14px;margin:0 0 8px}.chart{position:relative;height:230px}
pre{margin:0;font:12px/1.5 ui-monospace,Menlo,monospace;color:#c9d1d9;white-space:pre-wrap;word-break:break-word;max-height:240px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--mut);font-weight:500}
.section{margin:18px 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut)}
@media(max-width:860px){.tiles,.grid{grid-template-columns:1fr 1fr}}@media(max-width:560px){.tiles,.grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="top"><div><h1 id="title">V17</h1>
<div class="sub">Stage 1: SL from the fixed MCTS-3000 teacher (auto-stop) → Stage 2: infinite RL, win/loss reward only</div></div>
<div class="small mut"><span class="dot" id="d-chain"></span><span id="upd">connecting…</span></div></div>
<div class="tiles">
 <div class="tile"><h3>Stage</h3><div class="big" id="t-phase">–</div><div class="small mut" id="t-phase2"></div></div>
 <div class="tile"><h3>Progress</h3><div class="big" id="t-prog">–</div><div class="small mut" id="t-prog2"></div></div>
 <div class="tile"><h3>Speed</h3><div class="big" id="t-speed">–</div><div class="small mut" id="t-speed2"></div></div>
 <div class="tile"><h3>Mac</h3><div class="big" id="t-sys">–</div><div class="small mut" id="t-sys2"></div></div>
</div>
<div class="section">Stage 1 · supervised learning from MCTS-3000</div>
<div class="grid">
 <div class="card"><h2>Held-out match to the teacher (cross-entropy ↓, top-1 ↑)</h2><div class="chart"><canvas id="c-slv"></canvas></div></div>
 <div class="card"><h2>Strength checks (greedy, no search)</h2><div class="chart"><canvas id="c-sls"></canvas></div></div>
 <div class="card" style="grid-column:1/-1"><h2>Auto-stop decisions</h2><div id="sl-table" class="small mut">no checks yet</div></div>
</div>
<div class="m4" style="display:none"><div class="section">Stage 1 · 4-player (held-out 4P teacher positions; strength = one model seat vs 3 copies, fair 25%)</div>
<div class="grid">
 <div class="card"><h2>4P held-out match to the teacher</h2><div class="chart"><canvas id="c-slv4"></canvas></div></div>
 <div class="card"><h2>4P strength checks (greedy)</h2><div class="chart"><canvas id="c-sls4"></canvas></div></div>
</div></div>
<div class="section">Stage 2 · reinforcement learning</div>
<div class="grid">
 <div class="card"><h2>Evaluation vs fixed references (win %, ±2σ)</h2><div class="chart"><canvas id="c-ev"></canvas></div></div>
 <div class="card"><h2>PPO health</h2><div class="chart"><canvas id="c-ppo"></canvas></div></div>
 <div class="card"><h2>Losses</h2><div class="chart"><canvas id="c-loss"></canvas></div></div>
 <div class="card"><h2>Ladder & league</h2><div id="ladder" class="small"></div></div>
</div>
<div class="m4" style="display:none"><div class="section">Stage 2 · 4-player evaluation (one model seat vs 3 copies, fair 25%)</div>
<div class="grid">
 <div class="card"><h2>4P evaluation (win %)</h2><div class="chart"><canvas id="c-ev4"></canvas></div></div>
 <div class="card"><h2>4P share of training decisions & 4P ladder</h2><div class="chart"><canvas id="c-mix"></canvas></div></div>
</div></div>
<div class="section">Run log</div><div class="card"><pre id="log"></pre></div>
</div>
<script>
const css=getComputedStyle(document.documentElement),C=k=>css.getPropertyValue(k).trim();
Chart.defaults.color=C('--mut');Chart.defaults.borderColor=C('--line');Chart.defaults.font.size=11;
const base=()=>({animation:false,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{boxWidth:10}}}});
const mk=(id,sets,extra)=>new Chart(document.getElementById(id),{type:'line',data:{labels:[],datasets:sets},options:Object.assign(base(),extra||{})});
const ds=(label,color,axis,dash)=>({label,data:[],borderColor:color,backgroundColor:color,yAxisID:axis||'y',tension:.2,pointRadius:2,borderDash:dash||[]});
const slv=mk('c-slv',[ds('val CE',C('--acc'),'y'),ds('val top-1',C('--good'),'y2'),ds('value CE',C('--org'),'y')],{scales:{y:{position:'left'},y2:{position:'right',grid:{drawOnChartArea:false}}}});
const sls=mk('c-sls',[ds('vs MCTS-3000',C('--acc')),ds('vs v15.2',C('--vio')),ds('50% (parity)',C('--mut'),'y',[5,4])],{scales:{y:{min:0,max:1}}});
const OPP=[['v15.2',C('--vio')],['mcts3000',C('--acc')],['depth2',C('--org')],['expert',C('--good')],['heuristic',C('--warn')]];
const ev=mk('c-ev',OPP.map(([o,c])=>ds(o,c)).concat([ds('50%',C('--mut'),'y',[5,4])]),{scales:{y:{min:0,max:1}}});
const OPP4=[['4p:teacher',C('--acc')],['4p:depth2',C('--org')],['4p:expectimax',C('--vio')],['4p:heuristic',C('--warn')]];
const slv4=mk('c-slv4',[ds('4P val CE',C('--acc'),'y'),ds('4P val top-1',C('--good'),'y2'),ds('4P value CE',C('--org'),'y')],{scales:{y:{position:'left'},y2:{position:'right',grid:{drawOnChartArea:false}}}});
const sls4=mk('c-sls4',[ds('vs 3x teacher',C('--acc')),ds('vs 3x expectimax',C('--vio')),ds('25% (parity)',C('--mut'),'y',[5,4])],{scales:{y:{min:0,max:1}}});
const ev4=mk('c-ev4',OPP4.map(([o,c])=>ds(o,c)).concat([ds('25%',C('--mut'),'y',[5,4])]),{scales:{y:{min:0,max:1}}});
const mix=mk('c-mix',[ds('4P share of decisions',C('--acc'),'y'),ds('4P ladder rung (0-3)',C('--good'),'y2')],{scales:{y:{min:0,max:1},y2:{position:'right',min:0,max:3,grid:{drawOnChartArea:false}}}});
const RUNG4=['4p:heuristic','4p:expectimax','4p:depth2','4p:teacher'];
const ppo=mk('c-ppo',[ds('KL',C('--acc'),'y'),ds('clip frac',C('--org'),'y'),ds('entropy (norm)',C('--good'),'y2')],{scales:{y:{position:'left'},y2:{position:'right',grid:{drawOnChartArea:false}}}});
const loss=mk('c-loss',[ds('policy loss',C('--acc'),'y'),ds('value CE',C('--org'),'y2')],{scales:{y:{position:'left'},y2:{position:'right',grid:{drawOnChartArea:false}}}});
const $=id=>document.getElementById(id),txt=(id,t)=>{$(id).textContent=t},fmt=n=>n==null?'–':Number(n).toLocaleString(undefined,{maximumFractionDigits:0}),
 pct=v=>v==null?'–':(100*v).toFixed(1)+'%',setDot=(id,on)=>$(id).classList.toggle('on',!!on);
function table(rows,cols){const t=document.createElement('table');const h=document.createElement('tr');
 for(const c of cols){const th=document.createElement('th');th.textContent=c[0];h.append(th)}t.append(h);
 for(const r of rows){const tr=document.createElement('tr');for(const c of cols){const td=document.createElement('td');td.textContent=c[1](r);tr.append(td)}t.append(tr)}return t}
async function tick(){
 let s;try{s=await (await fetch('/api/status',{cache:'no-store'})).json()}catch(e){txt('upd','dashboard server unreachable');return}
 setDot('d-chain',s.procs.chain||s.procs.sl||s.procs.rl);
 txt('upd',`${s.procs.sl?'SL running':s.procs.rl?'RL running':'idle'} · updated ${new Date(s.time*1000).toLocaleTimeString()}`);
 const sl=s.sl.status||{},rl=s.rl.status||{},sy=s.system||{};
 if(s.phase==='rl'){txt('t-phase','Stage 2 · RL'+(s.mixed?` · stage ${rl.stage}`:''));txt('t-phase2',`ladder rung: ${rl.rung||'–'}`+(s.mixed?` · 4P: ${rl.rung4} · 4P share ${pct(rl.decisions_4p_share)}`:'')+` · league ${rl.league||0}`);
  txt('t-prog',`${fmt(rl.games)} games`);txt('t-prog2',`update ${fmt(rl.update)} · ${fmt(rl.samples)} samples`);
  txt('t-speed',`${fmt(rl.games_per_min)} g/min`);txt('t-speed2',`play ${(rl.play_s||0).toFixed(0)}s · train ${(rl.train_s||0).toFixed(0)}s per update`);}
 else if(s.phase==='sl'){const st=sl.state||{};txt('t-phase',(sl.done?'Stage 1 done':'Stage 1 · SL')+(s.mixed&&!sl.done?` · stage ${st.stage} (4P ${pct(sl.buffer_4p_share)})`:''));txt('t-phase2',`LR x${st.lr_scale??1} · drops ${st.drops??0} · no-improve ${st.bad_checks??0}`);
  txt('t-prog',`${fmt(st.games_generated)} teacher games`);txt('t-prog2',`${fmt(st.samples)} samples · ${st.checks??0} checks`);
  txt('t-speed',`${fmt(sl.samples_per_s)} samples/s`);txt('t-speed2',`reuse ${(sl.reuse||0).toFixed(2)} · buffer ${fmt(sl.buffer)}`);}
 else{txt('t-phase','not started')}
 txt('t-sys',sy.cpu!=null?`CPU ${sy.cpu.toFixed(0)}%`:'–');txt('t-sys2',sy.mem!=null?`mem ${sy.mem.toFixed(0)}% · ${sy.mem_avail_gb.toFixed(1)} GB free · swap ${sy.swap.toFixed(0)}%`:'');
 const ck=s.sl.checks||[];slv.data.labels=ck.map(c=>fmt(c.games_generated));
 slv.data.datasets[0].data=ck.map(c=>c.val_ce);slv.data.datasets[1].data=ck.map(c=>c.val_top1);slv.data.datasets[2].data=ck.map(c=>c.val_value_ce);slv.update();
 const sk=ck.filter(c=>c.strength);sls.data.labels=sk.map(c=>fmt(c.games_generated));
 sls.data.datasets[0].data=sk.map(c=>c.strength.mcts3000?c.strength.mcts3000.win_rate:null);sls.data.datasets[1].data=sk.map(c=>c.strength['v15.2']?c.strength['v15.2'].win_rate:null);sls.data.datasets[2].data=sk.map(()=>0.5);sls.update();
 if(ck.length){$('sl-table').replaceChildren(table(ck.slice(-12).reverse(),[['check',r=>r.check],['teacher games',r=>fmt(r.games_generated)],['CE',r=>r.val_ce.toFixed(4)],['top-1',r=>pct(r.val_top1)],['value CE',r=>r.val_value_ce.toFixed(4)],['vs MCTS',r=>r.strength&&r.strength.mcts3000?pct(r.strength.mcts3000.win_rate):''],['vs v15.2',r=>r.strength&&r.strength['v15.2']?pct(r.strength['v15.2'].win_rate):'']].concat(s.mixed?[['4P CE',r=>r.val4_ce!=null?r.val4_ce.toFixed(4):''],['4P top-1',r=>pct(r.val4_top1)],['vs 3x teacher',r=>r.strength&&r.strength['4p:teacher']?pct(r.strength['4p:teacher'].win_rate):''],['stage',r=>r.stage!=null?r.stage:'']]:[]).concat([['decision',r=>r.decision]])))}
 document.querySelectorAll('.m4').forEach(e=>e.style.display=s.mixed?'':'none');txt('title',s.title||'V17');
 if(s.mixed){slv4.data.labels=ck.map(c=>fmt(c.positions_generated||c.games_generated));
  slv4.data.datasets[0].data=ck.map(c=>c.val4_ce);slv4.data.datasets[1].data=ck.map(c=>c.val4_top1);slv4.data.datasets[2].data=ck.map(c=>c.val4_value_ce);slv4.update();
  sls4.data.labels=sk.map(c=>fmt(c.positions_generated||c.games_generated));
  sls4.data.datasets[0].data=sk.map(c=>c.strength['4p:teacher']?c.strength['4p:teacher'].win_rate:null);sls4.data.datasets[1].data=sk.map(c=>c.strength['4p:expectimax']?c.strength['4p:expectimax'].win_rate:null);sls4.data.datasets[2].data=sk.map(()=>0.25);sls4.update();
  const E4=s.rl.evals||[];ev4.data.labels=E4.map(e=>fmt(e.games));
  OPP4.forEach(([o],i)=>{ev4.data.datasets[i].data=E4.map(e=>e.results[o]?e.results[o].win_rate:null)});ev4.data.datasets[OPP4.length].data=E4.map(()=>0.25);ev4.update();
  const U4=(s.rl.updates||[]).filter(u=>u.decisions_4p_share!=null);mix.data.labels=U4.map(u=>u.update);
  mix.data.datasets[0].data=U4.map(u=>u.decisions_4p_share);mix.data.datasets[1].data=U4.map(u=>RUNG4.indexOf(u.rung4));mix.update();}
 const E=s.rl.evals||[];ev.data.labels=E.map(e=>fmt(e.games));
 OPP.forEach(([o],i)=>{ev.data.datasets[i].data=E.map(e=>e.results[o]?e.results[o].win_rate:null)});ev.data.datasets[OPP.length].data=E.map(()=>0.5);ev.update();
 const U=s.rl.updates||[];const lab=U.map(u=>u.update);
 ppo.data.labels=lab;ppo.data.datasets[0].data=U.map(u=>u.kl);ppo.data.datasets[1].data=U.map(u=>u.clipfrac);ppo.data.datasets[2].data=U.map(u=>u.entropy_norm);ppo.update();
 loss.data.labels=lab;loss.data.datasets[0].data=U.map(u=>u.policy_loss);loss.data.datasets[1].data=U.map(u=>u.value_ce);loss.update();
 const L=U.filter(u=>u.ladder).slice(-10).reverse();const box=$('ladder');box.replaceChildren();
 const p=document.createElement('div');p.textContent=s.phase==='rl'?`Current rung: ${rl.rung} · league snapshots: ${rl.league}`:'starts with Stage 2';p.style.marginBottom='8px';box.append(p);
 if(L.length)box.append(table(L,[['update',r=>r.update],['games',r=>fmt(r.games)],['window',r=>r.ladder]]));
 txt('log',(s.log||[]).join('\n'));
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
                self._send(200, json.dumps(status(), default=float).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="checkpoints/v17")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--chain", default="run_v17.sh", help="chain script name (process check)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    global CHAIN, TITLE
    RUN = Path(a.run_dir)
    CHAIN = a.chain
    if a.title:
        TITLE = a.title
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"V17 dashboard on http://localhost:{a.port}/ (run dir {RUN})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
