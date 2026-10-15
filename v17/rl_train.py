"""V17 Stage 2 — infinite RL: PPO + GAE, win/loss reward only.

Opponents (docs/V17_CLEAN_PIPELINE_PLAN.md §3.1), mixed per game on one
persistent Rust farm:
  * the current learner on both seats (self-play; both seats are trained on),
  * past snapshots of the learner (league; recency-weighted),
  * a progressive bot ladder Heuristic -> Expert -> Depth2 -> MCTS-3000, advanced
    when the GREEDY learner beats the current rung >= --promote in --windows
    consecutive evaluation windows (demoted below --demote).
Bots and snapshots are never imitated; only the learner's own decisions train.

Learning: GAE (lambda, gamma) over each seat's decisions with the network's own
value estimate (P(win)) and a terminal reward of 1 (won) / 0 (lost); PPO clipped
policy loss on decisions with >= 2 distinct legal cells; entropy normalised by
log(#legal cells); value head = 4-way cross-entropy on the actual winner;
advantage normalisation; KL early stop; exact micro-batching.

The run never ends by itself. `touch <run>/stop` stops at a safe point;
re-running resumes (model, optimizer, counters, league, ladder, history).

2P+4P extension (2026-10-07; defaults reproduce the 2P-only run exactly):
  --frac4-stages "0.8,0.5": target share of TRAINING DECISIONS from 4P games per
  stage (a 4P game yields several times more decisions, so game weights are
  re-balanced every update from the measured decisions per game). 4P games: the
  learner in all 4 seats (self-play), learner + 3 bots of the current 4P rung,
  learner + 3 copies of one past snapshot. A second ladder RUNGS4 (heuristic ->
  expectimax -> depth2 -> the 4P teacher) uses the 4P equivalents of the 2P
  thresholds under a Luce strength model (60% in 2P = strength ratio 1.5 ->
  1.5/(1.5+3) = 33.3% vs 3 copies; 45% -> 21.4%). A stage that is not the last
  ends when the 4P ladder reaches its top rung or after --stage-games games.
  Full evaluations add the 4P suite (one model seat vs 3 copies of each bot).
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import alphaludo_rs

from .common import append_jsonl, dense_targets, pick_device, save_atomic, stop_requested, write_json_atomic
from .evaluate import match, run_isolated, DEFAULT_SUITE, FOURP_SUITE, reference_spec
from .inference import V17Player
from .model import V17Net, count_params

RUNGS = [("heuristic", {"bot": "heuristic"}), ("expert", {"bot": "expert"}), ("depth2", {"bot": "depth2"}),
         ("mcts3000", {"bot": "mcts", "sims": 3000})]
# 4P ladder: names are evaluate.py references (one learner seat vs 3 copies).
RUNGS4 = ["4p:heuristic", "4p:expectimax", "4p:depth2", "4p:teacher"]


def gae(records: dict, gamma: float, lam: float):
    """Advantages and value targets per decision. Trajectory = (game, seat), ordered by step.
    Terminal reward (1 won / 0 lost) arrives after the seat's last decision."""
    n = len(records["game"])
    adv = np.zeros(n, np.float32)
    keep = ~np.isnan(records["reward"])
    order = np.lexsort((records["step"], records["seat"], records["game"]))
    g, s, v, r = records["game"][order], records["seat"][order], records["value"][order], records["reward"][order]
    a_sorted = np.zeros(n, np.float32)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and g[j + 1] == g[i] and s[j + 1] == s[i]:
            j += 1
        if not np.isnan(r[i]):
            last = 0.0
            for t in range(j, i - 1, -1):
                nxt = r[t] if t == j else gamma * v[t + 1]
                delta = nxt - v[t]
                last = delta + gamma * lam * last
                a_sorted[t] = last
        i = j + 1
    adv[order] = a_sorted
    return adv, keep


def winner_rel(records: dict, results_winner: dict) -> np.ndarray:
    """(winner seat − mover seat) mod 4 per decision (255 if unfinished)."""
    w = np.array([results_winner.get(int(gm), -1) for gm in records["game"]], dtype=np.int64)
    rel = (w - records["seat"].astype(np.int64)) % 4
    rel[w < 0] = 255
    return rel


class League:
    """Snapshots of the learner on disk (capped), recency-weighted sampling."""

    def __init__(self, folder: Path, cap: int):
        self.folder = folder
        self.cap = cap
        folder.mkdir(parents=True, exist_ok=True)

    def paths(self):
        return sorted(self.folder.glob("snap_*.pt"))

    def add(self, net, step: int):
        save_atomic({"model_state_dict": net.state_dict(), "model_cfg": net.cfg, "step": step},
                    self.folder / f"snap_{step:09d}.pt")
        ps = self.paths()
        for old in ps[: max(0, len(ps) - self.cap)]:
            old.unlink(missing_ok=True)


def load_net(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = V17Net(**ck.get("model_cfg", {}))
    net.load_state_dict(ck["model_state_dict"])
    net.train(False)
    return net.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="checkpoints/v17")
    ap.add_argument("--init", default=None, help="initial weights (default: <run>/sl_final.pt)")
    ap.add_argument("--concurrency", type=int, default=256, help="games in flight on the farm")
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="coefficient on normalised entropy (0.003 until update 155; raised 2026-10-07 after flat evals 1-3)")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--kl-stop", type=float, default=0.03)
    ap.add_argument("--temp", type=float, default=1.0, help="sampling temperature of training games")
    # opponent mix (shares of games); shifts to the 'late' mix once the top rung is reached
    ap.add_argument("--mix-early", default="0.35,0.45,0.20", help="bot,self,past")
    ap.add_argument("--mix-late", default="0.15,0.60,0.25", help="bot,self,past")
    ap.add_argument("--snapshot-every", type=int, default=25, help="updates between league snapshots")
    ap.add_argument("--league-cap", type=int, default=20)
    # ladder
    ap.add_argument("--ladder-every", type=int, default=10, help="updates between ladder windows")
    ap.add_argument("--window-games", type=int, default=300)
    ap.add_argument("--windows", type=int, default=3, help="consecutive windows to promote/demote")
    ap.add_argument("--promote", type=float, default=0.60)
    ap.add_argument("--demote", type=float, default=0.45)
    # evaluation / bookkeeping
    ap.add_argument("--eval-every-games", type=int, default=25_000)
    ap.add_argument("--eval-games", type=int, default=1000)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=1717)
    # 2P+4P extension
    ap.add_argument("--frac4-stages", default="0", help="4P share of training decisions per stage, e.g. 0.8,0.5")
    ap.add_argument("--stage-games", type=int, default=150_000, help="max games in a non-final stage")
    ap.add_argument("--promote4", type=float, default=1.5 / 4.5, help="4P promotion (Luce-equivalent of 60%%)")
    ap.add_argument("--demote4", type=float, default=(0.45 / 0.55) / (0.45 / 0.55 + 3), help="4P demotion")
    ap.add_argument("--start-rung", type=int, default=0, help="2P ladder rung for a FRESH run")
    ap.add_argument("--start-rung4", type=int, default=0, help="4P ladder rung for a FRESH run")
    a = ap.parse_args()
    # Optional per-run overrides (applied on every start, over the command line): lets the
    # RL settings of a chain that is already running (e.g. still in SL) be changed safely.
    ov_path = Path(a.run_dir) / "rl_overrides.json"
    if ov_path.exists():
        import json as _json
        for k, v in _json.loads(ov_path.read_text()).items():
            k = k.replace("-", "_")
            if not hasattr(a, k):
                raise SystemExit(f"rl_overrides.json: unknown setting {k!r}")
            setattr(a, k, v)
            print(f"[override] {k} = {v!r}", flush=True)
    stages = [float(v) for v in str(a.frac4_stages).split(",")]
    mixed = max(stages) > 0

    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    ck_path, status_path, metrics_path = run / "rl_latest.pt", run / "rl_status.json", run / "rl_metrics.jsonl"
    league = League(run / "league", a.league_cap)
    mix_early = [float(v) for v in a.mix_early.split(",")]
    mix_late = [float(v) for v in a.mix_late.split(",")]

    st = {"update": 0, "games": 0, "samples": 0, "rung": a.start_rung, "windows": [], "next_eval": a.eval_every_games,
          "evals": 0, "top_reached": a.start_rung >= len(RUNGS) - 1,
          "stage": 0, "stage_start_games": 0, "rung4": a.start_rung4, "windows4": [],
          "top_reached4": a.start_rung4 >= len(RUNGS4) - 1, "dpg": {"2": 0.0, "4": 0.0}, "games4": 0}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net = V17Net(**ck.get("model_cfg", {}))
        net.load_state_dict(ck["model_state_dict"])
        net.to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        opt.load_state_dict(ck["optimizer_state_dict"])
        st.update(ck["state"])
        print(f"[resume] update {st['update']:,}, {st['games']:,} games, rung {RUNGS[st['rung']][0]}", flush=True)
    else:
        init = Path(a.init) if a.init else run / "sl_final.pt"
        net = V17Net()
        if init.exists():
            ck = torch.load(init, map_location="cpu", weights_only=False)
            net.load_state_dict(ck["model_state_dict"])
            print(f"[init] weights from {init}", flush=True)
        else:
            print(f"[init] {init} not found -> RANDOM weights", flush=True)
        net.to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        league.add(net, 0)
    print(f"[rl] V17Net {count_params(net):,} params on {dev}", flush=True)

    # Inference inside this process uses PyTorch on the training device: one Metal
    # framework per process (MLX evaluations run in a separate process).
    actor_dev = str(dev)

    def snapshot_players():
        out = {}
        for i, p in enumerate(league.paths()[-a.league_cap:]):
            out[1 + i] = V17Player(load_net(p), backend="torch", device=actor_dev)
        return out

    def learner_player():
        return V17Player(copy.deepcopy(net), backend="torch", device=actor_dev)

    players = {0: learner_player(), **snapshot_players()}
    n_models = 1 + a.league_cap
    farm = alphaludo_rs.Farm(["unified"] * n_models, concurrency=a.concurrency, seed=a.seed + st["update"])

    def mode_matchups(n_seats: int, top: bool, bot_spec: dict):
        bot_w, self_w, past_w = mix_late if top else mix_early
        L = {"net": 0, "temp": a.temp, "record": True}
        m = [{"players": [L] * n_seats, "weight": self_w},
             {"players": [L] + [dict(bot_spec)] * (n_seats - 1), "weight": bot_w}]
        snaps = sorted(k for k in players if k > 0)
        if snaps:
            w = np.arange(1, len(snaps) + 1, dtype=np.float64)        # newer snapshots more often
            w = w / w.sum() * past_w
            for k, wk in zip(snaps, w):
                m.append({"players": [L] + [{"net": k, "temp": a.temp}] * (n_seats - 1), "weight": float(wk)})
        return m

    def set_mix():
        m2 = mode_matchups(2, st["top_reached"], RUNGS[st["rung"]][1])
        f4 = stages[min(st["stage"], len(stages) - 1)]
        if f4 <= 0:
            farm.set_matchups(m2)
            return
        m4 = mode_matchups(4, st["top_reached4"], reference_spec(RUNGS4[st["rung4"]]))
        # Game weights so that 4P supplies ~f4 of the training decisions:
        # W4/W2 = f4 * d2 / ((1 - f4) * d4), with measured decisions per game (prior 2.5x).
        d2 = st["dpg"]["2"] or 1.0
        d4 = st["dpg"]["4"] or 2.5 * d2
        r = 1e6 if f4 >= 1 else f4 * d2 / ((1 - f4) * d4)
        w2, w4 = 1.0 / (1.0 + r), r / (1.0 + r)
        for mm in m2:
            mm["weight"] *= w2
        for mm in m4:
            mm["weight"] *= w4
        farm.set_matchups([mm for mm in m2 + m4 if mm["weight"] > 0])

    set_mix()

    def save(extra=None):
        save_atomic({"model_state_dict": net.state_dict(), "optimizer_state_dict": opt.state_dict(),
                     "model_cfg": net.cfg, "state": st, "args": vars(a), **(extra or {})}, ck_path)

    t_save = time.time()
    while not stop_requested(run):
        it_t0 = time.time()
        # ---- 1. play until enough games finished
        recs, results = [], []
        while sum(len(r) for r in results) < a.games_per_iter:
            if not farm.collect():
                break
            for m in farm.models_pending():
                x, mask = farm.batch(m)
                p, v = players[m].answer(x, mask)
                farm.submit(m, p, v)
            res = farm.take_results()
            if res:
                results.append(res)
                recs.append(farm.take_records())
            if stop_requested(run):
                break
        if not results:
            continue
        results = [r for rr in results for r in rr]
        rec = {k: np.concatenate([d[k] for d in recs]) for k in recs[0]}
        if mixed:
            # learner training decisions per finished game, per mode (EMA) -> set_mix balance
            n_seats = np.array([sum(1 for x in ros if x != 255) for (_, _, ros, _, _) in results])
            dec_all = rec["n_cells"] >= 2
            for nm in (2, 4):
                g_n = int((n_seats == nm).sum())
                if g_n:
                    d_n = float((dec_all & (rec["n_players"] == nm)).sum()) / g_n
                    k_ = str(nm)
                    st["dpg"][k_] = d_n if not st["dpg"][k_] else 0.8 * st["dpg"][k_] + 0.2 * d_n
            st["games4"] += int((n_seats == 4).sum())
        win_of = {int(g_): int(w_) for (g_, _, _, w_, _) in results}
        st["games"] += len(results)
        play_s = time.time() - it_t0

        # ---- 2. advantages
        adv, keep = gae(rec, a.gamma, a.lam)
        wrel = winner_rel(rec, win_of)
        decision = rec["n_cells"] >= 2
        sel = np.nonzero(keep)[0]
        if len(sel) < a.batch_size:
            continue
        adv_sel = adv[sel]
        adv_n = (adv_sel - adv_sel.mean()) / (adv_sel.std() + 1e-8)

        # ---- 3. PPO
        t_train = time.time()
        net.train(True)
        mask_all, _ = dense_targets(rec["cells"], np.zeros_like(rec["cells"], np.float32), rec["n_cells"])
        stats = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
        stopped_early = False
        for ep in range(a.epochs):
            perm = rng.permutation(len(sel))
            ep_kl = []
            for lo in range(0, len(perm) - a.batch_size + 1, a.batch_size):
                bi = perm[lo:lo + a.batch_size]
                idx = sel[bi]
                x = torch.from_numpy(rec["x"][idx]).to(dev)
                msk = torch.from_numpy(mask_all[idx]).to(dev)
                act = torch.from_numpy(rec["action"][idx].astype(np.int64)).to(dev)
                old = torch.from_numpy(rec["logp"][idx]).to(dev)
                A = torch.from_numpy(adv_n[bi].astype(np.float32)).to(dev)
                dec = torch.from_numpy(decision[idx]).to(dev)
                wr = torch.from_numpy(wrel[idx]).to(dev)
                ncell = torch.from_numpy(rec["n_cells"][idx].astype(np.float32)).to(dev)
                n_dec = dec.sum().clamp(min=1).float()
                n_fin = (wr < 4).sum().clamp(min=1).float()
                opt.zero_grad(set_to_none=True)
                mbs = a.micro_batch if 0 < a.micro_batch < a.batch_size else a.batch_size
                kl_b = 0.0
                for mlo in range(0, a.batch_size, mbs):
                    s_ = slice(mlo, mlo + mbs)
                    pol, val = net.forward_logits(x[s_], msk[s_])
                    logp_all = F.log_softmax(pol, -1)
                    lp = logp_all.gather(1, act[s_, None]).squeeze(1)
                    ratio = torch.exp(lp - old[s_])
                    d = dec[s_].float()
                    surr = torch.min(ratio * A[s_], ratio.clamp(1 - a.clip, 1 + a.clip) * A[s_])
                    pl = -(surr * d).sum() / n_dec
                    p_all = logp_all.exp()
                    ent = -(torch.where(msk[s_] > 0.5, p_all * logp_all, torch.zeros_like(p_all))).sum(-1)
                    ent_n = ent / torch.log(ncell[s_].clamp(min=2))
                    el = -(ent_n * d).sum() / n_dec
                    fin = (wr[s_] < 4)
                    vl = (F.cross_entropy(val, wr[s_].clamp(max=3).long(), reduction="none") * fin).sum() / n_fin
                    loss = pl + a.ent_coef * el + a.value_coef * vl
                    loss.backward()
                    with torch.no_grad():
                        stats["pl"] += float(pl); stats["vl"] += float(vl); stats["ent"] += float(-el)
                        k = ((old[s_] - lp) * d).sum() / n_dec
                        kl_b += float(k)
                        stats["kl"] += float(k)
                        stats["clipfrac"] += float((((ratio - 1).abs() > a.clip).float() * d).sum() / n_dec)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats["n"] += 1
                ep_kl.append(kl_b)
            if np.mean(ep_kl) > a.kl_stop:
                stopped_early = True
                break
        st["update"] += 1
        st["samples"] += int(len(sel) * (ep + 1))
        train_s = time.time() - t_train
        players[0] = learner_player()
        if dev.type == "mps":
            torch.mps.empty_cache()

        # ---- 4. league snapshot
        if st["update"] % a.snapshot_every == 0:
            league.add(net, st["update"])
            players = {0: players[0], **snapshot_players()}

        # ---- 5. ladder window (greedy learner vs current rung)
        ladder_msg = ""
        if st["update"] % a.ladder_every == 0:
            r = match(players[0], RUNGS[st["rung"]][0], a.window_games, seed=int(rng.integers(1 << 30)))
            st["windows"] = (st["windows"] + [{"rung": st["rung"], "wr": r["win_rate"]}])[-a.windows:]
            ws = [w["wr"] for w in st["windows"] if w["rung"] == st["rung"]]
            ladder_msg = f"ladder {RUNGS[st['rung']][0]} {r['win_rate']:.3f}"
            if len(ws) >= a.windows and min(ws) >= a.promote and st["rung"] < len(RUNGS) - 1:
                st["rung"] += 1
                st["windows"] = []
                ladder_msg += f" -> PROMOTED to {RUNGS[st['rung']][0]}"
            elif len(ws) >= a.windows and max(ws) < a.demote and st["rung"] > 0:
                st["rung"] -= 1
                st["windows"] = []
                ladder_msg += f" -> demoted to {RUNGS[st['rung']][0]}"
            if st["rung"] == len(RUNGS) - 1:
                st["top_reached"] = True
            if mixed:
                r4 = match(players[0], RUNGS4[st["rung4"]], a.window_games, seed=int(rng.integers(1 << 30)))
                st["windows4"] = (st["windows4"] + [{"rung": st["rung4"], "wr": r4["win_rate"]}])[-a.windows:]
                ws4 = [w["wr"] for w in st["windows4"] if w["rung"] == st["rung4"]]
                ladder_msg += f" | {RUNGS4[st['rung4']]} {r4['win_rate']:.3f}"
                if len(ws4) >= a.windows and min(ws4) >= a.promote4 and st["rung4"] < len(RUNGS4) - 1:
                    st["rung4"] += 1
                    st["windows4"] = []
                    ladder_msg += f" -> PROMOTED to {RUNGS4[st['rung4']]}"
                elif len(ws4) >= a.windows and max(ws4) < a.demote4 and st["rung4"] > 0:
                    st["rung4"] -= 1
                    st["windows4"] = []
                    ladder_msg += f" -> demoted to {RUNGS4[st['rung4']]}"
                if st["rung4"] == len(RUNGS4) - 1:
                    st["top_reached4"] = True
        if mixed and st["stage"] < len(stages) - 1:
            if st["top_reached4"] or st["games"] - st["stage_start_games"] >= a.stage_games:
                st["stage"] += 1
                st["stage_start_games"] = st["games"]
                ladder_msg += f" | -> RL stage {st['stage']} (4P share {stages[st['stage']]})"
        set_mix()

        # Each micro-batch term is already normalised by the FULL batch counts, so the
        # per-update value is the sum over its micro-batches: average over updates.
        n = max(1, stats["n"])
        msg = {"update": st["update"], "games": st["games"], "samples": st["samples"],
               "policy_loss": stats["pl"] / n, "value_ce": stats["vl"] / n, "entropy_norm": stats["ent"] / n,
               "kl": stats["kl"] / n, "clipfrac": stats["clipfrac"] / n, "kl_stopped": stopped_early,
               "decisions": int(len(sel)), "play_s": play_s, "train_s": train_s,
               "games_per_min": len(results) / max(1e-6, (time.time() - it_t0)) * 60,
               "rung": RUNGS[st["rung"]][0], "league": len(league.paths())}
        if mixed:
            dec4 = float(((rec["n_cells"][sel] >= 2) & (rec["n_players"][sel] == 4)).mean())
            msg.update({"rung4": RUNGS4[st["rung4"]], "stage": st["stage"], "decisions_4p_share": dec4,
                        "games4": st["games4"]})
        print(f"upd {st['update']:6d} | games {st['games']:,} | pl {msg['policy_loss']:+.4f} | V-CE {msg['value_ce']:.4f} "
              f"| H {msg['entropy_norm']:.3f} | KL {msg['kl']:.4f} | clip {msg['clipfrac']:.3f} | "
              f"{msg['games_per_min']:.0f} g/min | "
              + (f"4P {msg['decisions_4p_share']:.2f} st{st['stage']} | " if mixed else "") + ladder_msg, flush=True)
        write_json_atomic(status_path, {"phase": "rl", **msg, "state": st, "updated": time.time()})
        append_jsonl(metrics_path, {"kind": "update", **msg, "ladder": ladder_msg})

        # ---- 6. full evaluation
        if st["games"] >= st["next_eval"]:
            st["next_eval"] += a.eval_every_games
            st["evals"] += 1
            (run / "evals").mkdir(exist_ok=True)
            ev_ck = run / "evals" / f"model_{st['games']:010d}.pt"
            save_atomic({"model_state_dict": net.state_dict(), "model_cfg": net.cfg, "state": st}, ev_ck)
            suite_ = DEFAULT_SUITE + (FOURP_SUITE if mixed else [])
            res = run_isolated(ev_ck, a.eval_games, suite_, seed=1700, out_json=run / "evals" / "last.json")
            line = " | ".join(f"{o} {r['win_rate']:.3f}±{r['stderr']:.3f}" for o, r in res.items())
            print(f"[eval {st['evals']} @ {st['games']:,} games] {line}", flush=True)
            append_jsonl(metrics_path, {"kind": "eval", "games": st["games"], "update": st["update"],
                                        "results": res})
            save()
            t_save = time.time()
        elif time.time() - t_save > a.save_every_min * 60:
            save()
            t_save = time.time()
    save()
    print("[exit] clean", flush=True)


if __name__ == "__main__":
    main()
