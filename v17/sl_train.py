"""V17 Stage 1 — streaming supervised learning from the fixed MCTS-3000 teacher.

Rust worker threads play MCTS-3000 vs MCTS-3000 games (alphaludo_rs.SlStream);
kept positions flow into an in-memory replay buffer; the GPU learner samples
from it. Nothing is stored except a fixed held-out validation set and
checkpoints. Generation is paced to the learner by a target replay ratio
(samples trained per position generated) and by the stream's bounded queue.

Targets: the teacher's visit distribution over legal source cells (soft
cross-entropy, decisions with >= 2 legal cells only) and the winner's relative
seat for the 4-way value head (finished games only).

Auto-stop (docs/V17_CLEAN_PIPELINE_PLAN.md §3.0): every --check-every samples,
validation metrics on the fixed held-out set (+ a strength match every
--strength-every checks). A check "improves" if any metric beats its best by more
than its tolerance. After --patience consecutive non-improving checks the LR is
multiplied by --lr-drop (at most --max-drops times); a plateau at the lowest LR
ends SL — never before --min-games, always by --max-games. Then sl_final.pt is
written and status 'done' recorded (the launcher starts RL).

Resumable (model, optimizer, LR state, counters, history); `touch <run>/stop`.

2P+4P extension (2026-10-07; defaults reproduce the original 2P trainer exactly):
  --init            warm-start weights (e.g. the frozen V17-2P model), fresh optimizer.
  --frac4-stages    target share of 4P POSITIONS per stage, e.g. "0.85,0.5". The
                    stream's 4P games use the teacher in --teacher4 (v17/teacher4.json).
                    Every stage but the last tracks the 4P validation metrics only and
                    ends at a plateau (no LR drop) -> next stage; the last stage tracks
                    both modes and runs the LR-drop / auto-stop logic above.
  Validation and strength are reported per mode: val_* = 2P set (sl_val.npz),
  val4_* = 4P set (sl_val4.npz); strength vs MCTS-3000 (2P) and vs 3x teacher (4P).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

import alphaludo_rs

from .common import (append_jsonl, dense_targets, pick_device, policy_value_losses, save_atomic,
                     stop_requested, write_json_atomic)
from .model import V17Net, count_params

FIELDS = ("x", "cells", "pi", "n_cells", "action", "winner_rel", "legal_tokens")
FIELDS4 = FIELDS + ("n_players",)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.x = np.zeros((capacity, 1125), np.int8)
        self.cells = np.zeros((capacity, 4), np.uint8)
        self.pi = np.zeros((capacity, 4), np.float32)
        self.n_cells = np.zeros(capacity, np.uint8)
        self.winner = np.zeros(capacity, np.uint8)
        self.legal = np.zeros(capacity, np.uint8)
        self.nplayers = np.full(capacity, 2, np.uint8)
        self.size = 0
        self.head = 0
        self.inserted = 0

    def add(self, d: dict):
        k = len(d["action"])
        for i in range(0, k, self.cap):
            chunk = {f: d[f][i:i + self.cap] for f in FIELDS}
            n = len(chunk["action"])
            idx = (self.head + np.arange(n)) % self.cap
            self.x[idx] = chunk["x"]
            self.cells[idx] = chunk["cells"]
            self.pi[idx] = chunk["pi"]
            self.n_cells[idx] = chunk["n_cells"]
            self.winner[idx] = chunk["winner_rel"]
            self.legal[idx] = chunk["legal_tokens"]
            if "n_players" in d:
                self.nplayers[idx] = d["n_players"][i:i + self.cap]
            self.head = (self.head + n) % self.cap
            self.size = min(self.cap, self.size + n)
            self.inserted += n

    def sample(self, n: int, rng: np.random.Generator):
        i = rng.integers(0, self.size, n)
        return self.x[i], self.cells[i], self.pi[i], self.n_cells[i], self.winner[i], self.legal[i]


def to_tensors(x, cells, pi, n_cells, winner, legal, device):
    mask, target = dense_targets(cells, pi, n_cells)
    t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)
    return (t(x, torch.int8), t(mask), t(target), t(legal >= 2), t(winner.astype(np.int64), torch.int64))


def collect_validation(path: Path, n_positions: int, sims: int, keep_prob: float, threads: int,
                       teacher4: dict | None = None) -> dict:
    """Generates (once) and stores a fixed held-out set from separate games.
    teacher4 given -> a 4P-only set from that teacher (different seed)."""
    if path.exists():
        z = np.load(path)
        return {f: z[f] for f in FIELDS}
    kind = "4P" if teacher4 else "2P"
    print(f"[val] generating {n_positions:,} held-out {kind} positions (separate games, fixed seed)...", flush=True)
    if teacher4:
        s = alphaludo_rs.SlStream(threads=threads, sims=sims, keep_prob=keep_prob, capacity=n_positions + 5000,
                                  seed=444_333_222, frac4=1.0, teacher4=teacher4)
    else:
        s = alphaludo_rs.SlStream(threads=threads, sims=sims, keep_prob=keep_prob, capacity=n_positions + 5000,
                                  seed=987_654_321)
    parts, got = [], 0
    while got < n_positions:
        d = s.take(n_positions - got, 5000)
        if len(d["action"]):
            parts.append(d)
            got += len(d["action"])
    s.stop()
    val = {f: np.concatenate([p[f] for p in parts])[:n_positions] for f in FIELDS}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **val)
    tmp.replace(path)
    return val


@torch.no_grad()
def validate(net, val: dict, device, batch: int = 1024) -> dict:
    was = net.training
    net.train(False)
    tot = {"ce": 0.0, "nd": 0, "top1": 0, "vce": 0.0, "nv": 0, "vacc": 0}
    n = len(val["action"])
    for lo in range(0, n, batch):
        sl = slice(lo, lo + batch)
        x, mask, target, dec, win = to_tensors(val["x"][sl], val["cells"][sl], val["pi"][sl], val["n_cells"][sl],
                                               val["winner_rel"][sl], val["legal_tokens"][sl], device)
        p, v = net.forward_logits(x, mask)
        pl, nd, vl, nv = policy_value_losses(p, v, target, dec.float(), win)
        tot["ce"] += float(pl); tot["nd"] += int(nd)
        tot["vce"] += float(vl); tot["nv"] += int(nv)
        top = (p.argmax(-1) == target.argmax(-1)) & (dec > 0.5)
        tot["top1"] += int(top.sum())
        fin = win < 4
        tot["vacc"] += int(((v.argmax(-1) == win) & fin).sum())
    net.train(was)
    return {"val_ce": tot["ce"] / max(1, tot["nd"]), "val_top1": tot["top1"] / max(1, tot["nd"]),
            "val_value_ce": tot["vce"] / max(1, tot["nv"]), "val_value_acc": tot["vacc"] / max(1, tot["nv"])}


def strength(net, games: int, run: Path, step: int, opponents=("mcts3000", "v15.2")) -> dict:
    """Greedy matches (default vs MCTS-3000 and v15.2), run in a separate MLX process."""
    from .evaluate import run_isolated
    ck = run / "strength_tmp.pt"
    save_atomic({"model_state_dict": net.state_dict(), "model_cfg": net.cfg}, ck)
    res = run_isolated(ck, games, list(opponents), seed=5100, out_json=run / "strength_tmp.json")
    return {o: {"win_rate": r["win_rate"], "stderr": r["stderr"], "games": r["games"]} for o, r in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="checkpoints/v17")
    ap.add_argument("--threads", type=int, default=10, help="Rust generator threads")
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--keep-prob", type=float, default=0.25)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--warmup-positions", type=int, default=50_000)
    ap.add_argument("--replay-ratio", type=float, default=1.5, help="max samples trained per position generated")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--val-positions", type=int, default=40_000)
    ap.add_argument("--check-every", type=int, default=2_000_000, help="samples between checks")
    ap.add_argument("--strength-every", type=int, default=2, help="strength match every N checks (0 = never)")
    ap.add_argument("--strength-games", type=int, default=600)
    ap.add_argument("--ce-tol", type=float, default=0.002, help="relative CE improvement that counts")
    ap.add_argument("--top1-tol", type=float, default=0.002, help="absolute top-1 improvement that counts")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr-drop", type=float, default=0.3)
    ap.add_argument("--max-drops", type=int, default=2)
    ap.add_argument("--min-games", type=int, default=300_000)
    ap.add_argument("--max-games", type=int, default=1_000_000)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=17)
    # 2P+4P extension
    ap.add_argument("--init", default=None, help="warm-start weights (used only when no sl_latest.pt exists)")
    ap.add_argument("--frac4-stages", default="0", help="target 4P position share per stage, e.g. 0.85,0.5")
    ap.add_argument("--teacher4", default=None, help="4P teacher JSON (default v17/teacher4.json)")
    ap.add_argument("--val4-positions", type=int, default=40_000)
    ap.add_argument("--min-positions", type=int, default=0, help="no LR drop / stop before this many positions")
    ap.add_argument("--max-positions", type=int, default=0, help="stop at this many positions (0 = off)")
    ap.add_argument("--min-stage-positions", type=int, default=2_000_000,
                    help="no early-stage plateau switch before this many positions in the stage")
    ap.add_argument("--strength4-opponents", default="4p:teacher,4p:expectimax")
    a = ap.parse_args()
    stages = [float(v) for v in a.frac4_stages.split(",")]
    mixed = max(stages) > 0
    teacher4 = None
    if mixed:
        from .evaluate import TEACHER4_PATH
        import json as _json
        teacher4 = _json.loads(Path(a.teacher4 or TEACHER4_PATH).read_text())["spec"]

    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    status_path, metrics_path, ck_path = run / "sl_status.json", run / "sl_metrics.jsonl", run / "sl_latest.pt"

    net = V17Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    st = {"step": 0, "samples": 0, "games_generated": 0, "positions_generated": 0, "lr_scale": 1.0, "drops": 0,
          "bad_checks": 0, "checks": 0, "best": {}, "phase": "sl", "done": False,
          "stage": 0, "stage_start_positions": 0, "games4": 0, "positions4": 0}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        st.update(ck["state"])
        print(f"[resume] step {st['step']:,}, {st['samples']:,} samples, {st['games_generated']:,} teacher games, "
              f"lr x{st['lr_scale']}" + (f", stage {st['stage']}" if mixed else ""), flush=True)
    elif a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
        print(f"[init] weights from {a.init} (fresh optimizer)", flush=True)
    if st["done"]:
        print("[sl] already finished — nothing to do", flush=True)
        return
    print(f"[sl] V17Net {count_params(net):,} params on {dev} | teacher MCTS-{a.sims} self-play, "
          f"{a.threads} threads, keep {a.keep_prob}"
          + (f" | 4P teacher {teacher4} | 4P share per stage {stages}" if mixed else ""), flush=True)

    val = collect_validation(run / "sl_val.npz", a.val_positions, a.sims, a.keep_prob, a.threads)
    val4 = collect_validation(run / "sl_val4.npz", a.val4_positions, a.sims, a.keep_prob, a.threads,
                              teacher4=teacher4) if mixed else None
    st["stage"] = min(st["stage"], len(stages) - 1)
    stream = alphaludo_rs.SlStream(threads=a.threads, sims=a.sims, keep_prob=a.keep_prob,
                                   capacity=max(20_000, a.buffer // 4), seed=a.seed * 7919 + st["step"],
                                   **({"frac4": stages[st["stage"]], "teacher4": teacher4} if mixed else {}))
    buf = ReplayBuffer(a.buffer)
    games0, pos0 = st["games_generated"], st["positions_generated"]
    g40, p40 = st["games4"], st["positions4"]

    def pull(n, timeout_ms=200):
        d = stream.take(n, timeout_ms)
        if len(d["action"]):
            buf.add(d)
        g, p, _ = stream.stats()
        st["games_generated"], st["positions_generated"] = games0 + g, pos0 + p
        if mixed:
            g4, p4 = stream.stats4()
            st["games4"], st["positions4"] = g40 + g4, p40 + p4

    print(f"[sl] warming the buffer to {a.warmup_positions:,} positions...", flush=True)
    while buf.size < a.warmup_positions and not stop_requested(run):
        pull(a.warmup_positions - buf.size, 2000)

    def lr_now():
        w = min(1.0, (st["step"] + 1) / a.warmup_steps)
        return a.lr * st["lr_scale"] * w

    def save(path=ck_path, extra=None):
        save_atomic({"model_state_dict": net.state_dict(), "optimizer_state_dict": opt.state_dict(),
                     "model_cfg": net.cfg, "state": st, "args": vars(a), **(extra or {})}, path)

    net.train(True)
    samples0 = st["samples"]          # replay pacing uses this session's counters (buffer restarts empty)
    t_log, t_save, n_log = time.time(), time.time(), 0
    agg = {"pl": 0.0, "vl": 0.0}
    next_check = (st["samples"] // a.check_every + 1) * a.check_every
    while not stop_requested(run):
        # Keep samples-trained / positions-generated <= replay ratio (wait for data if needed).
        need = int((st["samples"] - samples0) / a.replay_ratio) - (buf.inserted - a.warmup_positions)
        if need > 0:
            pull(min(need, 50_000), 50 if need < 5_000 else 1000)
            if int((st["samples"] - samples0) / a.replay_ratio) - (buf.inserted - a.warmup_positions) > 20_000:
                continue  # generator is behind: wait rather than over-reuse data
        else:
            pull(10_000, 0)

        for g_ in opt.param_groups:
            g_["lr"] = lr_now()
        xb, cb, pb, nb, wb, lb = buf.sample(a.batch_size, rng)
        x, mask, target, dec, win = to_tensors(xb, cb, pb, nb, wb, lb, dev)
        n_dec = dec.sum().clamp(min=1).float()
        n_fin = (win < 4).sum().clamp(min=1).float()
        opt.zero_grad(set_to_none=True)
        mb = a.micro_batch if 0 < a.micro_batch < a.batch_size else a.batch_size
        pl_tot, vl_tot = 0.0, 0.0
        for lo in range(0, a.batch_size, mb):
            sl = slice(lo, lo + mb)
            p, v = net.forward_logits(x[sl], mask[sl])
            pl, _, vl, _ = policy_value_losses(p, v, target[sl], dec[sl].float(), win[sl])
            loss = pl / n_dec + a.value_coef * vl / n_fin
            loss.backward()
            pl_tot += pl.detach(); vl_tot += vl.detach()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        st["step"] += 1
        st["samples"] += a.batch_size
        agg["pl"] += pl_tot / n_dec; agg["vl"] += vl_tot / n_fin; n_log += 1

        if st["step"] % 200 == 0:
            dt = time.time() - t_log
            msg = {"step": st["step"], "samples": st["samples"], "games_generated": st["games_generated"],
                   "train_ce": float(agg["pl"]) / n_log, "train_value_ce": float(agg["vl"]) / n_log,
                   "samples_per_s": n_log * a.batch_size / dt, "lr": lr_now(), "buffer": buf.size,
                   "reuse": (st["samples"] - samples0) / max(1, buf.inserted)}
            if mixed:
                msg["buffer_4p_share"] = float((buf.nplayers[:buf.size] == 4).mean())
                msg["stage"] = st["stage"]
            print(f"step {st['step']:7d} | CE {msg['train_ce']:.4f} | V-CE {msg['train_value_ce']:.4f} | "
                  f"{msg['samples_per_s']:,.0f} samples/s | teacher games {st['games_generated']:,} | "
                  f"reuse {msg['reuse']:.2f} | lr {msg['lr']:.1e}"
                  + (f" | stage {st['stage']} 4P {msg['buffer_4p_share']:.2f}" if mixed else ""), flush=True)
            write_json_atomic(status_path, {"phase": "sl", **msg, "state": st, "updated": time.time()})
            agg = {"pl": 0.0, "vl": 0.0}; n_log = 0; t_log = time.time()
            if dev.type == "mps" and st["step"] % 2000 == 0:
                torch.mps.empty_cache()

        if time.time() - t_save > a.save_every_min * 60:
            save(); t_save = time.time()

        if st["samples"] >= next_check:
            next_check += a.check_every
            st["checks"] += 1
            m = validate(net, val, dev)
            if mixed:
                m.update({k.replace("val_", "val4_"): v for k, v in validate(net, val4, dev).items()})
            if a.strength_every and st["checks"] % a.strength_every == 0:
                opps = ["mcts3000", "v15.2"] if not mixed else ["mcts3000"] + a.strength4_opponents.split(",")
                m["strength"] = strength(net, a.strength_games, run, st["step"], opps)
            best = st["best"]
            last_stage = st["stage"] >= len(stages) - 1
            # Which metrics count: 2P only (original), 4P only (early mixed stages), both (last mixed stage).
            prefixes = ["val_"] if not mixed else (["val_", "val4_"] if last_stage else ["val4_"])
            improved = []
            for pre in prefixes:
                tag = "" if pre == "val_" else "4"
                ce, t1, vce = pre + "ce", pre + "top1", pre + "value_ce"
                if ce not in best or (best[ce] - m[ce]) / best[ce] > a.ce_tol:
                    improved.append("ce" + tag)
                if t1 not in best or m[t1] - best[t1] > a.top1_tol:
                    improved.append("top1" + tag)
                if vce not in best or (best[vce] - m[vce]) / best[vce] > a.ce_tol:
                    improved.append("value" + tag)
            if "strength" in m:
                checks = [("mcts3000", "mcts_wr")] if "val_" in prefixes else []
                if mixed and "4p:teacher" in m["strength"]:
                    checks.append(("4p:teacher", "teacher4_wr"))
                for opp, key in checks:
                    w, se = m["strength"][opp]["win_rate"], m["strength"][opp]["stderr"]
                    if key not in best or w > best[key] + 2 * se:
                        improved.append("strength" + ("4" if key == "teacher4_wr" else ""))
                    best[key] = max(best.get(key, 0.0), w)
            for pre in (["val_", "val4_"] if mixed else ["val_"]):
                for k in (pre + "ce", pre + "value_ce"):
                    best[k] = min(best.get(k, math.inf), m[k])
                best[pre + "top1"] = max(best.get(pre + "top1", 0.0), m[pre + "top1"])
            st["bad_checks"] = 0 if improved else st["bad_checks"] + 1
            decision = "improving" if improved else f"no improvement ({st['bad_checks']}/{a.patience})"
            enough = st["games_generated"] >= a.min_games and st["positions_generated"] >= a.min_positions
            in_stage = st["positions_generated"] - st["stage_start_positions"]
            if mixed and not last_stage:
                if st["bad_checks"] >= a.patience and in_stage >= a.min_stage_positions:
                    st["stage"] += 1
                    st["bad_checks"] = 0
                    st["stage_start_positions"] = st["positions_generated"]
                    stream.set_frac4(stages[st["stage"]])
                    decision = f"4P plateau -> stage {st['stage']} (4P share {stages[st['stage']]})"
            elif st["bad_checks"] >= a.patience and enough:
                if st["drops"] < a.max_drops:
                    st["drops"] += 1
                    st["lr_scale"] *= a.lr_drop
                    st["bad_checks"] = 0
                    decision = f"plateau -> LR x{a.lr_drop} (drop {st['drops']}/{a.max_drops})"
                else:
                    st["done"] = True
                    decision = "plateau at lowest LR -> SL done"
            if st["games_generated"] >= a.max_games:
                st["done"] = True
                decision = f"max games ({a.max_games:,}) reached -> SL done"
            if a.max_positions and st["positions_generated"] >= a.max_positions:
                st["done"] = True
                decision = f"max positions ({a.max_positions:,}) reached -> SL done"
            rec = {"check": st["checks"], "step": st["step"], "samples": st["samples"],
                   "games_generated": st["games_generated"], "lr": lr_now(), "improved": improved,
                   "decision": decision, **m}
            if mixed:
                rec["stage"] = st["stage"]
                rec["frac4"] = stages[st["stage"]]
                rec["games4"], rec["positions4"] = st["games4"], st["positions4"]
                rec["positions_generated"] = st["positions_generated"]
            append_jsonl(metrics_path, rec)
            sm = ""
            if "strength" in m:
                sm = " | " + " ".join(f"vs {o} {r['win_rate']:.3f}" for o, r in m["strength"].items())
            print(f"[check {st['checks']}] CE {m['val_ce']:.4f} top1 {m['val_top1']:.3f} "
                  f"V-CE {m['val_value_ce']:.4f} V-acc {m['val_value_acc']:.3f}"
                  + (f" | 4P CE {m['val4_ce']:.4f} top1 {m['val4_top1']:.3f} V-CE {m['val4_value_ce']:.4f} "
                     f"V-acc {m['val4_value_acc']:.3f} | stage {st['stage']}" if mixed else "")
                  + f"{sm} -> {decision}", flush=True)
            save()
            if st["done"]:
                save(run / "sl_final.pt")
                write_json_atomic(status_path, {"phase": "sl", "done": True, "state": st, "updated": time.time()})
                break
    stream.stop()
    save()
    print("[exit] clean" + (" (SL done)" if st["done"] else ""), flush=True)


if __name__ == "__main__":
    main()
