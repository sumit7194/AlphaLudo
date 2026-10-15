"""AlphaLudo V15.3 — Supervised distillation of a Rust search teacher.

Trains a V15 GraphTransformer (history_len=1, 587,306 params) from teacher
shards written by `generate_sl_teacher` (formats v1 and v2).

REWRITTEN in the 2026-10-02 audit. What the first version got wrong:
  1. DOUBLE SOFTMAX. V15GraphTransformer.forward returns probabilities; the
     loss applied log_softmax to them again. Gradients were squashed (the
     "policy loss" could never fall much below log(n_legal) - small), and the
     logged loss / entropy / accuracy were all of the wrong distribution.
     -> forward_logits + cross-entropy on logits.
  2. VALUE LOSS. MSE between a sigmoid output in (0,1) and targets ±1: the
     optimum is "always 0 or 1" and -1 is unreachable. -> BCE-with-logits on
     (z + 1) / 2, unfinished games (z = 0) masked out.
  3. EVAL. 15 games per bot every 1000 steps, no three-six rule, model_best
     chosen by the max of that noise (0.883 "best" = 38% vs the champion).
     -> v153_eval.evaluate_suite: batched, rule-correct, head-to-head vs the
     v15.2 champion plus the champion's own Expert/Heuristic protocol and
     native Depth2, with common random numbers. model_best is chosen by the
     head-to-head score.
  4. NO RESUME, RARE SAVES. model_latest was written only after a full pass
     over every shard (hours) and never read back. -> atomic time-based saves
     (model, optimizer, step, counters, best, metrics) and --resume.
  5. NEW SHARDS IGNORED until a full pass ended. -> data is re-indexed every
     --round-steps steps, so new shards join within minutes.
  6. DISHONEST METRICS. Top-1 counted forced moves (one legal option); a
     hard-coded "Elo" table; games = states // 50. -> forced moves excluded
     (default), top-1 on multi-option decisions, real game counts from the
     data, no fabricated Elo.
  7. NO VALIDATION. -> the first --val-shards shards are held out (a
     game-level split) and scored every eval.
  8. Unlabeled teacher/opponent mix and missing scores in the input: fixed in
     sl_dataset_v153 (--movers, scores from positions).
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TD_LUDO_DIR = ROOT_DIR / "td_ludo"
PYTHON_DIR = ROOT_DIR / "python"
V15_ROOT = ROOT_DIR / "td_ludo_v15"
for p in (str(TD_LUDO_DIR), str(PYTHON_DIR), str(V15_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sl_dataset_v153 import MOVER_CHOICES, V153TeacherDataset, open_shard
from td_ludo_v15.models.v15 import V15GraphTransformer
from v153_eval import CHAMPION_PATH, evaluate_suite, load_v15_model


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTTP server
# ─────────────────────────────────────────────────────────────────────────────
class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, stats_path=None, metrics_path=None, landing="v13_dashboard.html", **kw):
        self._stats_path = stats_path
        self._metrics_path = metrics_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):
        return

    def do_GET(self):
        try:
            if self.path in ("/", ""):
                self.path = "/" + self._landing
                return super().do_GET()
            if self.path == "/api/stats":
                return self._serve_json_file(self._stats_path)
            if self.path == "/api/metrics":
                return self._serve_json_file(self._metrics_path)
            if self.path == "/api/elo":
                return self._send_json(b'{"rankings": [], "history": {}}')
            if self.path.startswith("/api/games"):
                return self._send_json(b'{"games": []}')
            if self.path == "/api/system":
                return self._serve_system()
            # Only the landing page and the /api routes are served (no static fallback:
            # the handler's directory is all of td_ludo/, including checkpoints and logs).
            self.send_response(404)
            self.end_headers()
        except (ConnectionResetError, BrokenPipeError):
            pass

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

    def _serve_system(self):
        try:
            import psutil
            payload = {
                "cpu_percent": float(psutil.cpu_percent(interval=None)),
                "memory_percent": float(psutil.virtual_memory().percent),
                "pid": os.getpid(),
            }
        except Exception:
            payload = {"cpu_percent": 0.0, "memory_percent": 0.0, "pid": os.getpid()}
        self._send_json(json.dumps(payload).encode())

    def _send_json(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def start_dashboard(port: int, stats_path: str, metrics_path: str, dashboard_dir: str):
    handler = functools.partial(
        DashboardHandler, directory=dashboard_dir, stats_path=stats_path, metrics_path=metrics_path,
    )
    try:
        server = HTTPServer(("127.0.0.1", port), handler)  # local only, like the other dashboards
    except OSError as e:
        print(f"[Dashboard] could not bind port {port}: {e} — continuing without dashboard", flush=True)
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Dashboard] Server live at http://localhost:{port}/v13_dashboard.html", flush=True)


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Losses / metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_losses(model, batch, device, value_coef: float):
    x = batch["x"].to(device, non_blocking=True)
    legal_mask = batch["legal_mask"].to(device, non_blocking=True)
    target_pi = batch["target_pi"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    z = batch["target_value"].to(device, non_blocking=True)
    legal_count = batch["legal_count"].to(device, non_blocking=True)

    logits, v_logit = model.forward_logits(x, legal_mask)  # illegal cells = -1e9
    log_p = F.log_softmax(logits, dim=-1)
    # Soft cross-entropy against the target distribution (one-hot or the
    # search's visit distribution). Illegal cells have target 0.
    policy_loss = -(target_pi * log_p).sum(dim=-1).mean()

    finished = z != 0
    v_target = (z + 1.0) * 0.5
    v_terms = F.binary_cross_entropy_with_logits(v_logit, v_target, reduction="none")
    value_loss = (v_terms * finished).sum() / finished.sum().clamp(min=1)

    total = policy_loss + value_coef * value_loss

    with torch.no_grad():
        p = log_p.exp()
        legal = legal_mask > 0
        entropy = -(torch.where(legal, p * log_p, torch.zeros_like(p))).sum(-1)
        multi = legal_count >= 2
        correct = (logits.argmax(-1) == target) & multi
        v_correct = ((v_logit > 0) == (z > 0)) & finished
        m = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy_sum": entropy[multi].sum(),
            "n_multi": multi.sum(),
            "correct": correct.sum(),
            "v_correct": v_correct.sum(),
            "n_finished": finished.sum(),
        }
    return total, m


@torch.no_grad()
def validate(model, loader, device, value_coef: float) -> dict:
    was = model.training
    model.train(False)
    agg = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_sum": 0.0, "n_multi": 0, "correct": 0,
           "v_correct": 0, "n_finished": 0}
    n_batches = 0
    for batch in loader:
        _, m = compute_losses(model, batch, device, value_coef)
        n_batches += 1
        for k in agg:
            agg[k] += float(m[k])
    model.train(was)
    nb = max(1, n_batches)
    return {
        "val_policy_ce": agg["policy_loss"] / nb,
        "val_value_bce": agg["value_loss"] / nb,
        "val_top1": agg["correct"] / max(1, agg["n_multi"]),
        "val_entropy": agg["entropy_sum"] / max(1, agg["n_multi"]),
        "val_value_acc": agg["v_correct"] / max(1, agg["n_finished"]),
    }


def count_games(shard_files) -> int:
    """Real game count: v2 from game_id, v1 from all-base game starts."""
    total = 0
    for sf in shard_files:
        try:
            version, recs = open_shard(sf)
        except ValueError:
            continue
        if len(recs) == 0:
            continue
        if version == 2:
            total += len(np.unique(recs["game_id"]))
        else:
            total += int(np.all(recs["positions"].reshape(len(recs), 16) == -1, axis=1).sum())
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=str, default="data/sl_teacher_v2")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/v15_3b")
    ap.add_argument("--resume", action="store_true", help="continue from <checkpoint-dir>/model_latest.pt")
    ap.add_argument("--init-from", type=str, default=None, help="initialise weights (not optimizer) from a checkpoint")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--total-steps", type=int, default=400_000, help="cosine horizon; training continues at min-lr after")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--movers", choices=MOVER_CHOICES, default="teacher")
    ap.add_argument("--keep-forced", action="store_true", help="train on single-legal-move records too")
    ap.add_argument("--policy-target", choices=("pi", "action"), default="pi")
    ap.add_argument("--round-steps", type=int, default=2000, help="steps between data re-indexing (new shards)")
    ap.add_argument("--val-shards", type=int, default=1, help="first N shards held out (game-level split)")
    ap.add_argument("--val-records", type=int, default=20_000)
    ap.add_argument("--eval-every-steps", type=int, default=5000)
    ap.add_argument("--eval-games-champion", type=int, default=400)
    ap.add_argument("--eval-games-bots", type=int, default=400)
    ap.add_argument("--eval-games-depth2", type=int, default=200)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = run until stop file)")
    args = ap.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}", flush=True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ckpt_dir / "stats.json"
    metrics_path = ckpt_dir / "metrics.json"
    latest_path = ckpt_dir / "model_latest.pt"
    best_path = ckpt_dir / "model_best.pt"
    stop_files = [ROOT_DIR / "stop", ckpt_dir / "stop"]
    shard_dir = Path(args.shard_dir)

    model = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    states_trained = 0
    best_h2h = -1.0
    metrics_history = []
    if args.resume and latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = int(ck["step"])
        states_trained = int(ck["states"])
        best_h2h = float(ck.get("best_h2h", -1.0))
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics_history = json.load(f)
        print(f"[Resume] step {step:,}, {states_trained:,} states, best h2h {best_h2h:.3f}", flush=True)
    elif args.resume:
        print(f"[Resume] no {latest_path} — starting fresh", flush=True)
    elif latest_path.exists():
        raise SystemExit(f"{latest_path} exists. Pass --resume to continue it, or use a new --checkpoint-dir.")
    if args.init_from and step == 0:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("model_state_dict", ck))
        print(f"[Init] weights from {args.init_from}", flush=True)

    champion = load_v15_model(CHAMPION_PATH, device) if args.eval_games_champion > 0 else None

    def lr_at(s: int) -> float:
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        t = min(1.0, (s - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * t))

    run_info = {
        "model_name": "AlphaLudo V15.3b (search-distilled GraphTransformer)",
        "badge": f"teacher shards: {shard_dir.name} · movers={args.movers}",
        "arch_summary": f"V15 GraphTransformer 4x128 · {n_params:,} params · zero search",
        "description": "Eval WR = champion protocol (Expert + Heuristic, champion scored 0.8465). "
                       "Head-to-head vs v15.2 champion is the 'vs Champion' line (0.50 = parity).",
        "target_wr": 0.8465,
        "target_label": "v15.2 champion on this protocol (84.65%)",
        "baseline_wr": 0.50,
        "baseline_label": "Parity",
        "eval_games": args.eval_games_bots,
    }
    stats = {"run_info": run_info, "timestamp": time.time(), "total_games": 0, "total_states": states_trained,
             "games_per_minute": 0.0, "states_per_minute": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
             "total_loss": 0.0, "policy_entropy": 0.0, "temperature": 0.0, "main_elo": None,
             "eval_win_rate": metrics_history[-1]["eval_win_rate"] if metrics_history else 0.0,
             "opp_breakdown": {}, "recent_opponent_stats": {}, "elo_rankings": []}
    write_json_atomic(stats_path, stats)
    start_dashboard(args.port, str(stats_path), str(metrics_path), str(TD_LUDO_DIR))

    print("=" * 80)
    print(f"V15.3b SL | {n_params:,} params | shards {shard_dir} | movers={args.movers} "
          f"| forced={'kept' if args.keep_forced else 'skipped'} | target={args.policy_target}")
    print("=" * 80, flush=True)

    def save_latest(reason: str):
        save_atomic({
            "step": step, "states": states_trained, "best_h2h": best_h2h,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args), "saved_at": time.time(),
        }, latest_path)
        print(f"[Save] model_latest.pt @ step {step:,} ({reason})", flush=True)

    def stop_requested() -> bool:
        return any(p.exists() for p in stop_files)

    val_loader = None
    val_files: list = []
    t_session = time.time()
    states_session = 0
    last_save = time.time()
    done = False

    while not done:
        if stop_requested():
            break
        all_files = sorted(shard_dir.glob("shard_*.bin"))
        if len(all_files) <= args.val_shards:
            print(f"[Waiting] {len(all_files)} shard(s) in {shard_dir}; need > {args.val_shards}", flush=True)
            time.sleep(30)
            continue

        if val_loader is None:
            val_files = all_files[: args.val_shards]
            val_ds = V153TeacherDataset(shard_files=val_files, movers=args.movers,
                                        skip_forced=not args.keep_forced, policy_target=args.policy_target)
            rng = np.random.default_rng(0)
            pick = rng.choice(len(val_ds), size=min(args.val_records, len(val_ds)), replace=False)
            n_keep = (len(pick) // args.batch_size) * args.batch_size
            val_loader = DataLoader(Subset(val_ds, np.sort(pick[:n_keep]).tolist()), batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, drop_last=True)
            print(f"[Val] held out {[f.name for f in val_files]}: {n_keep:,} records", flush=True)

        train_files = all_files[args.val_shards:]
        train_ds = V153TeacherDataset(shard_files=train_files, movers=args.movers,
                                      skip_forced=not args.keep_forced, policy_target=args.policy_target)
        n_games = count_games(train_files)
        sampler = RandomSampler(train_ds, replacement=False,
                                num_samples=min(len(train_ds), args.round_steps * args.batch_size))
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            drop_last=True, persistent_workers=False)
        print(f"[Data] {len(train_files)} train shards · {n_games:,} games · {len(train_ds):,} records "
              f"(of {train_ds.n_raw:,})", flush=True)

        acc = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_sum": 0.0, "n_multi": 0, "correct": 0,
               "v_correct": 0, "n_finished": 0, "batches": 0}
        t_log = time.time()

        for batch in loader:
            if stop_requested():
                done = True
                break
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            loss, m = compute_losses(model, batch, device, args.value_coef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            bs = batch["x"].shape[0]
            states_trained += bs
            states_session += bs
            acc["batches"] += 1
            for k in ("policy_loss", "value_loss", "entropy_sum", "n_multi", "correct", "v_correct", "n_finished"):
                acc[k] += m[k]  # tensors; synced only at log time (no per-step .item() stalls)

            if step % args.log_every == 0:
                nb = acc["batches"]
                pl = float(acc["policy_loss"]) / nb
                vl = float(acc["value_loss"]) / nb
                ent = float(acc["entropy_sum"]) / max(1, int(acc["n_multi"]))
                top1 = int(acc["correct"]) / max(1, int(acc["n_multi"]))
                vacc = int(acc["v_correct"]) / max(1, int(acc["n_finished"]))
                sps = nb * args.batch_size / max(1e-6, time.time() - t_log)
                print(f"Step {step:7d} | lr {lr_at(step):.2e} | CE {pl:.4f} | top1 {top1:6.1%} | H {ent:.3f} "
                      f"| V-BCE {vl:.4f} | V-acc {vacc:5.1%} | {sps:,.0f} st/s", flush=True)
                elapsed_min = max(1e-6, (time.time() - t_session) / 60)
                stats.update({
                    "timestamp": time.time(), "total_games": n_games, "total_states": states_trained,
                    "states_per_minute": states_session / elapsed_min,
                    "games_per_minute": 0.0,
                    "policy_loss": pl, "value_loss": vl, "total_loss": pl + args.value_coef * vl,
                    "policy_entropy": ent, "train_top1": top1, "train_value_acc": vacc,
                    "lr": lr_at(step), "step": step,
                })
                write_json_atomic(stats_path, stats)
                acc = {k: 0 if k != "policy_loss" else 0.0 for k in acc}
                t_log = time.time()

            if time.time() - last_save > args.save_every_min * 60:
                save_latest("timer")
                last_save = time.time()

            if step % args.eval_every_steps == 0:
                print(f"\n[Eval @ step {step:,}]", flush=True)
                t_e = time.time()
                val = validate(model, val_loader, device, args.value_coef)
                suite = evaluate_suite(model, device, champion, args.eval_games_champion,
                                       args.eval_games_bots, args.eval_games_depth2)
                ex, he = suite.get("Expert"), suite.get("Heuristic")
                if ex and he:
                    protocol_wr = (ex["wins"] + he["wins"]) / max(1, ex["games"] + he["games"])
                else:
                    protocol_wr = None
                h2h = suite.get("Champion_v15_2", {}).get("win_rate")
                line = " | ".join(f"{k} {r['win_rate']:.3f}±{r['stderr']:.3f}" for k, r in suite.items())
                print(f"[Val] CE {val['val_policy_ce']:.4f} | top1 {val['val_top1']:.1%} | H {val['val_entropy']:.3f} "
                      f"| V-BCE {val['val_value_bce']:.4f} | V-acc {val['val_value_acc']:.1%}", flush=True)
                print(f"[Eval] {line} | protocol {protocol_wr} | {time.time() - t_e:.0f}s\n", flush=True)

                opponents = {"Champion": h2h, "Expert": ex and ex["win_rate"], "Heuristic": he and he["win_rate"],
                             "Depth2": suite.get("Depth2", {}).get("win_rate")}
                entry = {"step": step, "games": n_games, "states": states_trained,
                         "eval_win_rate": protocol_wr, "h2h_champion": h2h, "opponents": opponents,
                         "suite": suite, **val, "time": time.time()}
                metrics_history.append(entry)
                write_json_atomic(metrics_path, metrics_history)
                stats.update({
                    "eval_win_rate": protocol_wr, "opp_breakdown": opponents,
                    "recent_opponent_stats": {k: {"win_rate": 100 * r["win_rate"], "wins": r["wins"],
                                                  "games": r["games"]} for k, r in suite.items()},
                    **val,
                })
                write_json_atomic(stats_path, stats)

                if h2h is not None and h2h > best_h2h:
                    best_h2h = h2h
                    save_atomic({"step": step, "states": states_trained, "best_h2h": best_h2h, "suite": suite,
                                 "val": val, "model_state_dict": model.state_dict()}, best_path)
                    print(f"🌟 model_best.pt: h2h vs champion {h2h:.3f} @ step {step:,}", flush=True)
                save_latest("eval")
                last_save = time.time()

            if args.max_steps and step >= args.max_steps:
                done = True
                break

    save_latest("stop" if stop_requested() else "end")
    # The stop file is shared with the generator (it checks between shards),
    # so it is NOT deleted here; the launcher clears it on the next start.
    print("[Exit] clean", flush=True)


if __name__ == "__main__":
    main()
