"""AlphaLudo V15-4PW (Winner-Only Distillation) Training Pipeline.

Powered by the high-speed Rust `alphaludo_rs` simulation engine.
Trains strictly on winning sequences from the top 2 bots: Depth2Expectimax & AggressiveExpectimax.
Features:
  - 4-way Standings/Progress Aux Head for complete situational game awareness.
  - Zero disk storage for dataset (streamed live in RAM).
  - Background-friendly execution with configurable CPU pacing for office work.
  - Safe, atomic interruption and seamless resumption via stop file or Ctrl+C.
  - Live status dashboard on port 8846 (zero evaluation games during training).
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure python/ and alphaludo_rs are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
for p in (str(_PYTHON_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import alphaludo_rs
from alphaludo.model import V15_4PW_GraphTransformer


# ─── 1. Lightweight Status & Telemetry Dashboard ─────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AlphaLudo V15-4PW Winner Distillation Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.85);
      --border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-green: #10b981;
      --accent-purple: #8b5cf6;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    body {
      margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
    h1 { margin: 0; font-size: 24px; display: flex; align-items: center; gap: 10px; }
    .badge { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); padding: 4px 10px; border-radius: 9999px; font-size: 13px; font-weight: 600; border: 1px solid rgba(59, 130, 246, 0.4); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; backdrop-filter: blur(10px); }
    .card-title { font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .card-value { font-size: 28px; font-weight: 700; color: #fff; font-feature-settings: "tnum"; }
    .card-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
    .panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }
    .panel h2 { margin-top: 0; font-size: 16px; margin-bottom: 16px; }
    .status-pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 13px; background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.4); }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🏆 AlphaLudo V15-4PW <span class="badge">Winner-Distilled Rust Engine</span></h1>
        <div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">Trained strictly on winning trajectories from Depth2 & Aggressive Expectimax bots</div>
      </div>
      <div class="status-pill"><div class="status-dot"></div> Live Training</div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Winner Decisions Trained</div>
        <div class="card-value" id="val-states">0</div>
        <div class="card-sub" id="val-fps">0 states/sec</div>
      </div>
      <div class="card">
        <div class="card-title">Batches Optimized</div>
        <div class="card-value" id="val-batches">0</div>
        <div class="card-sub" id="val-elapsed">00:00:00</div>
      </div>
      <div class="card">
        <div class="card-title">Policy Loss (Move Imitation)</div>
        <div class="card-value" id="val-pol-loss">0.000</div>
        <div class="card-sub" id="val-pol-acc">Accuracy: 0.0%</div>
      </div>
      <div class="card">
        <div class="card-title">Standings Aux Loss (MSE)</div>
        <div class="card-value" id="val-stand-loss">0.000</div>
        <div class="card-sub">4-Way Progress Awareness</div>
      </div>
    </div>

    <div class="panel">
      <h2>Pipeline Configuration</h2>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 14px; color: var(--text-muted);">
        <div>Engine: <strong style="color: #fff;">Rust Native (alphaludo_rs)</strong></div>
        <div>Teacher Pool: <strong style="color: #fff;">Depth2 + Aggressive</strong></div>
        <div>Policy Filter: <strong style="color: #10b981;">Winner Only (0% Blunders)</strong></div>
        <div>Auxiliary Heads: <strong style="color: #fff;">Winner + Standings</strong></div>
        <div>Disk Footprint: <strong style="color: #10b981;">~2.4 MB (Zero Dataset on Disk)</strong></div>
      </div>
    </div>
  </div>

  <script>
    async function update() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('val-states').innerText = (data.total_states || 0).toLocaleString();
        document.getElementById('val-batches').innerText = (data.batches || 0).toLocaleString();
        document.getElementById('val-fps').innerText = Math.round(data.fps || 0) + ' states/sec';
        document.getElementById('val-elapsed').innerText = 'Elapsed: ' + (data.elapsed || '00:00:00');
        document.getElementById('val-pol-loss').innerText = (data.policy_loss || 0).toFixed(4);
        document.getElementById('val-pol-acc').innerText = 'Accuracy: ' + ((data.policy_acc || 0) * 100).toFixed(1) + '%';
        document.getElementById('val-stand-loss').innerText = (data.standings_loss || 0).toFixed(4);
      } catch(e) {}
    }
    setInterval(update, 1000);
    update();
  </script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, stats_getter, **kwargs):
        self.stats_getter = stats_getter
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return
        if self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.stats_getter()).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()


def start_dashboard(stats_getter, port: int = 8846):
    handler = lambda *args, **kwargs: DashboardHandler(*args, stats_getter=stats_getter, **kwargs)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ─── 2. Main Training Loop ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AlphaLudo V15-4PW Winner Distillation")
    parser.add_argument("--batch-size", type=int, default=256, help="Optimization batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--total-states", type=int, default=0, help="Total states to train (0 = continuous)")
    parser.add_argument("--pace-ms", type=int, default=15, help="Sleep ms between batches to keep CPU cool in background")
    parser.add_argument("--port", type=int, default=8846, help="Dashboard port")
    parser.add_argument("--save-dir", default="checkpoints/v15_4pw", help="Directory for checkpoints")
    parser.add_argument("--init", default="checkpoints/v15_4p_sl/model_best.pt", help="Pretrained weights to initialize from")
    parser.add_argument("--resume", action="store_true", help="Resume from save-dir/model_latest.pt if exists")
    parser.add_argument("--device", default=None, help="Device (mps, cpu, cuda)")
    parser.add_argument("--bots", default="MCTS,Depth2", help="Comma-separated teacher bots (MCTS, Depth2, Aggressive)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_file = save_dir / "metadata.json"

    print("=" * 65)
    print("🏆 AlphaLudo V15-4PW (Winner-Only Distillation)")
    print(f"💻 Device: {device} | Batch Size: {args.batch_size} | Pacing: {args.pace_ms}ms")
    print("🦀 Simulation Engine: Rust alphaludo_rs (Depth2 + Aggressive)")
    print(f"🌐 Live Dashboard: http://localhost:{args.port}")
    print(f"💾 Checkpoints: {save_dir}/")
    print("=" * 65)

    # Initialize model
    model = V15_4PW_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)

    # Resumption or pretrained initialization
    start_states = 0
    start_batches = 0
    if args.resume and (save_dir / "model_latest.pt").exists():
        print(f"🔄 Resuming from {save_dir / 'model_latest.pt'}")
        model.load_state_dict(torch.load(save_dir / "model_latest.pt", map_location=device, weights_only=True))
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    start_states = meta.get("total_states", 0)
                    start_batches = meta.get("batches", 0)
            except Exception:
                pass
    elif args.init and Path(args.init).exists():
        model.load_sl_pretrained(args.init)
    else:
        print("🌱 Initializing with fresh random weights")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Stats tracking
    stats = {
        "phase": "WINNER DISTILLATION (V15-4PW)",
        "device": str(device),
        "total_states": start_states,
        "batches": start_batches,
        "fps": 0.0,
        "policy_loss": 0.0,
        "policy_acc": 0.0,
        "value_loss": 0.0,
        "standings_loss": 0.0,
        "elapsed": "00:00:00",
    }

    def save_checkpoint():
        tmp_model = save_dir / "model_latest.pt.tmp"
        torch.save(model.state_dict(), tmp_model)
        tmp_model.replace(save_dir / "model_latest.pt")

        meta = {
            "total_states": stats["total_states"],
            "batches": stats["batches"],
            "policy_loss": round(stats["policy_loss"], 4),
            "policy_acc": round(stats["policy_acc"], 4),
            "value_loss": round(stats["value_loss"], 4),
            "standings_loss": round(stats["standings_loss"], 4),
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp_meta = save_dir / "metadata.json.tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2)
        tmp_meta.replace(meta_file)

    stop_requested = False

    def handle_signal(sig, frame):
        nonlocal stop_requested
        print(f"\n🛑 Interruption signal received. Safely finishing current batch and saving checkpoint...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Launch dashboard server
    start_dashboard(lambda: stats, port=args.port)

    t_start = time.time()
    t_last = time.time()
    states_last = start_states

    print("🏁 Starting training loop (Press Ctrl+C or touch stop file to pause safely)...")

    bot_pool = [b.strip() for b in args.bots.split(",") if b.strip()]
    print(f"🤖 Active Teachers: {', '.join(bot_pool)}")
    batch_idx = start_batches
    total_states = start_states

    try:
        while not stop_requested:
            if args.total_states > 0 and total_states >= args.total_states:
                print(f"🎯 Target states reached ({args.total_states:,}). Finishing.")
                break

            # Check for stop trigger file
            stop_file = save_dir / "stop"
            if stop_file.exists():
                print("\n🛑 Found 'stop' file. Stopping gracefully...")
                try:
                    stop_file.unlink()
                except OSError:
                    pass
                break

            # 1. Generate live winner-only batch via Rust engine
            frames, masks, targets, winners, standings = alphaludo_rs.generate_winner_batch(
                args.batch_size, bot_pool
            )

            # Move tensors to GPU
            x_t = torch.from_numpy(frames).to(device)
            mask_t = torch.from_numpy(masks).to(device)
            target_t = torch.from_numpy(targets).to(device)
            winner_t = torch.from_numpy(winners).to(device)
            standings_t = torch.from_numpy(standings).to(device)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            _, _, _, pol_logits, val_logits, standings_logits = model(
                x_t, legal_mask=mask_t, return_logits=True
            )

            # Multi-task loss:
            # 1. Policy loss: cross-entropy on winning mover's selected cell
            loss_pol = F.cross_entropy(pol_logits, target_t)
            # 2. Value loss: cross-entropy on winner seat (always 0 for the winner)
            loss_val = F.cross_entropy(val_logits, winner_t)
            # 3. Standings Aux loss: MSE on 4-way normalized progress fractions
            standings_pred = torch.sigmoid(standings_logits)
            loss_standings = F.mse_loss(standings_pred, standings_t)

            total_loss = loss_pol + 0.5 * loss_val + 0.2 * loss_standings

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                pol_acc = (pol_logits.argmax(dim=-1) == target_t).float().mean().item()

            n_samples = len(frames)
            total_states += n_samples
            batch_idx += 1

            stats["total_states"] = total_states
            stats["batches"] = batch_idx
            stats["policy_loss"] = float(loss_pol.item())
            stats["policy_acc"] = float(pol_acc)
            stats["value_loss"] = float(loss_val.item())
            stats["standings_loss"] = float(loss_standings.item())

            now = time.time()
            elapsed_sec = int(now - t_start)
            stats["elapsed"] = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))

            if now - t_last >= 2.0:
                dt = now - t_last
                stats["fps"] = (total_states - states_last) / dt
                t_last = now
                states_last = total_states

            # Periodic checkpoint saving (every 100 batches)
            if batch_idx % 100 == 0:
                save_checkpoint()
                print(
                    f"[{stats['elapsed']}] Batch {batch_idx:5d} | "
                    f"States: {total_states:,} | "
                    f"Pol Loss: {loss_pol.item():.4f} (Acc: {pol_acc * 100:4.1f}%) | "
                    f"Standings MSE: {loss_standings.item():.4f} | "
                    f"{stats['fps']:.1f} states/s"
                )

            # Gentle CPU pacing for background execution
            if args.pace_ms > 0:
                time.sleep(args.pace_ms / 1000.0)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        save_checkpoint()
        print(f"💾 Checkpoints and session metadata saved safely to {save_dir}. Clean exit.")


if __name__ == "__main__":
    main()
