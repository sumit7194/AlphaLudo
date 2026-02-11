#!/usr/bin/env python3
"""
Streaming Kickstart Pipeline v2
- Generates fresh data each cycle (~8GB)
- Trains for N epochs in-memory
- Overlaps generation with training
- Auto-cleanup after each cycle
- Graceful shutdown on SIGTERM/SIGINT
"""

import os
import sys
import time
import json
import signal
import pickle
import glob
import shutil
import subprocess
from threading import Thread, Event

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_v3 import AlphaLudoV3
from src.config import CONFIGS

# --- Configuration ---
SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SANDBOX_DIR, "model_kickstart.pt")
CHECKPOINT_PATH = os.path.join(SANDBOX_DIR, "checkpoint.json")
STATS_PATH = os.path.join(SANDBOX_DIR, "kickstart_stats.json")
GHOSTS_DIR = os.path.join(SANDBOX_DIR, "ghosts")

BUFFER_DIR_CURRENT = os.path.join(SANDBOX_DIR, "buffer_current")
BUFFER_DIR_STAGING = os.path.join(SANDBOX_DIR, "buffer_staging")

CHUNK_SIZE_GB = 8.0
EPOCHS_PER_CYCLE = 2
TOTAL_CYCLES = 200
BATCH_SIZE = 1024
LEARNING_RATE = CONFIGS["PROD"]["LEARNING_RATE"]
GHOST_SAVE_INTERVAL = 20  # Save ghost model every N cycles

# --- Graceful Shutdown ---
shutdown_event = Event()
save_requested = Event()

def signal_handler(signum, frame):
    print(f"\n⚠️  Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# --- Helper Functions ---

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return {"cycle": 0, "total_steps": 0}

def save_checkpoint(cycle, total_steps):
    ckpt = {
        "cycle": cycle,
        "total_steps": total_steps,
        "model_path": MODEL_PATH,
        "timestamp": time.time()
    }
    # Atomic write via temp file
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(ckpt, f)
    os.replace(tmp_path, CHECKPOINT_PATH)

def write_stats(step, p_loss, v_loss, sps, duration, cycle, epoch, total_samples_trained, buffer_gb):
    stats = {
        'step': step,
        'cycle': cycle,
        'epoch': epoch,
        'total_cycles': TOTAL_CYCLES,
        'epochs_per_cycle': EPOCHS_PER_CYCLE,
        'total_samples_trained': total_samples_trained,
        'policy_loss': p_loss,
        'value_loss': v_loss,
        'samples_per_sec': sps,
        'duration_sec': duration,
        'buffer_gb': buffer_gb,
        'timestamp': time.time()
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f)

def generate_buffer(output_dir, size_gb):
    """Run generate_data.py as subprocess."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, 
        os.path.join(SANDBOX_DIR, "generate_data.py"),
        "--output", output_dir,
        "--size_gb", str(size_gb)
    ]
    print(f"🔧 Starting buffer generation: {output_dir} ({size_gb}GB)")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc

def load_buffer_to_ram(buffer_dir):
    """Load all pickle shards into RAM."""
    samples = []
    files = glob.glob(os.path.join(buffer_dir, "*.pkl"))
    for f in files:
        with open(f, 'rb') as fp:
            chunk = pickle.load(fp)
            samples.extend(chunk)
    print(f"📦 Loaded {len(samples):,} samples into RAM")
    return samples

def cleanup_buffer(buffer_dir):
    """Delete buffer directory."""
    if os.path.exists(buffer_dir):
        shutil.rmtree(buffer_dir)
        print(f"🗑️  Cleaned up {buffer_dir}")

def save_model(model, path):
    """Atomic model save."""
    tmp_path = path + ".tmp"
    torch.save({'model_state_dict': model.state_dict()}, tmp_path)
    os.replace(tmp_path, path)
    print(f"💾 Model saved to {path}")

def save_ghost(model, cycle):
    """Save a ghost model snapshot for comparison."""
    os.makedirs(GHOSTS_DIR, exist_ok=True)
    ghost_path = os.path.join(GHOSTS_DIR, f"ghost_cycle_{cycle}.pt")
    torch.save({'model_state_dict': model.state_dict()}, ghost_path)
    print(f"👻 Ghost saved: {ghost_path}")

# --- Training Loop ---

def train_one_cycle(model, optimizer, samples, device, start_step, cycle):
    """Train for EPOCHS_PER_CYCLE on in-memory samples."""
    
    dataset = samples
    num_samples = len(dataset)
    num_batches = num_samples // BATCH_SIZE
    
    total_policy_loss = 0
    total_value_loss = 0
    step = start_step
    start_time = time.time()
    
    for epoch in range(EPOCHS_PER_CYCLE):
        np.random.shuffle(dataset)
        
        for batch_idx in range(num_batches):
            if shutdown_event.is_set():
                print("⏸️  Shutdown requested, saving and exiting...")
                save_model(model, MODEL_PATH)
                save_checkpoint(cycle, step)
                return step, True  # Interrupted
            
            # Extract batch
            batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            
            states = torch.stack([s[0] for s in batch]).to(device)
            target_pis = torch.stack([s[1] for s in batch]).to(device)
            target_vs = torch.stack([s[2] for s in batch]).to(device)
            
            # Forward
            pred_pis, pred_vs, _ = model(states)
            
            # Losses
            log_pis = torch.log(pred_pis + 1e-8)
            policy_loss = -torch.sum(target_pis * log_pis, dim=1).mean()
            value_loss = nn.MSELoss()(pred_vs, target_vs)
            loss = policy_loss + value_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            step += 1
            
            # Log every 50 steps
            if step % 50 == 0:
                duration = time.time() - start_time
                sps = step * BATCH_SIZE / duration
                avg_p = total_policy_loss / ((epoch * num_batches) + batch_idx + 1)
                avg_v = total_value_loss / ((epoch * num_batches) + batch_idx + 1)
                total_samples = step * BATCH_SIZE
                write_stats(step, avg_p, avg_v, sps, duration, cycle, epoch + 1, total_samples, CHUNK_SIZE_GB)
                print(f"  [Cycle {cycle} | Epoch {epoch+1}/{EPOCHS_PER_CYCLE}] "
                      f"Step {step} | P_Loss: {avg_p:.4f} | V_Loss: {avg_v:.4f} | "
                      f"Speed: {sps:.0f} sps")

    
    return step, False  # Completed normally

# --- Main Loop ---

def main():
    print("=" * 60)
    print("🚀 Streaming Kickstart Pipeline v2")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load checkpoint
    ckpt = load_checkpoint()
    start_cycle = ckpt["cycle"]
    total_steps = ckpt["total_steps"]
    print(f"Resuming from Cycle {start_cycle}, Step {total_steps}")
    
    # Check if training is already complete
    if start_cycle >= TOTAL_CYCLES:
        print("\n" + "=" * 60)
        print("✅ Training Already Complete!")
        print(f"   Total Cycles: {TOTAL_CYCLES}")
        print(f"   Total Steps: {total_steps}")
        print("=" * 60)
        print("\nTo restart from scratch, delete checkpoint.json and run again.")
        return
    
    # Ensure directories exist
    os.makedirs(BUFFER_DIR_CURRENT, exist_ok=True)
    os.makedirs(BUFFER_DIR_STAGING, exist_ok=True)
    
    # Pre-generate first buffer if starting fresh (blocking)
    if start_cycle == 0 and not glob.glob(os.path.join(BUFFER_DIR_CURRENT, "*.pkl")):
        print("📦 Generating initial buffer (this may take 1-2 minutes)...")
        proc = generate_buffer(BUFFER_DIR_CURRENT, CHUNK_SIZE_GB)
        proc.wait()  # Wait for initial generation to complete
        print("✅ Initial buffer ready!")
    
    staging_proc = None
    last_cycle = start_cycle  # Track last completed cycle
    
    for cycle in range(start_cycle, TOTAL_CYCLES):
        if shutdown_event.is_set():
            break
        
        last_cycle = cycle  # Update last completed cycle
            
        print(f"\n{'='*60}")
        print(f"🔄 CYCLE {cycle + 1}/{TOTAL_CYCLES}")
        print(f"{'='*60}")
        
        # Load current buffer into RAM
        samples = load_buffer_to_ram(BUFFER_DIR_CURRENT)
        
        if not samples:
            print("❌ No samples in current buffer! This should not happen.")
            break
        
        # Start generating next buffer in background (overlapping with training)
        cleanup_buffer(BUFFER_DIR_STAGING)
        os.makedirs(BUFFER_DIR_STAGING, exist_ok=True)
        staging_proc = generate_buffer(BUFFER_DIR_STAGING, CHUNK_SIZE_GB)
        
        # Train
        total_steps, interrupted = train_one_cycle(
            model, optimizer, samples, device, total_steps, cycle + 1
        )
        
        # Clear RAM after training
        del samples
        
        if interrupted:
            break
        
        # Save model and checkpoint after each cycle
        save_model(model, MODEL_PATH)
        save_checkpoint(cycle + 1, total_steps)
        
        # Save ghost model every N cycles for comparison
        if (cycle + 1) % GHOST_SAVE_INTERVAL == 0:
            save_ghost(model, cycle + 1)
        
        # Wait for staging to complete before swapping
        if staging_proc and staging_proc.poll() is None:
            print("⏳ Waiting for next buffer to finish generating...")
            staging_proc.wait()
        
        # Swap: staging -> current
        cleanup_buffer(BUFFER_DIR_CURRENT)
        if os.path.exists(BUFFER_DIR_STAGING):
            shutil.move(BUFFER_DIR_STAGING, BUFFER_DIR_CURRENT)
        else:
            print("⚠️  Staging buffer missing, regenerating...")
            os.makedirs(BUFFER_DIR_CURRENT, exist_ok=True)
            proc = generate_buffer(BUFFER_DIR_CURRENT, CHUNK_SIZE_GB)
            proc.wait()
    
    # Final save
    save_model(model, MODEL_PATH)
    save_checkpoint(last_cycle + 1, total_steps)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Total Cycles: {last_cycle + 1}")
    print(f"   Total Steps: {total_steps}")
    print("=" * 60)

if __name__ == "__main__":
    main()
