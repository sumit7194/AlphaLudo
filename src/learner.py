
import time
import torch
import torch.multiprocessing as mp
import os
import sys
import numpy as np
from collections import deque 

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_v3 import AlphaLudoV3
from src.replay_buffer_mastery import ReplayBufferMastery
from src.train_v3 import TrainerV3
from src.async_shared import (
    SAMPLE_BATCH_SIZE, 
    TRAIN_STEPS_PER_LOOP
)
from src.config import (
    MAIN_CKPT_PATH, 
    CHECKPOINT_DIR, 
    WC_STATS_PATH, 
    ELOS_PATH, 
    GHOST_SAVE_FREQ,
    LEARNING_RATE,
    LR_WARMUP_STEPS,
    BUFFER_SIZE_LIMIT
)
from src.game_db import GameDB
import json

from src.training_utils import EloTracker

def save_wc_stats(total_games, fps, buffer_size=0, iteration=0):
    try:
        with open(WC_STATS_PATH, 'w') as f:
            json.dump({
                'total_games': total_games,
                'fps': fps,
                'buffer_size': buffer_size,
                'iteration': iteration
            }, f)
    except:
        pass

def learner_loop(data_queue, device_str='mps', viz_queue=None, all_actors_done=None, learner_done=None):
    """
    Learner Process with graceful shutdown support.
    
    Args:
        data_queue: mp.Queue to receive training data from actors
        device_str: Device for training ('mps', 'cuda', 'cpu')
        viz_queue: mp.Queue for visualization updates (optional)
        all_actors_done: mp.Event - when set, drain queue, save, and exit
        learner_done: mp.Event - set this when learner has finished saving
    """
    print(f"[Learner] Initializing on {device_str}...")
    device = torch.device(device_str)
    
    # Start Visualizer Client (Connect to Main Process via IPC)
    if viz_queue:
        from src.visualizer import visualizer
        visualizer.set_ipc_queue(viz_queue)
        # visualizer.start_server(port=8765) # Hosted by Main Process now
        print("[Learner] Visualizer IPC connected.")
    
    # Model - v3 with 4-action policy
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    trainer = TrainerV3(model, device, learning_rate=LEARNING_RATE, warmup_steps=LR_WARMUP_STEPS)
    
    # Load if exists
    if trainer.load_checkpoint(MAIN_CKPT_PATH):
        print(f"[Learner] Resumed from epoch {trainer.total_epochs}")
    else:
        print("[Learner] Starting fresh model.")
        trainer.save_checkpoint(MAIN_CKPT_PATH) # Initial save for actors
        
    # Replay Buffer
    replay_buffer = ReplayBufferMastery(max_size=BUFFER_SIZE_LIMIT)
    buffer_path = os.path.join(os.path.dirname(MAIN_CKPT_PATH), "replay_buffer.pkl")
    if replay_buffer.load(buffer_path):
        print(f"[Learner] Loaded Replay Buffer from {buffer_path}")
    else:
        print("[Learner] No Replay Buffer found, starting fresh.")
    
    # Stats & Elo
    elo_tracker = EloTracker(save_path=ELOS_PATH)
    win_history = deque(maxlen=100)
    game_durations = deque(maxlen=1000)
    iteration_times = deque(maxlen=100)
    last_iter_time = time.time()
    total_games = 0
    start_time = time.time()
    last_save_time = time.time()
    last_log_time = time.time()
    games_since_log = 0
    
    print("[Learner] Waiting for data...")
    
    avg_loss = 0.0 # Init
    
    try:
        while True:
            # 1. Drain Queue (Non-blocking or Short Block)
            # Try to fetch up to N batches
            fetched_batches = 0
            new_episodes_count = 0
            
            while not data_queue.empty() and fetched_batches < 10:
                try:
                    item = data_queue.get_nowait()
                    
                    # Handle Format: Dict or List (Legacy)
                    if isinstance(item, dict) and 'examples' in item:
                        examples = item['examples']
                        results = item.get('results', [])
                    else:
                        examples = item
                        results = []
                        
                    replay_buffer.add(examples)
                    total_games += len(examples) # Tracking transitions/samples
                    new_episodes_count += len(results)
                    
                    # Update Elo
                    if results:
                        elo_updated = False
                        for res in results:
                            winner = res['winner']
                            identities = res['identities']
                            if 'duration' in res:
                                game_durations.append(res['duration'])
                            
                            # Track Win Rate for Main
                            is_self_play = all(pid == 'Main' for pid in identities)
                            
                            if not is_self_play and 'Main' in identities:
                                if 0 <= winner < 4 and identities[winner] == 'Main':
                                     win_history.append(1)
                                     print(f"[Learner] Won Eval Game vs {identities}")
                                else:
                                     win_history.append(0)
                                     print(f"[Learner] Lost Eval Game vs {identities}")
                            elif 'Main' in identities:
                                # Self-Play
                                pass  
                            
                            elo_tracker.update_from_game(identities, winner, epoch=trainer.total_epochs)
                            elo_updated = True
                        
                        if elo_updated:
                            elo_tracker.save()
                    
                    fetched_batches += 1
                except Exception as e:
                    print(f"[Learner] Error processing batch: {e}")
                    break
            
            # Update Speed Tracker
            games_since_log += new_episodes_count

            # 2. Train if enough data
            if len(replay_buffer) > SAMPLE_BATCH_SIZE:
                 # Reduced log spam: Only print if buffer size changed significantly? 
                 # Actually, rely on the periodic status line.
                 avg_loss = 0
                 t_start_iter = time.time()
                 for _ in range(TRAIN_STEPS_PER_LOOP):
                     s, p, v = replay_buffer.sample(SAMPLE_BATCH_SIZE)
                     # v3 trainer returns (total_loss, policy_loss, value_loss, aux_loss)
                     l, pl, vl, al = trainer.train_step(s, p, v)
                     avg_loss += l
                 iteration_times.append(time.time() - t_start_iter)
                 
                 avg_loss /= TRAIN_STEPS_PER_LOOP
                 trainer.total_epochs += 1
            
            # 3. Periodically Save
            curr_time = time.time()
            if curr_time - last_save_time > 30: # Save every 30s
                trainer.save_checkpoint(MAIN_CKPT_PATH)
                last_save_time = curr_time
                
                # Save Ghost Checkpoint
                if trainer.total_epochs > 0 and trainer.total_epochs % GHOST_SAVE_FREQ == 0:
                     ghost_dir = os.path.join(os.path.dirname(MAIN_CKPT_PATH), "ghosts")
                     os.makedirs(ghost_dir, exist_ok=True)
                     ghost_path = os.path.join(ghost_dir, f"ghost_{trainer.total_epochs}.pt")
                     trainer.save_checkpoint(ghost_path)
                     print(f"[Learner] Saved Ghost: {os.path.basename(ghost_path)}")
            
            # 4. Logging
            if curr_time - last_log_time > 2.0: # 2s Interval
                duration = curr_time - last_log_time
                
                # Smooth Speed Calculation
                # We simply track games/s in this window. 
                # Ideally we'd use a moving average, but let's stick to window-based for responsiveness,
                # BUT since data is bursty, we might see 0.
                # Let's accumulate games_since_log over a slightly longer logical window or just show avg?
                
                # Improved: Use total games delta?
                # games_since_log tracks exact new episodes from results
                
                fps = games_since_log / duration
                
                # Simple smoothing
                if not hasattr(learner_loop, 'avg_fps'): learner_loop.avg_fps = 0.0
                learner_loop.avg_fps = 0.9 * learner_loop.avg_fps + 0.1 * fps
                
                main_elo = elo_tracker.get_rating('Main')
                print(f"[Learner] Buffer: {len(replay_buffer)} | Loss: {avg_loss:.4f} | Elo: {main_elo:.0f} | Speed: {learner_loop.avg_fps:.2f} games/s")
                
                # Save WC Stats
                save_wc_stats(total_games, learner_loop.avg_fps, len(replay_buffer), iteration=trainer.total_epochs)
                
                # Broadcast Metrics to Dashboard
                if viz_queue:
                    win_rate = (sum(win_history) / len(win_history)) if len(win_history) > 0 else 0.0
                    avg_game_time = (sum(game_durations) / len(game_durations)) if len(game_durations) > 0 else 0.0
                    avg_iter_time = (sum(iteration_times) / len(iteration_times)) if len(iteration_times) > 0 else 0.0
                    
                    try:
                        visualizer.broadcast_metrics(
                            iteration=trainer.total_epochs,
                            loss=float(avg_loss),
                            win_rate=win_rate,
                            avg_game_time=avg_game_time,
                            avg_iter_time=avg_iter_time
                        )
                        visualizer.broadcast_stats(total_games, len(replay_buffer), fps)
                    except Exception as e:
                        print(f"[Learner] Broadcast Error: {e}")
                
                last_log_time = curr_time
                games_since_log = 0
            
            # Avoid busy spin if queue empty
            if fetched_batches == 0:
                time.sleep(0.1)

            # 5. Check for graceful shutdown signal (Event-based)
            shutdown_triggered = False
            
            # Check Event (preferred)
            if all_actors_done is not None and all_actors_done.is_set():
                print("[Learner] All actors done - initiating graceful shutdown...")
                shutdown_triggered = True
            
            # Fallback: Check file (for backward compatibility)
            elif os.path.exists("data/stop_signal"):
                print("[Learner] Stop signal file detected - initiating graceful shutdown...")
                shutdown_triggered = True
            
            if shutdown_triggered:
                # Drain remaining items from queue
                print("[Learner] Draining remaining queue...")
                drain_count = 0
                drain_start = time.time()
                while not data_queue.empty() and (time.time() - drain_start) < 30:  # 30 sec max drain
                    try:
                        item = data_queue.get_nowait()
                        replay_buffer.add(item.get('examples', []))
                        drain_count += len(item.get('examples', []))
                    except:
                        break
                if drain_count > 0:
                    print(f"[Learner]   Drained {drain_count} additional samples")
                
                # Final save
                print("[Learner] Saving checkpoint and buffer...")
                trainer.save_checkpoint(MAIN_CKPT_PATH)
                if 'elo_tracker' in locals(): 
                    elo_tracker.save()
                replay_buffer.save(buffer_path)
                print("[Learner] Save complete!")
                
                # Signal that learner is done
                if learner_done is not None:
                    learner_done.set()
                break
                
    except KeyboardInterrupt:
        print("[Learner] Interrupted - saving...")
        trainer.save_checkpoint(MAIN_CKPT_PATH)
        if 'elo_tracker' in locals(): 
            elo_tracker.save()
        replay_buffer.save(buffer_path)
        if learner_done is not None:
            learner_done.set()

