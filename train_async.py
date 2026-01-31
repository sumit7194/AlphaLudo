"""
Async Training Orchestrator for AlphaLudo

Manages Learner and Actor processes with dynamic spawning/stopping.
"""

import torch.multiprocessing as mp
import time
import sys
import os
import json
import argparse

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_worker import data_worker_loop
from src.learner import learner_loop
from src.async_shared import NUM_ACTORS, QUEUE_MAX_SIZE
from src.visualizer import visualizer

# Paths for inter-process communication
CMDS_DIR = "data/cmds"
STATS_FILE = "data/actor_stats.json"
STOP_SIGNAL = "data/stop_signal"

# Graceful Shutdown Timeouts
ACTOR_SHUTDOWN_TIMEOUT = 300  # 5 min - max time to wait for actor batch
LEARNER_SHUTDOWN_TIMEOUT = 120  # 2 min - max time for learner to save
QUEUE_DRAIN_TIMEOUT = 30  # 30 sec - max time to drain remaining queue


def write_actor_stats(actors: dict, learner_pid: int):
    """Write actor stats to JSON file for dashboard consumption."""
    stats = {
        'learner_pid': learner_pid,
        'actors': {
            str(actor_id): {
                'pid': info['process'].pid if info['process'].is_alive() else None,
                'device': info['device'],
                'status': 'running' if info['process'].is_alive() else 'stopped',
            }
            for actor_id, info in actors.items()
        },
        'active_count': sum(1 for a in actors.values() if a['process'].is_alive()),
        'timestamp': time.time()
    }
    
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
    except:
        pass


def check_commands(actors: dict, data_queue, next_actor_id: int, default_device: str, viz_queue=None, p_learner=None, learner_device='mps') -> tuple:
    """Check for spawn/kill commands from dashboard."""
    if not os.path.exists(CMDS_DIR):
        os.makedirs(CMDS_DIR, exist_ok=True)
        return next_actor_id
    
    for cmd_file in os.listdir(CMDS_DIR):
        cmd_path = os.path.join(CMDS_DIR, cmd_file)
        
        try:
            if cmd_file == 'spawn':
                # Spawn new actor
                actor_id = next_actor_id
                print(f"[Main] Spawning Actor {actor_id}...")
                p = mp.Process(target=data_worker_loop, args=(actor_id, data_queue, default_device, viz_queue))
                p.start()
                actors[actor_id] = {'process': p, 'device': default_device}
                print(f"[Main] Started Actor {actor_id} PID: {p.pid}")
                next_actor_id += 1
                os.remove(cmd_path)
                
                os.remove(cmd_path)

            elif cmd_file == 'kill_learner':
                if p_learner and p_learner.is_alive():
                    print("[Main] Stopping Learner (Graceful)...")
                    with open(STOP_SIGNAL, 'w') as f: f.write("STOP")
                    # Wait for learner to save and exit
                    p_learner.join(timeout=30)
                    if p_learner.is_alive():
                        print("[Main] Force killing Learner...")
                        p_learner.terminate()
                    print("[Main] Learner stopped.")
                    # Remove stop signal so validation doesn't kill next one immediately
                    if os.path.exists(STOP_SIGNAL): os.remove(STOP_SIGNAL)
                    p_learner = None
                os.remove(cmd_path)

            elif cmd_file == 'spawn_learner':
                if p_learner is None or not p_learner.is_alive():
                    print("[Main] Spawning New Learner...")
                    # Ensure no stop signal exists
                    if os.path.exists(STOP_SIGNAL): os.remove(STOP_SIGNAL)
                    p_learner = mp.Process(target=learner_loop, args=(data_queue, learner_device, viz_queue))
                    p_learner.start()
                    print(f"[Main] Started Learner PID: {p_learner.pid}")
                os.remove(cmd_path)
                
            elif cmd_file.startswith('kill_'):
                # Kill specific actor
                actor_id = int(cmd_file.split('_')[1])
                if actor_id in actors:
                    print(f"[Main] Killing Actor {actor_id}...")
                    actors[actor_id]['process'].terminate()
                    actors[actor_id]['process'].join(timeout=5)
                    del actors[actor_id]
                    print(f"[Main] Actor {actor_id} stopped.")
                os.remove(cmd_path)
                
        except Exception as e:
            print(f"[Main] Error processing command {cmd_file}: {e}")
            try:
                os.remove(cmd_path)
            except:
                pass
    
    return next_actor_id, p_learner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actors', type=int, default=NUM_ACTORS, help='Number of initial actor processes')
    parser.add_argument('--device', type=str, default='mps', help='Learner device (mps/cuda/cpu)')
    parser.add_argument('--actor-device', type=str, default='mps', help='Actor device (mps for M1/M2 GPUs, cpu for safety)')
    args = parser.parse_args()
    
    # Set Start Method (Spawn is safer for C++ ext/MPS)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"Starting Async Training with {args.actors} Actor(s) and 1 Learner...")
    
    # Ensure directories exist
    os.makedirs(CMDS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    
    # Clear old command files and stop signal
    for f in os.listdir(CMDS_DIR):
        os.remove(os.path.join(CMDS_DIR, f))
    if os.path.exists(STOP_SIGNAL):
        os.remove(STOP_SIGNAL)
    
    # === GRACEFUL SHUTDOWN EVENTS ===
    # These Events coordinate the shutdown sequence across all processes
    stop_requested = mp.Event()      # Signals actors to finish current batch and stop
    all_actors_done = mp.Event()     # Signals learner that all actors finished
    learner_done = mp.Event()        # Signals main that learner finished saving
    
    # Per-actor done events (dynamically allocated)
    max_actors = 16  # Max actors we'll ever have
    actor_done_events = {i: mp.Event() for i in range(max_actors)}
    
    # Shared Data Queue
    data_queue = mp.Queue(maxsize=QUEUE_MAX_SIZE)
    # IPC Queue for Visualization (Bridging Actors -> Dashboard Server)
    viz_queue = mp.Queue()
    
    # Actor registry
    actors = {}  # actor_id -> {'process': Process, 'device': str, 'done_event': Event}
    next_actor_id = 0
    
    # 1. Start Visualizer Server (Host in Main Process)
    visualizer.set_ipc_queue(viz_queue)
    visualizer.start_server(port=8765)
    
    # 2. Start Learner (pass graceful shutdown events)
    p_learner = mp.Process(
        target=learner_loop, 
        args=(data_queue, args.device, viz_queue, all_actors_done, learner_done)
    )
    p_learner.start()
    print(f"Started Learner PID: {p_learner.pid}")
    
    # 3. Start Initial Actors (pass graceful shutdown events)
    for i in range(args.actors):
        time.sleep(0.5)  # Stagger start
        actor_done = actor_done_events[next_actor_id]
        p_actor = mp.Process(
            target=data_worker_loop, 
            args=(next_actor_id, data_queue, args.actor_device, viz_queue, stop_requested, actor_done)
        )
        p_actor.start()
        actors[next_actor_id] = {'process': p_actor, 'device': args.actor_device, 'done_event': actor_done}
        print(f"Started Actor {next_actor_id} PID: {p_actor.pid}")
        next_actor_id += 1
    
    # 4. Main Loop
    last_stats_time = 0
    shutdown_in_progress = False
    
    def graceful_shutdown():
        """Coordinated graceful shutdown sequence."""
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            return
        shutdown_in_progress = True
        
        print("\n" + "="*60)
        print("[Main] GRACEFUL SHUTDOWN INITIATED")
        print("="*60)
        
        # Step 1: Signal all actors to stop after current batch
        print("[Main] Step 1: Signaling actors to finish current batch...")
        stop_requested.set()
        
        # Step 2: Wait for each actor to signal done (with timeout)
        print(f"[Main] Step 2: Waiting for {len(actors)} actor(s) to finish...")
        actors_timed_out = []
        for actor_id, info in actors.items():
            done_event = info.get('done_event')
            if done_event:
                if done_event.wait(timeout=ACTOR_SHUTDOWN_TIMEOUT):
                    print(f"[Main]   ✓ Actor {actor_id} finished gracefully")
                else:
                    print(f"[Main]   ✗ Actor {actor_id} timed out (forcing termination)")
                    actors_timed_out.append(actor_id)
                    if info['process'].is_alive():
                        info['process'].terminate()
        
        # Step 3: Force terminate any remaining actors
        for actor_id, info in actors.items():
            if info['process'].is_alive():
                print(f"[Main]   Force terminating Actor {actor_id}")
                info['process'].terminate()
                info['process'].join(timeout=5)
        
        # Step 4: Signal learner that all actors are done
        print("[Main] Step 3: Signaling learner to save and exit...")
        all_actors_done.set()
        
        # Step 5: Wait for learner to finish saving
        if learner_done.wait(timeout=LEARNER_SHUTDOWN_TIMEOUT):
            print("[Main]   ✓ Learner saved and exited gracefully")
        else:
            print("[Main]   ✗ Learner timed out (forcing termination)")
            if p_learner and p_learner.is_alive():
                p_learner.terminate()
        
        # Step 6: Final cleanup
        if p_learner and p_learner.is_alive():
            p_learner.join(timeout=5)
        
        # Clean up signal file
        if os.path.exists(STOP_SIGNAL):
            os.remove(STOP_SIGNAL)
        
        print("="*60)
        print("[Main] GRACEFUL SHUTDOWN COMPLETE")
        print("="*60)
    
    try:
        while True:
            time.sleep(1)
            
            # Check for stop signal from dashboard
            if os.path.exists(STOP_SIGNAL):
                graceful_shutdown()
                break
            
            # Check for dashboard commands (spawn/kill actors)
            next_actor_id, p_learner = check_commands(
                actors, data_queue, next_actor_id, args.actor_device, 
                viz_queue, p_learner, args.device
            )
            
            # Write stats every 2 seconds
            if time.time() - last_stats_time > 2:
                write_actor_stats(actors, p_learner.pid if p_learner and p_learner.is_alive() else None)
                last_stats_time = time.time()
            
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C detected - initiating graceful shutdown...")
        graceful_shutdown()
        
    finally:
        # Force cleanup if graceful shutdown didn't run
        if not shutdown_in_progress:
            print("[Main] Force terminating all processes...")
            for actor_id, info in actors.items():
                if info['process'].is_alive():
                    info['process'].terminate()
                    info['process'].join(timeout=3)
            if p_learner and p_learner.is_alive():
                p_learner.terminate()
                p_learner.join(timeout=3)
        
        # Clean up stats file
        try:
            os.remove(STATS_FILE)
        except:
            pass
        
        print("[Main] Done.")


if __name__ == "__main__":
    main()
