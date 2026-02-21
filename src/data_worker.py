
import time
import torch
import torch.multiprocessing as mp
import os
import sys

print(f"[DEBUG] Loading data_worker.py in PID {os.getpid()}...")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_v3 import AlphaLudoV3
from src.vector_league import VectorLeagueWorker
from src.async_shared import BATCH_SIZE, SYNC_INTERVAL
import src.config as cfg

def data_worker_loop(rank, data_queue, device='cpu', viz_queue=None, stop_requested=None, actor_done=None):
    """
    Actor Process Loop with graceful shutdown support.
    
    Args:
        rank: Actor ID
        data_queue: mp.Queue to push batches to.
        device: 'cpu' or 'mps' for Metal Performance Shaders
        viz_queue: mp.Queue for broadcasting (optional)
        stop_requested: mp.Event - when set, finish current batch and stop
        actor_done: mp.Event - set this when actor has finished saving and is ready to exit
    """
    print(f"[Actor {rank}] Initializing on {device}...")
    
    # Initialize Model - v3 with 4-action policy
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device)
    
    # Apply float16 for 2x speedup on M4 (if enabled)
    if cfg.USE_FLOAT16 and device != 'cpu':
        model = model.half()
        print(f"[Actor {rank}] Using float16 precision")
    
    model.eval()
    
    # Load initial weights if exist
    if os.path.exists(cfg.MAIN_CKPT_PATH):
        try:
            ckpt = torch.load(cfg.MAIN_CKPT_PATH, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"[Actor {rank}] Loaded initial checkpoint.")
        except Exception as e:
            print(f"[Actor {rank}] Failed load initial ckpt: {e}")
    else:
        print(f"[Actor {rank}] No checkpoint found, starting random.")
    
    # Probabilities (Loaded from Config)
    probabilities = cfg.GAME_COMPOSITION
    
    # Initialize Vector League
    # We pass None for elo_tracker for now to simplify, or load readonly.
    # Ideally Actors just play, Learner updates Elo. 
    # For MVP: Actors pick random or simple logic. Let's pass None.
    # Note: VectorLeagueWorker requires model.
    # Enable Visualization for ALL Actors
    visualize_actor = True
    
    # MOVED: Visualizer Server now runs in Learner process to ensure stability
    if visualize_actor and viz_queue:
        from src.visualizer import visualizer
        visualizer.set_ipc_queue(viz_queue)
        visualizer.set_actor_id(rank)
        # We don't start server here, but setting queue enables broadcast methods to push to it


    
    # Initialize VectorLeagueWorker with v3 MCTS parameters
    worker = VectorLeagueWorker(
        main_model=model,
        probabilities=probabilities,
        mcts_simulations=cfg.MCTS_SIMS,
        visualize=visualize_actor, 
        ghost_pool=[],
        elo_tracker=None, 
        temp_schedule='alphazero',
        c_puct=cfg.C_PUCT,
        dirichlet_alpha=cfg.DIRICHLET_ALPHA,
        dirichlet_eps=cfg.DIRICHLET_EPS,
        actor_id=rank
    )
    
    if visualize_actor:
        visualizer.set_worker(worker)
    
    last_sync_time = time.time()
    last_config_check = time.time()
    
    # Main loop - checks stop_requested Event for graceful shutdown
    while True:
        try:
            # Check for graceful shutdown request BEFORE starting new batch
            if stop_requested is not None and stop_requested.is_set():
                print(f"[Actor {rank}] Stop requested - finishing up...")
                break
            
            # Sync Weights periodically
            if time.time() - last_sync_time > SYNC_INTERVAL:
                if os.path.exists(cfg.MAIN_CKPT_PATH):
                    try:
                        ckpt = torch.load(cfg.MAIN_CKPT_PATH, map_location=device)
                        model.load_state_dict(ckpt['model_state_dict'])
                        last_sync_time = time.time()
                    except Exception as e:
                        print(f"[Actor {rank}] Sync failed: {e}")
            
            # 2. Reload Config (Dynamic Tuning)
            if time.time() - last_config_check > 60:
                reloaded = cfg.load_config_from_json()
                conf = cfg.CONFIGS[cfg.MODE]
                new_cpuct = conf.get("C_PUCT", 3.0)
                new_eps = conf.get("DIRICHLET_EPS", 0.25)
                new_probs = conf.get("GAME_COMPOSITION", None)
                
                # Always log current config state for diagnostics
                print(f"[Actor {rank}] Config Check: Reload={reloaded}, CPUCT={new_cpuct}, EPS={new_eps}")
                
                if reloaded:
                    worker.update_params(new_cpuct, new_eps, new_probs)
                last_config_check = time.time()

            # Play Batch - this may take several minutes
            # We complete the full batch even if stop is requested mid-batch
            start_t = time.time()
            examples, results, duration = worker.play_batch(batch_size=BATCH_SIZE, temperature=1.0)
            
            if len(examples) > 0:
                # Push to Queue
                data_queue.put({'examples': examples, 'results': results})
            
        except KeyboardInterrupt:
            print(f"[Actor {rank}] Interrupted")
            break
        except Exception as e:
            print(f"[Actor {rank}] Error: {e}")
            time.sleep(1)  # Prevent busy loop crash
    
    # Graceful exit - signal that we're done
    print(f"[Actor {rank}] Exiting gracefully.")
    if actor_done is not None:
        actor_done.set()
