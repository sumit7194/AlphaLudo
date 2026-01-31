"""
Parallel Game Generation for AlphaLudo Training.

Uses multiprocessing to run games in parallel across CPU cores.
Each worker process has its own model copy and plays games independently.
Results are collected and used for training.
"""

import os
import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process
import time
import ludo_cpp
from tensor_utils_mastery import state_to_tensor_mastery, get_board_coords, BOARD_SIZE
from vector_mcts import VectorMCTSMastery, get_action_probs_vector
from model_mastery import AlphaLudoTopNet
from training_utils import get_temperature


def worker_process(worker_id, model_path, games_per_worker, mcts_sims, 
                   result_queue, ghost_path=None, temp_schedule='alphazero'):
    """
    Worker process that plays games independently.
    
    Args:
        worker_id: Unique ID for this worker
        model_path: Path to model checkpoint
        games_per_worker: Number of games to play
        mcts_sims: MCTS simulations per move
        result_queue: Queue to send results back
        ghost_path: Optional ghost model path
        temp_schedule: Temperature schedule to use
    """
    try:
        # Load model in this process
        device = torch.device('cpu')  # Use CPU for multiprocessing
        
        checkpoint = torch.load(model_path, map_location=device)
        model = AlphaLudoTopNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load ghost if specified
        ghost_model = None
        ghost_name = None
        if ghost_path and os.path.exists(ghost_path):
            ghost_ckpt = torch.load(ghost_path, map_location=device)
            ghost_model = AlphaLudoTopNet()
            ghost_model.load_state_dict(ghost_ckpt['model_state_dict'])
            ghost_model.to(device)
            ghost_model.eval()
            ghost_name = os.path.basename(ghost_path).replace('.pt', '')
        
        # Initialize MCTS
        mcts = ludo_cpp.MCTSEngine(games_per_worker)
        
        # Play games
        all_examples = []
        results = []
        
        for game_idx in range(games_per_worker):
            state = ludo_cpp.create_initial_state()
            history = []
            move_count = 0
            
            # Determine if this is a ghost game (last 25%)
            is_ghost_game = ghost_model and (game_idx >= games_per_worker * 0.75)
            ghost_player = np.random.randint(0, 4) if is_ghost_game else -1
            
            identities = ['Main'] * 4
            if is_ghost_game:
                identities[ghost_player] = ghost_name or 'Ghost'
            
            while not state.is_terminal and move_count < 500:
                if state.current_dice_roll == 0:
                    state.current_dice_roll = np.random.randint(1, 7)
                
                current_player = state.current_player
                legal_moves = ludo_cpp.get_legal_moves(state)
                
                if len(legal_moves) == 0:
                    state.current_player = (state.current_player + 1) % 4
                    state.current_dice_roll = 0
                    continue
                
                # Get policy from appropriate model
                if is_ghost_game and current_player == ghost_player:
                    active_model = ghost_model
                else:
                    active_model = model
                
                # Simple policy (using NN only, no MCTS for parallel workers)
                state_tensor = state_to_tensor_mastery(state)  # Returns torch.Tensor
                state_np = state_tensor.numpy()  # Convert for serialization
                with torch.no_grad():
                    input_t = state_tensor.unsqueeze(0).float().to(device)
                    pi_logits, value = active_model(input_t)
                    probs = torch.softmax(pi_logits, dim=1).squeeze().cpu().numpy()
                
                # Get temperature based on move count
                temp = get_temperature(move_count, temp_schedule)
                
                # Sample action
                valid_probs = np.array([probs[m] for m in legal_moves])
                valid_probs = np.clip(valid_probs, 1e-8, None)
                
                if temp < 0.01:
                    action_idx = np.argmax(valid_probs)
                else:
                    valid_probs = valid_probs ** (1.0 / temp)
                    valid_probs /= valid_probs.sum()
                    action_idx = np.random.choice(len(legal_moves), p=valid_probs)
                
                action = legal_moves[action_idx]
                
                # Record for main model decisions only
                if identities[current_player] == 'Main':
                    positions = state.player_positions[current_player]
                    token_indices = []
                    for t in range(4):
                        r, c = get_board_coords(current_player, positions[t], t)
                        token_indices.append(r * BOARD_SIZE + c)
                    
                    history.append({
                        'state': state_np,  # Use numpy for serialization
                        'token_indices': token_indices,
                        'policy': probs,
                        'player': current_player
                    })
                
                # Apply move
                state = ludo_cpp.apply_move(state, action)
                move_count += 1
            
            # Game finished
            winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
            
            # Create training examples
            for record in history:
                p = record['player']
                if winner == -1:
                    val = 0.0
                elif winner == p:
                    val = 1.0
                else:
                    val = -1.0 / 3.0  # Partial negative for 4-player game
                
                all_examples.append((
                    record['state'],
                    np.array(record['token_indices'], dtype=np.int64),
                    record['policy'].astype(np.float32),
                    np.float32(val)
                ))
            
            results.append({
                'winner': winner,
                'identities': identities,
                'moves': move_count
            })
        
        # Send results back
        result_queue.put({
            'worker_id': worker_id,
            'examples': all_examples,
            'results': results,
            'games': games_per_worker
        })
        
    except Exception as e:
        result_queue.put({
            'worker_id': worker_id,
            'error': str(e),
            'examples': [],
            'results': []
        })


class ParallelGameGenerator:
    """
    Manages parallel game generation using multiple processes.
    """
    
    def __init__(self, model_path, num_workers=4, games_per_worker=2,
                 mcts_sims=200, ghost_pool=None, temp_schedule='alphazero'):
        """
        Args:
            model_path: Path to current model checkpoint
            num_workers: Number of parallel worker processes
            games_per_worker: Games each worker plays per batch
            mcts_sims: MCTS simulations per move
            ghost_pool: List of ghost model paths
            temp_schedule: Temperature schedule
        """
        self.model_path = model_path
        self.num_workers = num_workers
        self.games_per_worker = games_per_worker
        self.mcts_sims = mcts_sims
        self.ghost_pool = ghost_pool or []
        self.temp_schedule = temp_schedule
        
        print(f"ParallelGameGenerator: {num_workers} workers, {games_per_worker} games each")
    
    def update_model_path(self, path):
        """Update model path for next batch."""
        self.model_path = path
    
    def generate_batch(self):
        """
        Generate a batch of games using parallel workers.
        
        Returns:
            all_examples: List of (state, token_indices, policy, value) tuples
            results: List of game result dicts
        """
        result_queue = Queue()
        processes = []
        
        # Select ghost for this batch
        ghost_path = None
        if self.ghost_pool:
            ghost_path = np.random.choice(self.ghost_pool)
        
        start_time = time.time()
        
        # Spawn workers
        for worker_id in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(
                    worker_id,
                    self.model_path,
                    self.games_per_worker,
                    self.mcts_sims,
                    result_queue,
                    ghost_path,
                    self.temp_schedule
                )
            )
            p.start()
            processes.append(p)
        
        # Collect results
        all_examples = []
        all_results = []
        total_games = 0
        
        for _ in range(self.num_workers):
            result = result_queue.get(timeout=600)  # 10 min timeout
            
            if 'error' in result:
                print(f"Worker {result['worker_id']} error: {result['error']}")
                continue
            
            # Convert numpy arrays back to tensors
            for ex in result['examples']:
                state, token_idx, policy, value = ex
                all_examples.append((
                    torch.from_numpy(state).float(),
                    torch.from_numpy(token_idx).long(),
                    torch.from_numpy(policy).float(),
                    torch.tensor(value, dtype=torch.float32)
                ))
            
            all_results.extend(result['results'])
            total_games += result['games']
        
        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        duration = time.time() - start_time
        print(f"Generated {total_games} games in {duration:.1f}s ({duration/total_games:.1f}s/game)")
        
        return all_examples, all_results


def test_parallel():
    """Test parallel game generation."""
    # Check if model exists
    model_path = 'checkpoints_mastery/mastery_no6_v1/model_latest.pt'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    generator = ParallelGameGenerator(
        model_path=model_path,
        num_workers=4,
        games_per_worker=2,
        mcts_sims=50,  # Lower for testing
        temp_schedule='alphazero'
    )
    
    print("\nGenerating batch...")
    examples, results = generator.generate_batch()
    
    print(f"\nResults:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Total games: {len(results)}")
    
    # Show winners
    winners = [r['winner'] for r in results]
    print(f"  Winners: {winners}")


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    test_parallel()
