
import asyncio
import websockets
import json
import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

try:
    import ludo_cpp
    from model_mastery import AlphaLudoTopNet
    from heuristic_bot import HeuristicLudoBot
    from tensor_utils_mastery import state_to_tensor_mastery
    from tensor_utils import get_board_coords
    from mcts_viz import MCTSVisualizer
    from trainer import Trainer
    from replay_buffer import ReplayBuffer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


# Constants
PORT = 8766
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class ManualRunner:
    def __init__(self):
        self.clients = set()
        self.step_event = asyncio.Event()
        self.state = ludo_cpp.create_initial_state()
        self.turn_count = 0
        self.log_history = []
        self.auto_run = False # Auto-Run Toggle
        
        # Load Agents
        print(f"Loading Agents on {DEVICE}...") 


        
        # Define Pool
        # Main Model
        self.main_model = AlphaLudoTopNet().to(DEVICE)
        ckpt_path = "checkpoints_mastery/mastery_no6_v1/model_latest.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            self.main_model.load_state_dict(ckpt['model_state_dict'])
            self.main_model.eval()
            print("  Main Model Loaded")
        else:
            print("  Main Model (Fresh Init - Warning!)")

        # Heuristic Bot
        self.heuristic = HeuristicLudoBot()
        print("  Heuristic Bot Loaded")

        # Ghost A
        self.ghost_a = AlphaLudoTopNet().to(DEVICE)
        print("  Ghost A Loaded")

        # Ghost B
        self.ghost_b = AlphaLudoTopNet().to(DEVICE)
        print("  Ghost B Loaded")
        
        # MCTS Visualizer
        self.mcts_viz = MCTSVisualizer(self.main_model, num_simulations=200, device=DEVICE)
        print("  MCTS Visualizer Ready (200 sims)")
        
        # Training Components (For "Learning" Demo)
        self.trainer = Trainer(self.main_model, learning_rate=0.001, device=DEVICE)
        self.replay_buffer = ReplayBuffer(max_size=1000)
        self.last_training_data = None # Store (state, policy, value) for re-training
        print("  Trainer & ReplayBuffer Initialized")

        # Shuffle Identities
        pool = ['Main', 'Heuristic', 'Ghost_A', 'Ghost_B']
        np.random.shuffle(pool)
        self.identities = pool
        print(f"Player Assignment: {self.identities}")

    async def broadcast(self, data):
        if not self.clients:
            return
        message = json.dumps(data)
        to_remove = set()
        for client in self.clients:
            try:
                await client.send(message)
            except:
                to_remove.add(client)
        self.clients -= to_remove

    def log(self, message, category='info'):
        entry = {'id': len(self.log_history), 'msg': message, 'cat': category}
        self.log_history.append(entry)
        print(f"[{category.upper()}] {message}")
        return entry

    async def run_game_loop(self):
        print("Waiting for client...")
        # Wait for at least one client
        while not self.clients:
            await asyncio.sleep(0.5)
        
        print("Client connected. Starting game loop.")
        self.log(f"Game Started. Assignment: {self.identities}", 'system')
        
        # Initial Roll
        roll = np.random.randint(1, 7)
        self.state.current_dice_roll = roll
        self.consecutive_sixes = 1 if roll == 6 else 0
        p = self.state.current_player
        self.log(f"Match Start: P{p} ({self.identities[p]}) Rolled {roll}", 'roll')
        await self.broadcast_full_state()

        while True:
            self.turn_count += 1
            p = self.state.current_player
            identity = self.identities[p]
            current_dice = self.state.current_dice_roll
            
            # --- STEP 1: Wait for User Approval to Move ---
            moves = ludo_cpp.get_legal_moves(self.state)
            
            label = f"P{p} ({identity}) {current_dice} -> "
            if len(moves) == 0:
                label += "SKIP"
            else:
                label += f"MOVE ({len(moves)} opts)"
            
            await self.wait_for_step(label)

            # --- STEP 2: Execute Action ---
            if len(moves) == 0:
                self.log(f"P{p} No Moves -> SKIP", 'skip')
                # Manual Advance
                self.state.current_player = (p + 1) % 4
                self.state.current_dice_roll = 0
                self.consecutive_sixes = 0 # Reset
                await self.broadcast_full_state()
            else:
                # Decide
                action = -1
                mcts_data = None
                
                if identity == 'Main':
                     action, mcts_data = self.get_model_action(self.main_model, moves, run_mcts=True)
                     self.log(f"P{p} (Main) Chose Token {action}", 'move')
                     
                elif identity == 'Heuristic':
                     action = self.heuristic.select_move(self.state, moves)
                     self.log(f"P{p} (Heuristic) Chose Token {action}", 'move')
                     
                     # Create One-Hot Target for Heuristic Move (Imitation Learning!)
                     # 1. Get Canonical Index
                     p_curr = self.state.current_player
                     pos = self.state.player_positions[p_curr][action]
                     r, c = get_board_coords(0, pos, action)
                     flat_idx = r * 15 + c
                     
                     policy_target = np.zeros(225, dtype=np.float32)
                     policy_target[flat_idx] = 1.0
                     
                     self.last_training_data = {
                        'state': state_to_tensor_mastery(self.state).clone(),
                        'policy': torch.tensor(policy_target),
                        'value': torch.tensor([0.0]) # Unknown value
                     }
                     
                elif identity == 'Ghost_A':
                     action, mcts_data = self.get_model_action(self.ghost_a, moves, run_mcts=True)
                     self.log(f"P{p} (Ghost A) Chose Token {action}", 'move')
                     
                elif identity == 'Ghost_B':
                     action, mcts_data = self.get_model_action(self.ghost_b, moves, run_mcts=True)
                     self.log(f"P{p} (Ghost B) Chose Token {action}", 'move')

                # Broadcast MCTS Visualization Data immediately if available
                if mcts_data:
                    await self.broadcast({
                        'type': 'mcts_overlay',
                        'data': mcts_data
                    })

                # Apply
                self.state = ludo_cpp.apply_move(self.state, action)
                
                # AUTO-TRAIN: Learn from this move immediately
                await self.trigger_training_step()
                
                # Check Win
                if self.state.is_terminal:
                    winner = ludo_cpp.get_winner(self.state)
                    self.log(f"GAME OVER! Winner: P{winner}", 'win')
                    await self.broadcast_full_state()
                    
                    await self.wait_for_step("Game Finished - Click to Reset")
                    
                    self.state = ludo_cpp.create_initial_state()
                    np.random.shuffle(self.identities) # Shuffle again
                    self.log(f"Game Reset. New Assignment: {self.identities}", 'system')
                    
                    # Roll for new game
                    roll = np.random.randint(1, 7)
                    self.state.current_dice_roll = roll
                    self.consecutive_sixes = 1 if roll == 6 else 0
                    await self.broadcast_full_state()
                    continue

            # --- STEP 3: Auto-Roll for Next Turn ---
            p_after = self.state.current_player
            next_id = self.identities[p_after]
            
            # Roll Logic
            roll = np.random.randint(1, 7)
            
            # RULE CHECK: Max 2 Consecutive Sixes
            # If same player (Bonus Turn) AND already has 2 sixes:
            if p_after == p and self.consecutive_sixes >= 2:
                # We must keep rolling until we get a non-6
                while roll == 6:
                     self.log(f"Bonus P{p_after} Rolled 6 -> IGNORED (Max 2 Rule)", 'roll')
                     roll = np.random.randint(1, 7)
                
                # We finally got a non-6
                self.log(f"Bonus P{p_after} Rolled {roll} (Accepted)", 'roll')
                self.consecutive_sixes = 0 # Sequence ends
                
            else:
                # Normal Roll processing
                if p_after == p:
                    # Bonus turn (1st or 2nd six)
                    if roll == 6:
                        self.consecutive_sixes += 1
                        self.log(f"BONUS! P{p_after} ({next_id}) Rolled {roll} (Six #{self.consecutive_sixes})", 'roll')
                    else:
                        self.consecutive_sixes = 0
                        self.log(f"BONUS! P{p_after} ({next_id}) Rolled {roll}", 'roll')
                else:
                    # New Player
                    self.consecutive_sixes = 1 if roll == 6 else 0
                    self.log(f"P{p_after} ({next_id}) Rolled {roll}", 'roll')

            self.state.current_dice_roll = roll
            await self.broadcast_full_state()

    def get_model_action(self, model, legal_moves, run_mcts=False):
        if run_mcts:
            # Update model reference and run
            self.mcts_viz.model = model 
            stats = self.mcts_viz.get_search_stats(self.state)
            
            # Pick best action by visits
            best_action = -1
            max_v = -1
            for child in stats['children']:
                if child['visits'] > max_v:
                    max_v = child['visits']
                    best_action = child['action']
            # Store Data for Potential Training
            # We construct a policy target from the Visit Counts
            policy_target = np.zeros(225, dtype=np.float32)
            
            # Map visits to spatial policy
            total_visits = sum(c['visits'] for c in stats['children'])
            if total_visits > 0:
                 for c in stats['children']:
                     # We need to map 'action' (token index) back to spatial pos
                     # But we don't have exact spatial map here easily without re-calculation
                     # wait, stats['children'] has 'pos_to' but that's raw board pos (0-51)
                     # We need the Canonical Flat Index (0-224) that the network outputs.
                     
                     # Re-calculate spatial for training target
                     p = self.state.current_player
                     t_idx = c['action']
                     pos = self.state.player_positions[p][t_idx] # Current pos (before move)
                     r, col = get_board_coords(0, pos, t_idx) # Canonical P0 frame
                     flat_idx = r * 15 + col
                     policy_target[flat_idx] = c['visits'] / total_visits
            
            # Store for "Train on Last Move"
            self.last_training_data = {
                'state': state_to_tensor_mastery(self.state).clone(), # Snapshot state BEFORE move
                'policy': torch.tensor(policy_target),
                'value': torch.tensor([0.0]) # Placeholder value, will be updated if we knew winner, but for single-step we might just use MCTS value?
                # Actually, MCTS stats has 'value_prior'? No.
                # We should use the Q-value of the chosen move? Or the root value?
                # For "Lesson", let's use the Q-value of the BEST action as the target value (optimistic).
            }
            
            return best_action, stats
        else:
            # Fallback to Raw Policy (Spatial Mapping)
            # Prepare tensor
            state_tensor = state_to_tensor_mastery(self.state)
            # Fix: state_tensor is already a Tensor, do not wrap in from_numpy
            t_batch = state_tensor.float().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pi, v = model(t_batch)
            
            pi_np = pi.exp().cpu().numpy()[0] # [225]
            
            # Spatial Masking Logic
            mask = np.zeros(225, dtype=np.float32)
            flat_to_token = {}
            
            # Current Player logic for Canonical View:
            # We pretend we are P0.
            for t_idx in legal_moves:
                # state.player_positions[p][t] is the path index (0-56).
                p = self.state.current_player
                pos = self.state.player_positions[p][t_idx]
                
                # Use P0 frame for canonical view
                r, c = get_board_coords(0, pos, t_idx) 
                flat_idx = r * 15 + c
                
                mask[flat_idx] = 1.0
                flat_to_token[flat_idx] = t_idx
                
            pi_masked = pi_np * mask
            
            if pi_masked.sum() > 0:
                pi_masked /= pi_masked.sum()
                chosen_flat = np.argmax(pi_masked)
                return int(flat_to_token[chosen_flat]), None
            else:
                # Fallback if model output zero prob for valid moves (unlikely but safe)
                # Return the first legal move and no MCTS data
                return legal_moves[0], None

    async def wait_for_step(self, Label):
        # Update UI with "Waiting for [Label]"
        await self.broadcast({
            'type': 'status',
            'status': 'waiting',
            'label': Label
        })
        self.step_event.clear()
        print(f"WAITING: {Label}")
        await self.step_event.wait()
        print("STEP RECEIVED")

    async def broadcast_full_state(self):
        # Serialize
        s_dict = {
            'player_positions': [[int(self.state.player_positions[i][j]) for j in range(4)] for i in range(4)],
            'scores': [int(self.state.scores[i]) for i in range(4)],
            'current_player': int(self.state.current_player),
            'current_dice_roll': int(self.state.current_dice_roll),
            'last_log': self.log_history[-1] if self.log_history else None,
            'log_history': self.log_history
        }
        await self.broadcast({
            'type': 'state',
            'state': s_dict,
            'identities': self.identities
        })

    async def wait_for_step(self, Label):
        # Update UI with "Waiting for [Label]"
        await self.broadcast({
            'type': 'status',
            'label': Label,
            'auto_run': self.auto_run
        })
        
        if self.auto_run:
             # Wait briefly so UI can render
             await asyncio.sleep(0.5)
             return

        self.step_event.clear()
        
        # Listen for messages while waiting
        while not self.step_event.is_set():
             await asyncio.sleep(0.1)
             if self.auto_run:
                 break

    async def trigger_training_step(self):
        if not self.last_training_data:
            print("[TRAIN] No training data available.", flush=True)
            return

        try:
            print("[TRAIN] Executing Single Training Step...", flush=True)
            
            # 1. Prepare Data
            state = self.last_training_data['state'].unsqueeze(0).to(DEVICE) # [1, 8, 15, 15]
            target_policy = self.last_training_data['policy'].unsqueeze(0).to(DEVICE) # [1, 225]
            target_value = self.last_training_data['value'].unsqueeze(0).to(DEVICE) # [1, 1]
            
            # 2. "Before" Prediction
            self.main_model.eval()
            with torch.no_grad():
                pi_before, v_before = self.main_model(state)
                pi_before = torch.exp(pi_before) # LogSoftmax -> Prob

            # 3. Train Step
            self.main_model.train()
            loss, p_loss, v_loss = self.trainer.train_step(state, target_policy, target_value)
            
            # 4. "After" Prediction
            self.main_model.eval()
            with torch.no_grad():
                pi_after, v_after = self.main_model(state)
                pi_after = torch.exp(pi_after)
                
            # 5. Broadcast Visualization
            print(f"[TRAIN] Broadcasting to {len(self.clients)} clients...", flush=True)
            await self.broadcast({
                'type': 'training_viz',
                'loss': loss,
                'policy_loss': p_loss,
                'value_loss': v_loss,
            })
            print(f"[TRAIN] Training Step Complete. Loss: {loss:.4f}", flush=True)
            
        except Exception as e:
            print(f"[TRAIN] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                if data['type'] == 'step':
                    self.step_event.set()
                elif data['type'] == 'toggle_auto':
                    self.auto_run = not self.auto_run
                    print(f"Auto-Run Toggled: {self.auto_run}")
                    if self.auto_run:
                        self.step_event.set()
                    
                    await self.broadcast({
                        'type': 'status',
                        'label': "Auto-Run Toggled",
                        'auto_run': self.auto_run
                    })
                elif data['type'] == 'train_step':
                     await self.trigger_training_step()
        finally:
            self.clients.remove(websocket)
            print("Client Left")

async def main():
    runner = ManualRunner()
    server = await websockets.serve(runner.handler, "localhost", PORT)
    print(f"Manual Runner Server on ws://localhost:{PORT}")
    
    await asyncio.gather(
        server.wait_closed(),
        runner.run_game_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
