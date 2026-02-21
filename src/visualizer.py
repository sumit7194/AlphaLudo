"""
Game Visualization Server

WebSocket server that streams game states to the browser visualizer.
"""

import asyncio
import json
import threading
import time
import os
from collections import deque
import ludo_cpp
import torch
from src.tensor_utils_mastery import state_to_tensor_mastery
from src.config import METRICS_PATH


try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Warning: websockets not installed. Run: pip install websockets")


class GameVisualizer:
    """
    Singleton class to broadcast game states to connected clients using Channels.
    Channels:
    - 'dashboard': global stats, elo, batch updates
    - 'game_{id}': specific game states, moves
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Channel Subscriptions: channel_name -> set(websockets)
        self.subscriptions = {} 
        self.clients = set() # Keep track of all connected clients for cleanup
        
        self.game_states = {} 
        self.state_queue = deque(maxlen=100)
        self.current_state = None
        self.server_thread = None
        self.loop = None
        self.dice_stats = [[0] * 7 for _ in range(4)]
        self.metrics_history = []
        self.identities = ['Main'] * 4 
        
        # New State Persistence
        self.actor_id = 0  # Default actor ID, overwritten by set_actor_id
        self.batch_size = 32
        self.game_results = {} # game_id -> winner
        self.ghost_game_ids = [] # List of game IDs with ghost players
        self.heuristic_game_ids = [] # List of game IDs with heuristic players
        self.game_identities = {} # game_id -> [identities list]
        self.game_identities = {} # game_id -> [identities list]
        self.latest_game_objects = {} # Store raw C++ state objects
        self.elo_history = {} # Persist ELO history
        
        # Persistence
        # Persistence
        self.metrics_path = METRICS_PATH
        self.actors_path = "data/actor_stats.json"
        
        # Ensure Checkpoints Dir Exists
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        
        self.load_metrics() # Load on startup
        
        # IPC Queue for bridging processes
        self.ipc_queue = None
        
        # Worker Reference for Bridging Commands
        self.worker_instance = None
        
        # Timing data for dashboards
        self.session_start_time = time.time()
        self.batch_start_time = time.time()
        self.game_start_times = {} # game_id -> start_time (epoch seconds)
        
        # Elo data for UI sync
        self.elo_data = {
            'main_elo': 1500,
            'ghost_name': None,
            'ghost_elo': 1500,
            'rankings': []
        }
        
        self.mcts_settings = {
            'early_termination_threshold': 0.90
        }

    def set_actor_id(self, actor_id):
        """Set the actor ID for tagging broadcasts."""
        self.actor_id = int(actor_id)

    def load_metrics(self):
        """Load metrics history from disk."""
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = data.get('history', [])
                    print(f"Loaded {len(self.metrics_history)} metrics from disk.")
            except Exception as e:
                print(f"Error loading metrics: {e}")

    def save_metrics(self):
        """Save metrics history to disk."""
        try:
            os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
            with open(self.metrics_path, 'w') as f:
                json.dump({'history': self.metrics_history}, f)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def start_server(self, host='localhost', port=8765):
        """Start WebSocket server in background thread."""
        
        # DEBUG: Log start attempt
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/viz_debug.log", "a") as f:
                f.write(f"[{time.time()}] Attempting to start server on {host}:{port}. Has Websockets: {HAS_WEBSOCKETS}\n")
        except: pass

        if not HAS_WEBSOCKETS:
            print("Cannot start visualizer: websockets not installed")
            return
        
        def run_server():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                
                async def handler(websocket):
                    self.clients.add(websocket)
                    try:
                        async for message in websocket:
                            data = json.loads(message)
                            msg_type = data.get('type')
                            
                            if msg_type == 'subscribe':
                                channel = data.get('channel')
                                if channel:
                                    if channel not in self.subscriptions:
                                        self.subscriptions[channel] = set()
                                    self.subscriptions[channel].add(websocket)
                                    print(f"Client subbed to {channel}")
                                    
                                    # Send Initial Data based on channel
                                    if channel == 'dashboard':
                                        await self._send_dashboard_init(websocket)
                                    elif channel.startswith('game_'):
                                        gid = int(channel.split('_')[1])
                                        await self._send_game_init(websocket, gid)
                            
                            elif msg_type == 'pause_game':
                                gid = data.get('game_id')
                                if self.worker_instance and gid is not None:
                                    self.worker_instance.pause_game(int(gid))
                                    
                            elif msg_type == 'resume_game':
                                gid = data.get('game_id')
                                if self.worker_instance and gid is not None:
                                    self.worker_instance.resume_game(int(gid))
                            
                            elif msg_type == 'resume_all':
                                if self.worker_instance:
                                    self.worker_instance.resume_all()
                                    
                            elif msg_type == 'get_state':
                                 # Legacy / Fallback
                                 gid = data.get('game_id', 0)
                                 aid = data.get('actor_id', 0)
                                 await self._send_game_init(websocket, gid, aid)

                            elif msg_type == 'stop_training':
                                print("Stop signal received from Dashboard.")
                                with open("data/stop_signal", "w") as f:
                                    f.write("STOP")
                                if self.worker_instance:
                                    self.worker_instance.stop_server()
                            
                            elif msg_type == 'spawn_actor':
                                # Write spawn command for main process
                                os.makedirs("data/cmds", exist_ok=True)
                                with open("data/cmds/spawn", "w") as f:
                                    f.write("spawn")
                                print("Spawn actor command sent.")
                            
                            elif msg_type == 'kill_actor':
                                actor_id = data.get('actor_id')
                                if actor_id is not None:
                                    os.makedirs("data/cmds", exist_ok=True)
                                    with open(f"data/cmds/kill_{actor_id}", "w") as f:
                                        f.write(str(actor_id))
                                    print(f"Kill actor {actor_id} command sent.")
                            
                            elif msg_type == 'get_debug':
                                mode = data.get('mode')
                                gid = data.get('game_id', 0)
                                
                                if mode == 'tensor':
                                    state = self.latest_game_objects.get(gid)
                                    if state:
                                        try:
                                            tensor = state_to_tensor_mastery(state) # (18, 15, 15)
                                            # Convert to nested list for JSON
                                            tensor_data = tensor.numpy().tolist()
                                            await websocket.send(json.dumps({
                                                'type': 'debug_data',
                                                'mode': 'tensor',
                                                'data': tensor_data
                                            }))
                                        except Exception as e:
                                            print(f"Tensor error: {e}")
                            
                            elif msg_type == 'update_mcts':
                                key = data.get('key')
                                value = data.get('value')
                                if key in self.mcts_settings and isinstance(value, (int, float)):
                                    self.mcts_settings[key] = float(value)
                                    await self._broadcast_to_channel('dashboard', json.dumps({
                                        'type': 'mcts_settings',
                                        'settings': self.mcts_settings
                                    }))
                                    
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    except Exception as e:
                        print(f"WS Handler Error: {e}")
                    finally:
                        self.clients.discard(websocket)
                        # Cleanup subscriptions
                        for channel in self.subscriptions:
                            self.subscriptions[channel].discard(websocket)

                async def serve():
                    try:
                        async with websockets.serve(handler, host, port):
                            with open("data/viz_debug.log", "a") as f:
                                f.write(f"[{time.time()}] Server bound successfully to {host}:{port}\n")
                            print(f"Visualizer server started at ws://{host}:{port}")
                            await asyncio.Future()  # run forever
                    except Exception as e:
                         with open("data/viz_debug.log", "a") as f:
                            f.write(f"[{time.time()}] Serve Error: {e}\n")
                         print(f"Serve Error: {e}")
                
                # --- Helper Loops (Defined Inside run_server to access broadcast/state) ---
                async def _file_watcher_loop():
                    """Periodically check for external file updates (Elo, Stats) from Learner."""
                    from src.async_shared import ELOS_PATH, WC_STATS_PATH
                    last_elo_mtime = 0
                    last_stats_mtime = 0
                    last_tuner_mtime = 0
                    
                    while True:
                        await asyncio.sleep(2)
                        
                        # 1. Check Elo Ratings
                        if os.path.exists(ELOS_PATH):
                            try:
                                mtime = os.path.getmtime(ELOS_PATH)
                                if mtime > last_elo_mtime:
                                    last_elo_mtime = mtime
                                    with open(ELOS_PATH, 'r') as f:
                                        elo_data_file = json.load(f)
                                    
                                    # Convert to broadcast format
                                    ratings = elo_data_file.get('ratings', {})
                                    history = elo_data_file.get('history', {})
                                    
                                    rankings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
                                    augmented_rankings = []
                                    for name, elo in rankings:
                                        augmented_rankings.append((name, elo, None))
                                    
                                    # Update internal state
                                    self.elo_data = {
                                        'main_elo': ratings.get('Main', 1500),
                                        'ghost_name': None, 
                                        'ghost_elo': None,
                                        'rankings': augmented_rankings[:10]
                                    }
                                    self.elo_history = history
                                    
                                    # Broadcast
                                    await self._broadcast_to_channel('dashboard', json.dumps({
                                        'type': 'elo_update', 
                                        **self.elo_data
                                    }))
                                    await self._broadcast_to_channel('dashboard', json.dumps({
                                        'type': 'elo_history', 
                                        'history': {k: list(v) for k, v in history.items()},
                                        'timestamp': time.time()
                                    }))
                            except Exception as e:
                                print(f"Elo watcher error: {e}")
                                
                        # 2. Check Training Stats
                        if os.path.exists(WC_STATS_PATH):
                            try:
                                mtime = os.path.getmtime(WC_STATS_PATH)
                                if mtime > last_stats_mtime:
                                    last_stats_mtime = mtime
                                    with open(WC_STATS_PATH, 'r') as f:
                                        stats_data = json.load(f)
                                        
                                    await self._broadcast_to_channel('dashboard', json.dumps({
                                        'type': 'stats',
                                        'total_games': stats_data.get('total_games', 0),
                                        'speed': stats_data.get('fps', 0.0),
                                        'buffer_size': stats_data.get('buffer_size', 0)
                                    }))
                            except Exception as e:
                                print(f"Stats watcher error: {e}")

                        # 3. Check Tuner History
                        tuner_path = 'data/tuner_history.json'
                        if os.path.exists(tuner_path):
                            try:
                                mtime = os.path.getmtime(tuner_path)
                                if mtime > last_tuner_mtime:
                                    last_tuner_mtime = mtime
                                    with open(tuner_path, 'r') as f:
                                        tuner_data = json.load(f)
                                    
                                    # Broadcast last 50 entries
                                    # Handle empty file case
                                    if tuner_data:
                                        await self._broadcast_to_channel('dashboard', json.dumps({
                                            'type': 'tuner_update',
                                            'history': tuner_data[-50:] 
                                        }))
                            except Exception as e:
                                print(f"Tuner watcher error: {e}")
                        try:
                            from src.resource_monitor import get_system_stats
                            
                            pid_list = [os.getpid()]
                            if os.path.exists('data/actor_stats.json'):
                                try:
                                    with open('data/actor_stats.json', 'r') as f:
                                        actor_data = json.load(f)
                                        l_pid = actor_data.get('learner_pid')
                                        if l_pid and l_pid not in pid_list: pid_list.append(l_pid)
                                        actors = actor_data.get('actors', {})
                                        for aid, info in actors.items():
                                            if info.get('pid'): pid_list.append(info['pid'])
                                except: pass
                                    
                            system_data = get_system_stats(pid_list)
                            
                            # Enrich
                            if os.path.exists('data/actor_stats.json') and 'processes' in system_data:
                                 try:
                                     with open('data/actor_stats.json', 'r') as f:
                                        actor_data = json.load(f)
                                        learner_pid = actor_data.get('learner_pid')
                                        label_map = {}
                                        if learner_pid: label_map[int(learner_pid)] = "Learner"
                                        for aid, info in actor_data.get('actors', {}).items():
                                            if info.get('pid'): label_map[int(info['pid'])] = f"Actor {aid}"
                                        
                                        for pid, p_stats in system_data['processes'].items():
                                            p_pid = int(pid)
                                            p_stats['pid'] = p_pid # Ensure pid is in the dict
                                            if p_pid in label_map: p_stats['label'] = label_map[p_pid]
                                            elif p_pid == os.getpid(): p_stats['label'] = "Viz/Learner"
                                 except: pass
                            
                            await self._broadcast_to_channel('dashboard', json.dumps({
                                'type': 'system_stats',
                                **system_data
                            }))

                            # 4. Periodic Actor Stats Broadcast (for real-time dashboard updates)
                            if os.path.exists('data/actor_stats.json'):
                                try:
                                    with open('data/actor_stats.json', 'r') as f:
                                        actor_data = json.load(f)
                                    await self._broadcast_to_channel('dashboard', json.dumps({
                                        'type': 'actor_stats',
                                        **actor_data
                                    }))
                                except: pass
                        
                        except Exception as e:
                            print(f"System stats broadcast error: {e}")

                self.loop.create_task(_file_watcher_loop())
                self.loop.create_task(self._process_ipc_loop()) # Start IPC bridge
                self.loop.run_until_complete(serve())
            
            except Exception as e:
                 with open("data/viz_debug.log", "a") as f:
                    f.write(f"[{time.time()}] Run_server Crash: {e}\n")
                 print(f"Run Server Crash: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
    async def _send_dashboard_init(self, websocket):
        """Send all data needed for dashboard initialization."""
        # 0. Session start time for timer
        await websocket.send(json.dumps({
            'type': 'timing_init',
            'session_start_time': self.session_start_time,
            'batch_start_time': self.batch_start_time,
            'game_start_times': self.game_start_times,
            'server_time': time.time()
        }))
        
        # 0b. Config Mode
        from src.config import MODE
        await websocket.send(json.dumps({
            'type': 'config_mode',
            'mode': MODE
        }))
        
        # 1. Batch Size
        await websocket.send(json.dumps({
            'type': 'batch_init', 
            'batch_size': self.batch_size,
            'batch_start_time': self.batch_start_time
        }))
        
        # 2. Ghost Games
        if self.ghost_game_ids:
            await websocket.send(json.dumps({'type': 'ghost_games', 'game_ids': self.ghost_game_ids}))
        
        # 2b. Heuristic Games
        if self.heuristic_game_ids:
            await websocket.send(json.dumps({'type': 'heuristic_games', 'game_ids': self.heuristic_game_ids}))
            
        # 3. Known Results
        for gid, winner in self.game_results.items():
            await websocket.send(json.dumps({'type': 'game_result', 'game_id': gid, 'winner': winner}))
            
        # 4. MCTS Settings
        await websocket.send(json.dumps({'type': 'mcts_settings', 'settings': self.mcts_settings}))
        
        # 5. ELO Update (Immediate Load from Disk)
        from src.async_shared import ELOS_PATH
        if os.path.exists(ELOS_PATH):
            try:
                with open(ELOS_PATH, 'r') as f:
                    elo_data_file = json.load(f)
                
                # Process Ratings
                ratings = elo_data_file.get('ratings', {})
                rankings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
                augmented_rankings = [(n, e, None) for n, e in rankings]
                
                await websocket.send(json.dumps({
                    'type': 'elo_update',
                    'main_elo': ratings.get('Main', 1500),
                    'rankings': augmented_rankings[:10]
                }))
                
                # Process History
                history = elo_data_file.get('history', {})
                if history:
                    await websocket.send(json.dumps({
                        'type': 'elo_history', 
                        'history': {k: list(v) for k, v in history.items()},
                        'timestamp': time.time()
                    }))
                    
            except Exception as e:
                print(f"Error loading Elo on init: {e}")
        
        # Fallback to memory if file failed or didn't exist
        if self.elo_data.get('rankings'):
             await websocket.send(json.dumps({'type': 'elo_update', **self.elo_data}))

        # 5b. Training Stats (Immediate Load)
        from src.async_shared import WC_STATS_PATH
        if os.path.exists(WC_STATS_PATH):
             try:
                 with open(WC_STATS_PATH, 'r') as f:
                     stats_data = json.load(f)
                 
                 # Send Stats (Games, Buffer)
                 await websocket.send(json.dumps({
                     'type': 'stats',
                     'total_games': stats_data.get('total_games', 0),
                     'speed': stats_data.get('fps', 0.0),
                     'buffer_size': stats_data.get('buffer_size', 0)
                 }))
                 
                 # Send Iteration (via Metrics dummy)
                 # UI expects metrics.history list
                 iteration = stats_data.get('iteration', 0)
                 if iteration > 0:
                     await websocket.send(json.dumps({
                         'type': 'metrics',
                         'history': [{
                             'iteration': iteration,
                             'loss': 0.0,    # Placeholder to avoid undefined
                             'win_rate': 0.0 # Placeholder
                         }]
                     }))
                     
             except Exception as e:
                 print(f"Error loading stats on init: {e}")

        # 6. Metrics (Memory is fine as it's loaded in __init__)
        if self.metrics_history:
            await websocket.send(json.dumps({'type': 'metrics', 'history': self.metrics_history}))
        
        # 7. Actor Stats (read from JSON file if exists)
        try:
            if os.path.exists('data/actor_stats.json'):
                with open('data/actor_stats.json', 'r') as f:
                    actor_data = json.load(f)
                await websocket.send(json.dumps({'type': 'actor_stats', **actor_data}))
        except Exception as e:
            pass
        
        # 8. System Stats (use resource_monitor if available)
        try:
            from src.resource_monitor import get_system_stats
            system_data = get_system_stats()
            await websocket.send(json.dumps({'type': 'system_stats', **system_data}))
        except Exception as e:
            pass

        # 9. Training Stats (Current)
        try:
             if os.path.exists(WC_STATS_PATH):
                 with open(WC_STATS_PATH, 'r') as f:
                     stats_data = json.load(f)
                 await websocket.send(json.dumps({
                     'type': 'stats',
                     'total_games': stats_data.get('total_games', 0),
                     'speed': stats_data.get('fps', 0.0),
                     'buffer_size': stats_data.get('buffer_size', 0)
                 }))
        except Exception as e:
             pass

    async def _send_game_init(self, websocket, gid, aid=None):
         """Send current state for a specific game."""
         state_to_send = self.game_states.get(gid)
         if state_to_send:
            await websocket.send(json.dumps({
                'type': 'state',
                'game_id': gid,
                'state': state_to_send,
                'dice_stats': self.dice_stats,
                'identities': self.game_identities.get(gid, self.identities),
                'game_start_time': self.game_start_times.get(gid, time.time()),
                'server_time': time.time()
            }))

    # --- Broadcasting Helpers ---

    def set_ipc_queue(self, queue):
        """Set shared queue for broadcasting across processes."""
        self.ipc_queue = queue

    async def _process_ipc_loop(self):
        """Consume messages from IPC queue and broadcast them."""
        import queue # standard lib
        while True:
            # Non-blocking get with sleep
            try:
                # We can't use await queue.get() because it's a multiprocessing queue
                # So we poll
                while self.ipc_queue and not self.ipc_queue.empty():
                    try:
                        channel, message = self.ipc_queue.get_nowait()
                        # print(f"DEBUG: Server received from IPC: {channel}")
                        
                        # CRITICAL FIX: Update Server State from IPC Message
                        try:
                            # We must parse to update internal state, even if overhead
                            data = json.loads(message)
                            msg_type = data.get('type')
                            
                            if msg_type == 'state':
                                gid = data.get('game_id')
                                state = data.get('state')
                                ids = data.get('identities')
                                if gid is not None and state:
                                    self.game_states[gid] = state
                                if gid is not None and ids:
                                    self.game_identities[gid] = ids
                                    
                            elif msg_type == 'batch_init':
                                self.batch_size = data.get('batch_size', 32)
                                self.batch_start_time = data.get('batch_start_time', time.time())
                                self.game_start_times = {int(k): v for k, v in data.get('game_start_times', {}).items()}
                                self.game_results.clear()
                                self.ghost_game_ids = []
                                self.heuristic_game_ids = []
                                
                            elif msg_type == 'game_result':
                                gid = data.get('game_id')
                                winner = data.get('winner')
                                if gid is not None:
                                    self.game_results[gid] = winner
                                    
                            elif msg_type == 'ghost_games':
                                self.ghost_game_ids = data.get('game_ids', [])
                                
                            elif msg_type == 'heuristic_games':
                                self.heuristic_game_ids = data.get('game_ids', [])
                                
                            elif msg_type == 'metrics':
                                # Append to history if new
                                # Note: This might duplicate if we blindly append. 
                                # But metrics are usually lists of history. 
                                # If message contains 'history' list, replace ours.
                                history = data.get('history')
                                if history:
                                    self.metrics_history = history
                                    
                        except Exception as e:
                            print(f"IPC State Update Error: {e}")
                            
                        await self._broadcast_to_channel(channel, message)
                    except queue.Empty:
                        break
            except Exception as e:
                print(f"IPC Loop Error: {e}")
            
            await asyncio.sleep(0.01) # Low latency poll

    async def _broadcast_to_channel(self, channel, message):
        """Send message to all clients subscribed to channel. IPC aware."""
        # If we have a queue and NO loop (meaning we are a worker process), put in queue
        if self.ipc_queue and (self.loop is None or not self.loop.is_running()):
             try:
                 # print(f"DEBUG: Actor putting to IPC: {channel}")
                 self.ipc_queue.put((channel, message))
             except:
                 pass
             return

        # Otherwise, if we have a loop, we are the server (or standalone)
        if channel not in self.subscriptions:
            return
        
        subs = self.subscriptions[channel]
        if not subs:
            return

        to_remove = set()
        for client in subs:
            try:
                await client.send(message)
            except:
                to_remove.add(client)
        
        if to_remove:
            subs.difference_update(to_remove)
            self.clients.difference_update(to_remove)
            
    # --- Helper for Unifying Broadcast ---
    def _emit_message(self, channel, message):
        """Send message via Loop (Server) or IPC Queue (Worker)."""
        # Case 1: Active Event Loop (Server side)
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast_to_channel(channel, message), self.loop)
            
        # Case 2: No Loop but IPC Queue (Worker side)
        elif self.ipc_queue:
            try:
                # print(f"DEBUG: Actor putting to IPC: {channel}")
                self.ipc_queue.put((channel, message))
            except Exception as e:
                print(f"IPC Error: {e}")
                
    # --- Public API methods ---

    def broadcast_metrics(self, iteration, loss, win_rate, avg_game_time=0.0, avg_iter_time=0.0):
        """Update and broadcast training metrics to DASHBOARD."""
        # Sanitize inputs (handle NaN/Inf)
        if loss != loss or loss is None or loss == float('inf') or loss == float('-inf'):
            loss = 0.0
        if win_rate != win_rate or win_rate is None:
            win_rate = 0.0

        metric = {
            'iteration': iteration,
            'loss': round(float(loss), 4),
            'win_rate': round(float(win_rate) * 100, 1),
            'avg_game_time': round(float(avg_game_time), 2),
            'avg_iter_time': round(float(avg_iter_time), 3)
        }
        
        # Deduplication: Update last entry if same iteration
        if self.metrics_history and self.metrics_history[-1].get('iteration') == iteration:
            self.metrics_history[-1] = metric
        else:
            self.metrics_history.append(metric)
            
        self.save_metrics() # Persist
        
        message = json.dumps({
            'type': 'metrics',
            'history': self.metrics_history # Send full history to ensure client sync
        })
        
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast_to_channel('dashboard', message), self.loop)
        elif self.ipc_queue:
            # Worker process: send via IPC queue to main process visualizer
            try:
                self.ipc_queue.put(('dashboard', message))
            except:
                pass

    def broadcast_stats(self, total_games, buffer_size, fps=0.0):
        """Broadcast system/training stats to DASHBOARD."""
        message = json.dumps({
            'type': 'stats',
            'total_games': int(total_games),
            'buffer_size': int(buffer_size),
            'fps': float(fps)
        })
        self._emit_message('dashboard', message)

    def broadcast_state(self, state, game_id=0):
        """Broadcast_state tagged with game_id to GAME channel."""
        if not HAS_WEBSOCKETS: return
            
        state_dict = self._serialize_state(state)
        self.game_states[game_id] = state_dict
        self.latest_game_objects[game_id] = state # Save raw object for debug tensors

        
        message = json.dumps({
            'type': 'state',
            'actor_id': self.actor_id,
            'game_id': game_id,
            'state': state_dict,
            'identities': self.game_identities.get(game_id, self.identities)
        })
        
        self._emit_message(f'game_{self.actor_id}_{game_id}', message)
        
        # Also send lightweight progress update to Dashboard
        progress_msg = json.dumps({
            'type': 'game_progress',
            'actor_id': self.actor_id,
            'game_id': game_id,
            'state': {
                'player_positions': state_dict.get('player_positions', []),
                'scores': state_dict.get('scores', [])
            }
        })
        self._emit_message('dashboard', progress_msg)

    def broadcast_move(self, player, token, dice, game_id=0, from_pos=-1, to_pos=-1, token_stats=None):
        """Broadcast_move tagged with game_id to GAME channel."""
        if 1 <= dice <= 6:
            self.dice_stats[int(player)][int(dice)] += 1
            
        message = json.dumps({
            'type': 'move',
            'actor_id': self.actor_id,
            'game_id': game_id,
            'player': int(player),
            'token': int(token),
            'dice': int(dice),
            'from_pos': int(from_pos),
            'to_pos': int(to_pos),
            'token_stats': token_stats
        })
        
        self._emit_message(f'game_{self.actor_id}_{game_id}', message)
    
    def _serialize_state(self, state):
        """Convert GameState to JSON-serializable dict."""
        return {
            'player_positions': [
                [int(state.player_positions[p][t]) for t in range(4)]
                for p in range(4)
            ],
            'scores': [int(state.scores[p]) for p in range(4)],
            'current_player': int(state.current_player),
            'current_dice_roll': int(state.current_dice_roll),
            'is_terminal': bool(state.is_terminal)
        }



    def broadcast_elo_history(self, history):
        """Broadcast Elo history to DASHBOARD."""
        self.elo_history = history # Persist!
        
        message = json.dumps({
            'type': 'elo_history',
            'history': {k: list(v) for k, v in history.items()},
            'timestamp': time.time()
        })
        self._emit_message('dashboard', message)

    def broadcast_identities(self, identities, game_id=0):
        """Broadcast identities to GAME channel."""
        self.identities = identities
        self.game_identities[game_id] = identities
        
        message = json.dumps({'type': 'identities', 'data': identities, 'game_id': game_id, 'actor_id': self.actor_id})
        self._emit_message(f'game_{self.actor_id}_{game_id}', message)

    def broadcast_league_stats(self, stats):
        """Broadcast league stats to DASHBOARD."""
        message = json.dumps({'type': 'league_stats', 'stats': stats})
        self._emit_message('dashboard', message)

    def broadcast_batch_init(self, batch_size):
        """Broadcast batch init to DASHBOARD."""
        self.batch_size = int(batch_size)
        self.game_results.clear()
        self.batch_start_time = time.time()
        
        # Initialize game start times for all games in batch
        self.game_start_times = {i: self.batch_start_time for i in range(self.batch_size)}
        
        message = json.dumps({
            'type': 'batch_init', 
            'actor_id': self.actor_id,
            'batch_size': self.batch_size,
            'batch_start_time': self.batch_start_time,
            'game_start_times': self.game_start_times
        })
        self._emit_message('dashboard', message)

    def broadcast_game_result(self, game_id, winner):
        """Broadcast result to DASHBOARD (grid) AND GAME channel (modal/log)."""
        self.game_results[int(game_id)] = int(winner)
        
        # Get winner identity from stored game identities
        winner_identity = 'Unknown'
        identities = self.game_identities.get(int(game_id), [])
        if identities and 0 <= winner < len(identities):
            winner_identity = identities[winner]
        
        # Determine if Main model won (player 0 with 'Main' identity)
        main_won = (winner == 0 and winner_identity == 'Main')
        
        msg = {
            'type': 'game_result', 
            'actor_id': self.actor_id,
            'game_id': int(game_id), 
            'winner': int(winner),
            'winner_identity': winner_identity,
            'main_won': main_won,
            'identities': identities  # Full list for reference
        }
        message = json.dumps(msg)
        
        self._emit_message('dashboard', message)
        self._emit_message(f'game_{self.actor_id}_{game_id}', message)

    def broadcast_ghost_games(self, ghost_game_ids):
        """Broadcast ghost games to DASHBOARD."""
        self.ghost_game_ids = list(ghost_game_ids)
        message = json.dumps({'type': 'ghost_games', 'game_ids': self.ghost_game_ids})
        self._emit_message('dashboard', message)

    def broadcast_heuristic_games(self, game_ids):
        """Broadcast heuristic games to DASHBOARD."""
        message = json.dumps({'type': 'heuristic_games', 'game_ids': list(game_ids)})
        self._emit_message('dashboard', message)

    def broadcast_elo(self, main_elo, ghost_name=None, ghost_elo=None, rankings=None):
        """Broadcast Elo ratings to DASHBOARD."""
        self.elo_data = {
            'main_elo': float(main_elo),
            'ghost_name': ghost_name,
            'ghost_elo': float(ghost_elo) if ghost_elo else None,
            'rankings': rankings or []
        }
        
        message = json.dumps({'type': 'elo_update', **self.elo_data})
        self._emit_message('dashboard', message)

    def broadcast_training_config(self, temp_schedule, ghost_fraction, augmentation):
        """Broadcast training config to DASHBOARD."""
        message = json.dumps({
            'type': 'training_config',
            'temp_schedule': temp_schedule,
            'ghost_fraction': ghost_fraction,
            'augmentation': augmentation
        })
        self._emit_message('dashboard', message)

    def set_worker(self, worker):
        """Set the worker instance to bridge commands."""
        self.worker_instance = worker

# Global instance
visualizer = GameVisualizer()

def enable_visualization():
    visualizer.start_server()
    return visualizer
