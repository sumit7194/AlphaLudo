#!/usr/bin/env python3
"""
V6.1 Fast Training — GPU Inference Server + CPU Actors

Architecture:
  N Actor processes (CPU, game sim only, NO model) 
    → request_queue → GPU Inference Server (MPS/CUDA)
    ← response_queues ←
  Actors send trajectories → Learner (MPS/CUDA, PPO updates)

The GPU handles ALL neural network computation. Actors are pure game logic.
"""
import os, sys, time, signal, argparse, json, threading, functools
import multiprocessing as mp
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

os.environ.setdefault('TD_LUDO_RUN_NAME', 'ac_v6_1_strategic')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = os.environ.get('TD_LUDO_RUN_NAME', 'ac_v6_1_strategic')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', RUN_NAME)
GHOSTS_DIR = os.path.join(CHECKPOINT_DIR, 'ghosts')
MAIN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'model_latest.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'model_best.pt')
STATS_PATH = os.path.join(CHECKPOINT_DIR, 'live_stats.json')
WEIGHT_SYNC_PATH = os.path.join(CHECKPOINT_DIR, 'actor_weights.pt')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GHOSTS_DIR, exist_ok=True)

STOP_EVENT = mp.Event()
def signal_handler(sig, frame):
    if STOP_EVENT.is_set(): sys.exit(1)
    print("\n[Main] Graceful shutdown..."); STOP_EVENT.set()
signal.signal(signal.SIGINT, signal_handler)

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, directory=None, **kw): super().__init__(*a, directory=directory, **kw)
    def do_GET(self):
        if self.path == '/api/stats': self._j(STATS_PATH)
        elif self.path == '/api/metrics': self._j(os.path.join(CHECKPOINT_DIR, 'training_metrics.json'))
        elif self.path == '/api/elo': self._j(os.path.join(CHECKPOINT_DIR, 'elo_ratings.json'))
        else: super().do_GET()
    def _j(self, p):
        try:
            with open(p) as f: d = f.read()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Access-Control-Allow-Origin','*'); self.end_headers(); self.wfile.write(d.encode())
        except: self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

def start_dashboard(port):
    d = PROJECT_ROOT
    if not os.path.exists(os.path.join(d,'index.html')): return
    h = functools.partial(DashboardHandler, directory=d)
    try:
        s = HTTPServer(('0.0.0.0', port), h)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        print(f"[Dashboard] http://localhost:{port}")
    except: pass

def get_device():
    import torch
    if torch.backends.mps.is_available(): return 'mps'
    elif torch.cuda.is_available(): return 'cuda'
    return 'cpu'

# =========================================================================
# Actor Worker — NO model, sends requests to inference server
# =========================================================================
def actor_worker(actor_id, batch_size, request_queue, response_queue,
                 trajectory_queue, total_games_counter, stop_event, config):
    import random
    import td_ludo_cpp as ludo_cpp
    from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, ExpertBot, RandomBot
    from src.reward_shaping import compute_shaped_reward

    bots = {'Heuristic': HeuristicLudoBot(), 'Aggressive': AggressiveBot(),
            'Defensive': DefensiveBot(), 'Expert': ExpertBot(), 'Random': RandomBot()}
    game_comp = config.get('game_composition', {'SelfPlay':0.40,'Expert':0.25,'Heuristic':0.15,'Aggressive':0.10,'Defensive':0.10})

    env = ludo_cpp.VectorGameState(batch_size, True)
    consec = np.zeros((batch_size, 4), dtype=int)
    move_counts = np.zeros(batch_size, dtype=int)
    trajectories = [{} for _ in range(batch_size)]

    def rand_comp():
        r = random.random(); cum = 0; gt = 'SelfPlay'
        for g, p in game_comp.items():
            cum += p
            if r < cum: gt = g; break
        mp = random.choice([0, 2]); opp = 2 if mp == 0 else 0
        pt = ['Inactive']*4; ct = ['Inactive']*4; tp = {mp}
        pt[mp] = 'Model'; ct[mp] = 'Model'
        if gt == 'SelfPlay': pt[opp] = 'SelfPlay'; ct[opp] = 'SelfPlay'; tp.add(opp)
        else: pt[opp] = gt; ct[opp] = gt
        return {'model_player': mp, 'player_types': pt, 'controllers': ct, 'train_players': sorted(tp)}

    comps = [rand_comp() for _ in range(batch_size)]
    games_played = 0

    def get_temp(tg):
        ts=config.get('temp_start',1.1); te=config.get('temp_end',0.95); td=config.get('temp_decay_games',20000)
        if tg >= td: return te
        return ts - (tg/td)*(ts-te)

    print(f"[Actor {actor_id}] Started (batch={batch_size}, no model)", flush=True)

    while not stop_event.is_set():
        tg = total_games_counter.value
        temp = get_temp(tg)
        model_requests = []  # (game_idx, cp, lmoves, state_enc, mask)
        actions = [-1] * batch_size
        current_players = [-1] * batch_size

        for i in range(batch_size):
            game = env.get_game(i)
            if game.is_terminal: continue
            cp = game.current_player; current_players[i] = cp
            if move_counts[i] >= 10000: game.is_terminal = True; continue

            if game.current_dice_roll == 0:
                roll = random.randint(1,6); game.current_dice_roll = roll
                if roll == 6: consec[i,cp] += 1
                else: consec[i,cp] = 0
                if consec[i,cp] >= 3:
                    nxt=(cp+1)%4
                    while not game.active_players[nxt]: nxt=(nxt+1)%4
                    game.current_player=nxt; game.current_dice_roll=0; consec[i,cp]=0; continue

            lm = ludo_cpp.get_legal_moves(game)
            if not lm:
                nxt=(cp+1)%4
                while not game.active_players[nxt]: nxt=(nxt+1)%4
                game.current_player=nxt; game.current_dice_roll=0; continue

            ctrl = comps[i]['controllers'][cp]
            if ctrl in ('Model', 'SelfPlay'):
                enc = ludo_cpp.encode_state_v6(game)
                mask = np.zeros(4, dtype=np.float32)
                for m in lm: mask[m] = 1.0
                model_requests.append((i, cp, lm, np.array(enc, dtype=np.float32), mask))
                actions[i] = -2  # placeholder
            else:
                bot = bots.get(comps[i]['player_types'][cp], bots['Random'])
                bot.player_id = cp
                actions[i] = bot.select_move(game, lm)

        # Send batch to GPU inference server
        if model_requests:
            states_batch = np.stack([r[3] for r in model_requests])
            masks_batch = np.stack([r[4] for r in model_requests])
            request_queue.put((actor_id, states_batch, masks_batch, temp))

            # Wait for response
            try:
                sampled_actions, old_lps, probs = response_queue.get(timeout=5.0)
            except:
                # Fallback to random
                for j, (idx, cp, lm, _, _) in enumerate(model_requests):
                    actions[idx] = random.choice(lm)
                model_requests = []

            for j, (idx, cp, lm, enc, mask) in enumerate(model_requests):
                action = int(sampled_actions[j])
                if action not in lm: action = random.choice(lm)
                actions[idx] = action

                if cp in comps[idx]['train_players']:
                    if cp not in trajectories[idx]:
                        trajectories[idx][cp] = {'states':[],'actions':[],'legal_masks':[],'old_log_probs':[],'temperatures':[],'step_rewards':[]}
                    t = trajectories[idx][cp]
                    t['states'].append(enc)
                    t['actions'].append(action)
                    t['legal_masks'].append(mask)
                    t['old_log_probs'].append(float(old_lps[j]))
                    t['temperatures'].append(float(temp))
                    t['step_rewards'].append(0.0)

        # Pre-step states
        pre = []
        for i in range(batch_size):
            g = env.get_game(i)
            pre.append({p: list(g.player_positions[p]) for p in range(4)})

        fa = [a if a >= 0 else -1 for a in actions]
        for i, a in enumerate(fa):
            if a >= 0: move_counts[i] += 1
        _, _, dones, infos = env.step(fa)

        # Rewards and completions
        completed = []
        for i in range(batch_size):
            cp = current_players[i]
            if cp >= 0 and cp in trajectories[i]:
                t = trajectories[i][cp]
                if t['step_rewards']:
                    ng = env.get_game(i)
                    class _S:
                        def __init__(s, pos, sc=None, ac=None):
                            s.player_positions=pos; s.scores=sc or [0]*4; s.active_players=ac or [True,False,True,False]
                    r = compute_shaped_reward(_S(pre[i], list(env.get_game(i).scores), list(env.get_game(i).active_players)),
                                              _S(ng.player_positions, list(ng.scores), list(ng.active_players)), cp)
                    t['step_rewards'][-1] = r

            if dones[i]:
                w = infos[i]['winner']; c = comps[i]; mp = c['model_player']
                mw = (w == mp) if w >= 0 else False
                if w >= 0:
                    for tp in c['train_players']:
                        if tp in trajectories[i]:
                            tr = trajectories[i][tp]
                            if tr['states']:
                                T = len(tr['states']); z = 1.0 if tp == w else -1.0; gamma = 0.999
                                rets = np.zeros(T, dtype=np.float32); R = 0.0
                                for t in reversed(range(T)):
                                    rt = tr['step_rewards'][t]
                                    if t == T-1: rt += z
                                    R = rt + gamma*R; rets[t] = R
                                completed.append({
                                    'player_states': np.stack(tr['states']),
                                    'step_actions': np.array(tr['actions'], dtype=np.int64),
                                    'legal_masks': np.stack(tr['legal_masks']),
                                    'old_log_probs': np.array(tr['old_log_probs'], dtype=np.float32),
                                    'temperatures': np.array(tr['temperatures'], dtype=np.float32),
                                    'returns': rets,
                                    'game_info': {'winner':w,'model_won':mw,'model_player':mp,'identities':c['player_types'],'total_moves':T},
                                })
                games_played += 1
                env.reset_game(i); comps[i] = rand_comp(); consec[i]=0; trajectories[i]={}; move_counts[i]=0

        for gd in completed:
            try: trajectory_queue.put(gd, timeout=5.0)
            except: pass

    print(f"[Actor {actor_id}] Done ({games_played} games)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--actors', type=int, default=6)
    parser.add_argument('--actor-batch', type=int, default=64)
    parser.add_argument('--ppo-buffer', type=int, default=4096)
    parser.add_argument('--ppo-minibatch', type=int, default=256)
    parser.add_argument('--hours', type=float, default=0)
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--port', type=int, default=8789)
    parser.add_argument('--sl-weights', type=str, default=None)
    args = parser.parse_args()

    if args.fresh and os.path.exists(CHECKPOINT_DIR):
        import shutil
        for f in os.listdir(CHECKPOINT_DIR):
            if 'model_sl' in f: continue
            fp = os.path.join(CHECKPOINT_DIR, f)
            try:
                if os.path.isfile(fp): os.unlink(fp)
                elif os.path.isdir(fp): shutil.rmtree(fp)
            except: pass
        os.makedirs(GHOSTS_DIR, exist_ok=True)

    device_str = args.device or get_device()
    na = args.actors; ab = args.actor_batch; tp = na * ab

    sl_path = args.sl_weights
    if sl_path is None and not args.resume:
        for c in [os.path.join(CHECKPOINT_DIR,'model_sl_v6_1_best.pt'), os.path.join(CHECKPOINT_DIR,'model_sl.pt')]:
            if os.path.exists(c): sl_path = c; break

    resume_path = MAIN_CKPT_PATH if args.resume else None

    wv = mp.Value('i', 0)
    tgc = mp.Value('i', 0)
    tq = mp.Queue(maxsize=500)
    sq = mp.Queue(maxsize=100)

    config = {
        'learning_rate':1e-5,'weight_decay':1e-4,'max_grad_norm':1.0,
        'entropy_coeff':0.005,'value_loss_coeff':0.5,'clip_epsilon':0.2,
        'ppo_epochs':3,'ppo_buffer_steps':args.ppo_buffer,'ppo_minibatch_size':args.ppo_minibatch,
        'game_composition':{'SelfPlay':0.40,'Expert':0.25,'Heuristic':0.15,'Aggressive':0.10,'Defensive':0.10},
        'ghosts_dir':GHOSTS_DIR,'max_moves':10000,
        'temp_start':1.1,'temp_end':0.95,'temp_decay_games':20000,
        'eval_interval':2000,'eval_games':500,'save_interval':300,
        'ghost_save_interval':2000,'max_ghosts':20,'early_stop_patience':100,
    }

    if not args.no_dashboard: start_dashboard(args.port)

    print(f"\n{'='*60}")
    print(f"  V6.1 FAST Training — GPU Inference Server")
    print(f"  {na} CPU actors + 1 GPU inference server + 1 GPU learner")
    print(f"  Model: AlphaLudoV5(128ch, 10res, 24in)")
    print(f"  Parallel games: {tp}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    if sl_path: print(f"  SL weights: {sl_path}")
    if resume_path and os.path.exists(resume_path): print(f"  Resuming: {resume_path}")
    print(f"{'='*60}\n", flush=True)

    # Import learner (same as before, reuse from the inline version)
    # For now, import the learner_worker from the previous train_v6_1_fast
    # We need to use the same learner but with the inference server for actors

    # Start inference server
    from src.inference_server_v6 import inference_server_v6_worker
    request_queue = mp.Queue(maxsize=na * 2)
    response_queues = [mp.Queue(maxsize=4) for _ in range(na)]

    # Learner needs to export weights first
    # Inline learner startup to export initial weights
    import torch
    from src.model import AlphaLudoV5
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd)
        print(f"[Main] Loaded checkpoint for initial weights")
    elif sl_path and os.path.exists(sl_path):
        ckpt = torch.load(sl_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd)
        print(f"[Main] Loaded SL weights for initial weights")

    # Export initial weights
    clean = {k.replace('_orig_mod.',''): v.cpu() for k,v in model.state_dict().items()}
    torch.save(clean, WEIGHT_SYNC_PATH + '.tmp')
    os.replace(WEIGHT_SYNC_PATH + '.tmp', WEIGHT_SYNC_PATH)
    wv.value = 1
    del model, clean

    # Start inference server process
    infer_proc = mp.Process(
        target=inference_server_v6_worker,
        args=(device_str, request_queue, response_queues, WEIGHT_SYNC_PATH, wv, STOP_EVENT),
        name='infer-server', daemon=False)
    infer_proc.start()
    print(f"[Main] Inference server started (PID {infer_proc.pid})")
    time.sleep(2)

    # Start learner (reuse the inline learner from previous script)
    # Import it as a function
    from train_v6_1_fast_learner import learner_worker
    learner_proc = mp.Process(
        target=learner_worker,
        args=(tq, sq, WEIGHT_SYNC_PATH, wv, tgc, STOP_EVENT,
              device_str, config, CHECKPOINT_DIR, GHOSTS_DIR, resume_path, sl_path),
        name='learner', daemon=False)
    learner_proc.start()
    print(f"[Main] Learner started (PID {learner_proc.pid})")

    # Start actors (no model!)
    actor_procs = []
    for i in range(na):
        p = mp.Process(
            target=actor_worker,
            args=(i, ab, request_queue, response_queues[i], tq, tgc, STOP_EVENT, config),
            name=f'actor-{i}', daemon=False)
        p.start()
        actor_procs.append(p)
        print(f"[Main] Actor {i} started (PID {p.pid})")

    t0 = time.time()
    print("\n[Main] Training running. Press Ctrl+C to stop.\n", flush=True)

    try:
        while not STOP_EVENT.is_set():
            time.sleep(2.0)
            if not learner_proc.is_alive(): print("[Main] Learner died!"); STOP_EVENT.set(); break
            if not infer_proc.is_alive(): print("[Main] InferServer died!"); STOP_EVENT.set(); break
            if args.hours > 0 and (time.time()-t0)/3600 >= args.hours: STOP_EVENT.set(); break
    except KeyboardInterrupt:
        STOP_EVENT.set()

    print("[Main] Shutting down...")
    for i, p in enumerate(actor_procs):
        p.join(timeout=10)
        if p.is_alive(): p.terminate()
    infer_proc.join(timeout=10)
    if infer_proc.is_alive(): infer_proc.terminate()
    learner_proc.join(timeout=30)
    if learner_proc.is_alive(): learner_proc.terminate()

    print(f"\n{'='*60}\n  V6.1 Training Complete ({(time.time()-t0)/3600:.2f}h)\n{'='*60}")

if __name__ == '__main__':
    main()
