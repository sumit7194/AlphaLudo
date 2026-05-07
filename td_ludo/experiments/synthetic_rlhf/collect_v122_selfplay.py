"""Collect V12.2 self-play decisions into SQLite for synthetic-RLHF work.

V12.2 (pre-search, post-encoder-fix) plays both seats. Every decision is
recorded with full board state, legal mask, V12.2's policy distribution,
win-prob estimate, and the action sampled. After the game ends, the winner
is back-filled onto every recorded decision.

Designed for fast filtering ("all states where current player has 4 tokens
at base AND dice=6") via indexed SQL queries. The own_pos and opp_pos
fields are stored as comma-separated ints for human-readability; filtering
is done via SQL CASE expressions or parsed in Python.

Usage:
  python -m experiments.synthetic_rlhf.collect_v122_selfplay \\
      --n-games 10000 --db experiments/synthetic_rlhf/v122_selfplay.db
"""
import argparse
import os
import random
import signal
import sqlite3
import sys
import time
import numpy as np
import torch

# Signal-driven graceful shutdown: collector flushes its buffer before exit
# so SIGTERM/SIGINT never costs more than `commit_every_games` worth of work.
_STOP_REQUESTED = False
def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f'\n[signal] Received {signal.Signals(signum).name} — finishing '
          f'current step then flushing + exiting.', flush=True)

signal.signal(signal.SIGTERM, _request_stop)
signal.signal(signal.SIGINT,  _request_stop)

# Repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import td_ludo_cpp as cpp
from td_ludo.models.v12 import AlphaLudoV12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', default='play/model_weights/v12_2/model_latest.pt',
                   help='V12.2 (pre-search) checkpoint')
    p.add_argument('--db', default='experiments/synthetic_rlhf/v122_selfplay.db',
                   help='SQLite database path (will be created)')
    p.add_argument('--n-games', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=256,
                   help='Parallel games. Lower if MPS memory tight.')
    p.add_argument('--max-game-len', type=int, default=400,
                   help='Force terminate games beyond this many moves.')
    p.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--commit-every-games', type=int, default=200,
                   help='Flush DB transaction every N completed games')
    return p.parse_args()


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    move_idx INTEGER NOT NULL,
    current_player INTEGER NOT NULL,
    dice INTEGER NOT NULL,
    own_pos TEXT NOT NULL,        -- 'p0,p1,p2,p3' relative-to-current_player
    opp_pos TEXT NOT NULL,        -- 'p0,p1,p2,p3' opponent's positions
    n_own_at_base INTEGER NOT NULL,
    n_own_on_track INTEGER NOT NULL,
    n_own_at_home INTEGER NOT NULL,
    n_opp_at_base INTEGER NOT NULL,
    n_opp_on_track INTEGER NOT NULL,
    n_opp_at_home INTEGER NOT NULL,
    legal_mask TEXT NOT NULL,     -- '1,0,1,1' style — 4 chars
    policy_t0 REAL NOT NULL,
    policy_t1 REAL NOT NULL,
    policy_t2 REAL NOT NULL,
    policy_t3 REAL NOT NULL,
    win_prob REAL NOT NULL,
    moves_remaining REAL,
    action_chosen INTEGER NOT NULL,
    winner INTEGER                -- back-filled at game end (0 or 2 in 2P mode, -1 timeout)
);
CREATE INDEX IF NOT EXISTS idx_dice ON decisions(dice);
CREATE INDEX IF NOT EXISTS idx_cp ON decisions(current_player);
CREATE INDEX IF NOT EXISTS idx_n_own_base ON decisions(n_own_at_base);
CREATE INDEX IF NOT EXISTS idx_n_own_home ON decisions(n_own_at_home);
CREATE INDEX IF NOT EXISTS idx_game ON decisions(game_id);
CREATE INDEX IF NOT EXISTS idx_winner ON decisions(winner);
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    n_moves INTEGER,
    winner INTEGER,
    duration_sec REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

BASE_POS = -1
HOME_POS = 99


def count_phases(positions):
    """(n_at_base, n_on_track, n_at_home)"""
    nb = sum(1 for p in positions if p == BASE_POS)
    nh = sum(1 for p in positions if p == HOME_POS)
    nt = 4 - nb - nh
    return nb, nt, nh


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)

    print(f'Device: {device}')
    print(f'Teacher: {args.teacher}')
    print(f'DB: {args.db}')
    print(f'Games target: {args.n_games:,}, batch: {args.batch_size}')

    # Load model
    print('Loading V12.2...')
    model = AlphaLudoV12(num_res_blocks=3, num_channels=128, num_attn_layers=2,
                         num_heads=4, ffn_ratio=4, dropout=0.0, in_channels=33)
    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded ({n_params:,} params)')

    # DB init
    os.makedirs(os.path.dirname(os.path.abspath(args.db)) or '.', exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('teacher', ?)", (args.teacher,))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('encoder', 'encode_state_v11_post_fix')")
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('started', ?)",
                 (time.strftime('%Y-%m-%d %H:%M:%S'),))
    conn.commit()

    # Vectorised env
    BATCH = args.batch_size
    env = cpp.VectorGameState(batch_size=BATCH, two_player_mode=True)
    consec_sixes = np.zeros((BATCH, 4), dtype=np.int32)
    move_counts = np.zeros(BATCH, dtype=np.int32)
    game_ids = list(range(BATCH))
    next_game_id = BATCH
    game_start_times = [time.time()] * BATCH
    pending = [[] for _ in range(BATCH)]   # list of (decision_id_in_buffer) tuples
    buffered_rows = []                      # all rows pending DB insert

    games_completed = 0
    total_decisions = 0
    last_commit_at = 0
    t_start = time.time()
    last_log = t_start

    while games_completed < args.n_games and not _STOP_REQUESTED:
        # Roll dice if needed (and triple-six pass)
        for i in range(BATCH):
            g = env.get_game(i)
            if g.is_terminal: continue
            if move_counts[i] >= args.max_game_len: continue
            cp = g.current_player
            if g.current_dice_roll == 0:
                roll = random.randint(1, 6)
                g.current_dice_roll = roll
                if roll == 6: consec_sixes[i, cp] += 1
                else: consec_sixes[i, cp] = 0
                if consec_sixes[i, cp] >= 3:
                    nxt = (cp + 1) % 4
                    while not g.active_players[nxt]: nxt = (nxt + 1) % 4
                    g.current_player = nxt
                    g.current_dice_roll = 0
                    consec_sixes[i, cp] = 0

        # Gather decision states
        decision_idxs = []
        batch_states = []
        batch_legals = []
        batch_masks = []
        for i in range(BATCH):
            g = env.get_game(i)
            if g.is_terminal or move_counts[i] >= args.max_game_len:
                continue
            if g.current_dice_roll == 0:  # was passed by triple six
                continue
            legal = cpp.get_legal_moves(g)
            if not legal:
                # advance turn
                cp = g.current_player
                nxt = (cp + 1) % 4
                while not g.active_players[nxt]: nxt = (nxt + 1) % 4
                g.current_player = nxt
                g.current_dice_roll = 0
                continue
            mask = np.zeros(4, dtype=np.float32)
            for m in legal: mask[m] = 1.0
            decision_idxs.append(i)
            batch_states.append(np.asarray(cpp.encode_state_v11(g), dtype=np.float32))
            batch_legals.append(legal)
            batch_masks.append(mask)

        if not decision_idxs:
            # All games either terminal or stuck — reset terminals below
            pass
        else:
            states_t = torch.from_numpy(np.stack(batch_states)).to(device)
            masks_t  = torch.from_numpy(np.stack(batch_masks)).to(device)
            with torch.no_grad():
                policy, win_prob, moves_remaining = model(states_t, masks_t)
                policy_np = policy.cpu().numpy()
                win_prob_np = win_prob.view(-1).cpu().numpy()
                moves_np = moves_remaining.view(-1).cpu().numpy()
                actions = torch.multinomial(policy, num_samples=1).squeeze(1).cpu().numpy()

            for k, idx in enumerate(decision_idxs):
                g = env.get_game(idx)
                cp = int(g.current_player)
                opp_seat = (cp + 2) % 4
                own_pos = list(g.player_positions[cp])
                opp_pos = list(g.player_positions[opp_seat])
                n_own_b, n_own_t, n_own_h = count_phases(own_pos)
                n_opp_b, n_opp_t, n_opp_h = count_phases(opp_pos)
                action = int(actions[k])
                if batch_masks[k][action] == 0:
                    action = int(batch_legals[k][0])
                row = (
                    game_ids[idx], int(move_counts[idx]),
                    cp, int(g.current_dice_roll),
                    ','.join(str(p) for p in own_pos),
                    ','.join(str(p) for p in opp_pos),
                    n_own_b, n_own_t, n_own_h,
                    n_opp_b, n_opp_t, n_opp_h,
                    ''.join('1' if x > 0 else '0' for x in batch_masks[k]),
                    float(policy_np[k][0]), float(policy_np[k][1]),
                    float(policy_np[k][2]), float(policy_np[k][3]),
                    float(win_prob_np[k]),
                    float(moves_np[k]),
                    action,
                    None,  # winner — back-filled later
                )
                buffered_rows.append(row)
                pending[idx].append(len(buffered_rows) - 1)
                # Action will be applied via env.step() below.

        # Apply all moves via env.step (only one step per outer iteration).
        # Build action list of size BATCH: -1 for terminals/no-moves, actual otherwise.
        step_actions = [-1] * BATCH
        for k, idx in enumerate(decision_idxs):
            # action was sampled above
            if not buffered_rows: break
            step_actions[idx] = buffered_rows[pending[idx][-1]][-2]  # action_chosen field
        _, _, dones_np, infos = env.step(step_actions)

        for i in range(BATCH):
            if step_actions[i] >= 0:
                move_counts[i] += 1

        # Handle terminations
        for i in range(BATCH):
            g_now = env.get_game(i)
            if dones_np[i] or move_counts[i] >= args.max_game_len:
                winner = int(infos[i].get('winner', -1)) if dones_np[i] else -1
                # Back-fill winner on every pending decision for this game
                for ridx in pending[i]:
                    row_list = list(buffered_rows[ridx])
                    row_list[-1] = winner
                    buffered_rows[ridx] = tuple(row_list)
                # game record
                conn.execute(
                    'INSERT INTO games(game_id, n_moves, winner, duration_sec) VALUES(?,?,?,?)',
                    (game_ids[i], int(move_counts[i]), winner,
                     time.time() - game_start_times[i])
                )
                games_completed += 1
                # Reset slot for new game
                env.reset_game(i)
                consec_sixes[i] = 0
                move_counts[i] = 0
                pending[i] = []
                game_ids[i] = next_game_id
                game_start_times[i] = time.time()
                next_game_id += 1

        # Periodic commit
        if games_completed - last_commit_at >= args.commit_every_games:
            conn.executemany(
                'INSERT INTO decisions(game_id, move_idx, current_player, dice, '
                'own_pos, opp_pos, n_own_at_base, n_own_on_track, n_own_at_home, '
                'n_opp_at_base, n_opp_on_track, n_opp_at_home, legal_mask, '
                'policy_t0, policy_t1, policy_t2, policy_t3, win_prob, '
                'moves_remaining, action_chosen, winner) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                buffered_rows,
            )
            total_decisions += len(buffered_rows)
            buffered_rows = []
            # Pending lists now have stale indices — clear since data is committed.
            for i in range(BATCH):
                pending[i] = []
            conn.commit()
            last_commit_at = games_completed

        now = time.time()
        if now - last_log > 5:
            elapsed = now - t_start
            gpm = games_completed / max(1e-6, elapsed) * 60
            print(f'  [{elapsed:>6.0f}s] games {games_completed:>6}/{args.n_games:,} '
                  f'| decisions buffered+stored {total_decisions + len(buffered_rows):>9,} '
                  f'| gpm {gpm:>6.1f}', flush=True)
            last_log = now

    # Final commit
    if buffered_rows:
        conn.executemany(
            'INSERT INTO decisions(game_id, move_idx, current_player, dice, '
            'own_pos, opp_pos, n_own_at_base, n_own_on_track, n_own_at_home, '
            'n_opp_at_base, n_opp_on_track, n_opp_at_home, legal_mask, '
            'policy_t0, policy_t1, policy_t2, policy_t3, win_prob, '
            'moves_remaining, action_chosen, winner) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            buffered_rows,
        )
        total_decisions += len(buffered_rows)
        conn.commit()

    elapsed = time.time() - t_start
    print()
    print(f'Done. {games_completed} games in {elapsed/60:.1f} min '
          f'({games_completed/elapsed*60:.0f} gpm).')
    print(f'Total decisions: {total_decisions:,}')

    cur = conn.execute('SELECT COUNT(*) FROM decisions')
    print(f'DB row count: {cur.fetchone()[0]:,}')
    cur = conn.execute('SELECT COUNT(*) FROM games')
    print(f'DB game count: {cur.fetchone()[0]:,}')
    cur = conn.execute('SELECT winner, COUNT(*) FROM games GROUP BY winner')
    for w, c in cur.fetchall():
        print(f'  winner={w}: {c}')
    conn.close()


if __name__ == '__main__':
    main()
