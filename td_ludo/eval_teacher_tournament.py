"""
3-way tournament: V6.1 vs V6.3 vs Expert bot.

Head-to-head round robin. Each pair plays N games, model_player is swapped
randomly between P0 and P2 to control for seat advantage. Greedy policy
(argmax) for both CNNs; Expert uses its heuristic.

Goal: pick the single best teacher for V10's next SL run.
"""
import os, sys, random, time, argparse
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from td_ludo.models.v5 import AlphaLudoV5
from td_ludo.models.v6_3 import AlphaLudoV63
from td_ludo.game.heuristic_bot import ExpertBot
from src.config import MAX_MOVES_PER_GAME


V61_PATH = "checkpoints/ac_v6_1_strategic/model_best.pt"
V63_PATH = "checkpoints/ac_v6_3_capture/model_best.pt"


class V61Agent:
    name = "V6.1"
    def __init__(self, device):
        self.device = device
        self.model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        ckpt = torch.load(V61_PATH, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device).eval()

    def select_move(self, state, legal_moves, consec_sixes):
        if len(legal_moves) == 1:
            return legal_moves[0]
        enc = np.array(ludo_cpp.encode_state_v6(state), dtype=np.float32)
        mask = np.zeros(4, dtype=np.float32)
        for m in legal_moves:
            mask[m] = 1.0
        with torch.no_grad():
            s = torch.from_numpy(enc).unsqueeze(0).to(self.device)
            msk = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            pol, _ = self.model(s, msk)
            action = int(pol.argmax(dim=1).item())
        return action if action in legal_moves else random.choice(legal_moves)


class V63Agent:
    name = "V6.3"
    def __init__(self, device):
        self.device = device
        self.model = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
        ckpt = torch.load(V63_PATH, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device).eval()

    def select_move(self, state, legal_moves, consec_sixes):
        if len(legal_moves) == 1:
            return legal_moves[0]
        enc = np.array(ludo_cpp.encode_state_v6_3(state, consec_sixes), dtype=np.float32)
        mask = np.zeros(4, dtype=np.float32)
        for m in legal_moves:
            mask[m] = 1.0
        with torch.no_grad():
            s = torch.from_numpy(enc).unsqueeze(0).to(self.device)
            msk = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            out = self.model(s, msk)
            pol = out[0]
            action = int(pol.argmax(dim=1).item())
        return action if action in legal_moves else random.choice(legal_moves)


class ExpertAgent:
    name = "Expert"
    def __init__(self, device=None):
        self.bot = ExpertBot(player_id=0)

    def select_move(self, state, legal_moves, consec_sixes):
        return self.bot.select_move(state, legal_moves)


def play_game(agent_p0, agent_p2, max_moves=MAX_MOVES_PER_GAME):
    """Return winner (0 or 2) or -1 on timeout."""
    state = ludo_cpp.create_initial_state_2p()
    consec = [0, 0, 0, 0]
    moves = 0
    while not state.is_terminal and moves < max_moves:
        cp = state.current_player
        if not state.active_players[cp]:
            nxt = (cp + 1) % 4
            while not state.active_players[nxt]:
                nxt = (nxt + 1) % 4
            state.current_player = nxt
            continue

        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
            if state.current_dice_roll == 6:
                consec[cp] += 1
            else:
                consec[cp] = 0
            if consec[cp] >= 3:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                consec[cp] = 0
                continue

        legal = ludo_cpp.get_legal_moves(state)
        if not legal:
            nxt = (cp + 1) % 4
            while not state.active_players[nxt]:
                nxt = (nxt + 1) % 4
            state.current_player = nxt
            state.current_dice_roll = 0
            continue

        agent = agent_p0 if cp == 0 else agent_p2
        action = agent.select_move(state, legal, consec[cp])
        state = ludo_cpp.apply_move(state, action)
        moves += 1

    if state.is_terminal:
        return ludo_cpp.get_winner(state), moves
    return -1, moves


def run_pair(a, b, n_games, verbose=True):
    """Run n_games between a and b, randomly assigning seats. Returns (a_wins, b_wins, draws, game_lengths)."""
    a_wins = 0; b_wins = 0; draws = 0
    lengths = []
    seat_stats = {'a_p0': [0, 0], 'a_p2': [0, 0]}  # [wins, total]
    t0 = time.time()
    for i in range(n_games):
        a_at_p0 = (i % 2 == 0)
        if a_at_p0:
            winner, m = play_game(a, b)
            if winner == 0: a_wins += 1; seat_stats['a_p0'][0] += 1
            elif winner == 2: b_wins += 1
            else: draws += 1
            seat_stats['a_p0'][1] += 1
        else:
            winner, m = play_game(b, a)
            if winner == 2: a_wins += 1; seat_stats['a_p2'][0] += 1
            elif winner == 0: b_wins += 1
            else: draws += 1
            seat_stats['a_p2'][1] += 1
        lengths.append(m)
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            gpm = (i+1) / (elapsed / 60)
            wr = a_wins / (i+1) * 100
            print(f"    [{i+1:>4}/{n_games}] {a.name} {wr:.1f}% vs {b.name} | {gpm:.0f} gpm", flush=True)
    return a_wins, b_wins, draws, lengths, seat_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=500,
                        help='Games per pair (total = 3 * games)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')  # faster for single-game inference at batch 1
    print(f"[Tournament] Device: {device}  Games per pair: {args.games}\n")

    print("[Tournament] Loading agents...")
    v61 = V61Agent(device)
    v63 = V63Agent(device)
    expert = ExpertAgent(device)
    print(f"  V6.1:   {sum(p.numel() for p in v61.model.parameters()):,} params")
    print(f"  V6.3:   {sum(p.numel() for p in v63.model.parameters()):,} params")
    print(f"  Expert: hand-coded heuristic")
    print()

    # 3 pairings
    results = {}
    pairs = [('V6.1 vs V6.3', v61, v63),
             ('V6.1 vs Expert', v61, expert),
             ('V6.3 vs Expert', v63, expert)]
    for label, a, b in pairs:
        print(f"{'='*60}\n  {label} — {args.games} games\n{'='*60}")
        aw, bw, dr, lengths, seat = run_pair(a, b, args.games)
        n = aw + bw + dr
        results[label] = {
            'a': a.name, 'b': b.name,
            'a_wins': aw, 'b_wins': bw, 'draws': dr,
            'a_wr': aw / n * 100, 'b_wr': bw / n * 100,
            'avg_len': float(np.mean(lengths)),
            'seat_p0': seat['a_p0'], 'seat_p2': seat['a_p2'],
        }
        print(f"  {a.name}: {aw}/{n} ({aw/n*100:.1f}%)  |  "
              f"{b.name}: {bw}/{n} ({bw/n*100:.1f}%)  |  "
              f"draws: {dr}  |  avg len: {np.mean(lengths):.1f}")
        if seat['a_p0'][1] > 0 and seat['a_p2'][1] > 0:
            p0_wr = seat['a_p0'][0] / seat['a_p0'][1] * 100
            p2_wr = seat['a_p2'][0] / seat['a_p2'][1] * 100
            print(f"  {a.name} WR by seat: P0 {p0_wr:.1f}% ({seat['a_p0'][0]}/{seat['a_p0'][1]})  |  "
                  f"P2 {p2_wr:.1f}% ({seat['a_p2'][0]}/{seat['a_p2'][1]})")
        print()

    # Final leaderboard
    print(f"\n{'='*60}\n  FINAL LEADERBOARD\n{'='*60}")
    scores = {'V6.1': 0.0, 'V6.3': 0.0, 'Expert': 0.0}
    totals = {'V6.1': 0, 'V6.3': 0, 'Expert': 0}
    for r in results.values():
        scores[r['a']] += r['a_wins']; totals[r['a']] += r['a_wins'] + r['b_wins'] + r['draws']
        scores[r['b']] += r['b_wins']; totals[r['b']] += r['a_wins'] + r['b_wins'] + r['draws']
    for name in ['V6.1', 'V6.3', 'Expert']:
        wr = scores[name] / max(1, totals[name]) * 100
        print(f"  {name:<8}  {int(scores[name])}/{totals[name]}  ({wr:.1f}%)")

    print(f"\n{'='*60}\n  HEAD-TO-HEAD MATRIX (row beats col)\n{'='*60}")
    print(f"  {'':<8} {'V6.1':>8} {'V6.3':>8} {'Expert':>8}")
    for row in ['V6.1', 'V6.3', 'Expert']:
        line = f"  {row:<8}"
        for col in ['V6.1', 'V6.3', 'Expert']:
            if row == col:
                line += f" {'—':>8}"
                continue
            key = f'{row} vs {col}'
            rev = f'{col} vs {row}'
            if key in results:
                line += f" {results[key]['a_wr']:>7.1f}%"
            elif rev in results:
                line += f" {results[rev]['b_wr']:>7.1f}%"
            else:
                line += f" {'?':>8}"
        print(line)


if __name__ == '__main__':
    main()
