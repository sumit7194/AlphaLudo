"""Paired-dice A/B: rich-value candidate (Exp 62) vs v15.2 best. Greedy, both v15."""
import random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "td_ludo_v15"))
import td_ludo_cpp as cpp, torch
from eval_v135_v15_v151_local import load_v15_any, make_picker_v15

dev = torch.device("cpu")
CAND = "/tmp/richvalue_best.pt"
V152 = str(HERE.parent / "checkpoint_backups/v152_gaeterminal_2026-06-19/v152_gaeterminal_BEST_eval84.65pct_2026-06-19.pt")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600

def mk(p):
    return make_picker_v15(load_v15_any(p, dev, d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1), dev, history_len=1)
cand, base = mk(CAND), mk(V152)
print(f"loaded both | {2*N} games", flush=True)

def play(p0, p2, seed):
    random.seed(seed); s = cpp.create_initial_state_2p(); csix = [0, 0, 0, 0]; mc = 0
    while not s.is_terminal and mc < 400:
        cp = int(s.current_player)
        if not s.active_players[cp]:
            n = (cp + 1) % 4
            while not s.active_players[n]: n = (n + 1) % 4
            s.current_player = n; continue
        if s.current_dice_roll == 0:
            d = random.randint(1, 6)
            if d == 6:
                csix[cp] += 1
                if csix[cp] >= 3:
                    csix[cp] = 0; n = (cp + 1) % 4
                    while not s.active_players[n]: n = (n + 1) % 4
                    s.current_player = n; s.current_dice_roll = 0; continue
            else: csix[cp] = 0
            s.current_dice_roll = d
        legal = cpp.get_legal_moves(s)
        if not legal:
            n = (cp + 1) % 4
            while not s.active_players[n]: n = (n + 1) % 4
            s.current_player = n; s.current_dice_roll = 0; continue
        a = p0(s, list(legal), None) if cp == 0 else p2(s, list(legal), None)
        s = cpp.apply_move(s, int(a)); mc += 1
    return cpp.get_winner(s) if s.is_terminal else -1

cw = tot = 0; t0 = time.time()
for i in range(N):
    if play(cand, base, i) == 0: cw += 1
    tot += 1
    if play(base, cand, i) == 2: cw += 1   # candidate as P2, paired seed
    tot += 1
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N} pairs | cand {100*cw/tot:.1f}% | {2*(i+1)/(time.time()-t0):.1f} g/s", flush=True)
wr = 100 * cw / tot; se = (wr * (100 - wr) / tot) ** .5
print(f"\nRICH-VALUE candidate vs v15.2 (paired dice): {wr:.1f}% ({cw}/{tot}) +/-{se:.1f}pp")
print("VERDICT:", "CANDIDATE STRONGER (real squeeze)" if wr > 50 + 2 * se else "v15.2 STRONGER" if wr < 50 - 2 * se else "TIE - no improvement")
