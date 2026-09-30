"""DECISIVE ceiling experiment: searched-net vs raw-net at identical weights.

If 2-ply expectimax (v15 value net at leaves, v15 policy as opponent model)
beats the raw policy (argmax) significantly, inference-time search extracts
skill the raw policy left on the table → a lever to break the ~85% ceiling.
If it ties/loses, the bottleneck is value-net accuracy, not absence of search.

Paired dice (common random numbers): both orientations of game i use the SAME
seed, so the dice cancel as much as the diverging trajectories allow.
Includes a raw-vs-raw control (must be ~50%) to validate the harness.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 ./td_env/bin/python -u h2h_search_vs_raw.py --gpo 500
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "td_ludo_v15"))
import td_ludo_cpp as cpp
from inference_search import V15Eval, NeuralExpectimaxBot

MAX_MOVES = 400
V152_BEST = HERE.parent / "checkpoint_backups/v152_gaeterminal_2026-06-19/v152_gaeterminal_BEST_eval84.65pct_2026-06-19.pt"


def play(p0, p2, seed):
    random.seed(seed)
    s = cpp.create_initial_state_2p(); csix = [0, 0, 0, 0]; mc = 0
    while not s.is_terminal and mc < MAX_MOVES:
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
        a = p0(s, list(legal)) if cp == 0 else p2(s, list(legal))
        s = cpp.apply_move(s, int(a)); mc += 1
    return cpp.get_winner(s) if s.is_terminal else -1


def h2h(pa, pb, gpo, label):
    """pa vs pb, gpo games each orientation, paired seeds. Returns a's win%."""
    aw = bw = dr = 0; t0 = time.time()
    for i in range(gpo):
        w = play(pa, pb, seed=i)            # a=P0, b=P2
        if w == 0: aw += 1
        elif w == 2: bw += 1
        else: dr += 1
        w = play(pb, pa, seed=i)            # b=P0, a=P2  (SAME seed = paired dice)
        if w == 0: bw += 1
        elif w == 2: aw += 1
        else: dr += 1
        if (i + 1) % 25 == 0:
            tot = aw + bw + dr
            print(f"  [{label}] {i+1}/{gpo} pairs | a={aw} b={bw} d={dr} | "
                  f"a={100*aw/tot:.1f}% | {2*(i+1)/(time.time()-t0):.2f} g/s", flush=True)
    tot = aw + bw + dr
    return {"label": label, "a_wins": aw, "b_wins": bw, "draws": dr, "games": tot,
            "a_wr": 100.0 * aw / tot, "elapsed_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpo", type=int, default=500, help="game pairs per matchup (x2 = games)")
    ap.add_argument("--control-gpo", type=int, default=100)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=str(V152_BEST))
    ap.add_argument("--mode", default="anchored", choices=["anchored", "value"])
    ap.add_argument("--margin", type=float, default=0.03)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    dev = torch.device(args.device)
    ev = V15Eval(args.ckpt, device=dev)
    search = NeuralExpectimaxBot(ev, player_id=None, depth=2, mode=args.mode, margin=args.margin)
    print(f"mode={args.mode} margin={args.margin} device={args.device}", flush=True)
    raw = lambda s, legal: ev.policy_argmax(s, list(legal), int(s.current_player))
    search_p = lambda s, legal: search.select_move(s, list(legal))

    print(f"v15.2 best | torch threads {args.threads}\n", flush=True)

    print(f"=== CONTROL: raw vs raw ({2*args.control_gpo} games, expect ~50%) ===", flush=True)
    ctrl = h2h(raw, raw, args.control_gpo, "raw-vs-raw")

    print(f"\n=== MAIN: SEARCH (2-ply) vs RAW ({2*args.gpo} games) ===", flush=True)
    main_r = h2h(search_p, raw, args.gpo, "search-vs-raw")

    se = (main_r["a_wr"] * (100 - main_r["a_wr"]) / main_r["games"]) ** 0.5
    print("\n" + "=" * 60)
    print(f"  CONTROL raw-vs-raw : {ctrl['a_wr']:.1f}%  (sanity: ~50%)")
    print(f"  SEARCH vs RAW      : {main_r['a_wr']:.1f}%  (search wins)  ±{se:.1f}pp")
    edge = main_r["a_wr"] - 50.0
    verdict = ("SEARCH HELPS — extracts skill" if edge > 2 * se else
               "SEARCH HURTS — value-net ranker worse than policy" if edge < -2 * se else
               "NO SIGNIFICANT EFFECT — at the value-net ceiling")
    print(f"  EDGE: {edge:+.1f}pp → {verdict}")
    print("=" * 60)
    out = HERE / "h2h_search_vs_raw_results.json"
    out.write_text(json.dumps({"control": ctrl, "search_vs_raw": main_r,
                               "edge_pp": edge, "se_pp": se}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
