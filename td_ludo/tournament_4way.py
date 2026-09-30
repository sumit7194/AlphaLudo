"""4-way round-robin tournament + V15.2 best-vs-latest H2H (Mac, CPU).

Players: v13.5 best, v13.6 best (champion), v15.2 best, MCTSExpectimaxPrior
(strongest scripted bot). All play in the v13 engine via the shared adapters.
Round-robin: every pair plays `--gpo`*2 games (default 2000). PLUS a separate
V15.2 best vs V15.2 latest H2H. Matchups run in parallel (ProcessPool) so the
slow MCTS pairs don't serialize.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 ./td_env/bin/python -u tournament_4way.py --gpo 1000
"""
from __future__ import annotations
import argparse, collections, json, random, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "td_ludo_v15"))

MAX_MOVES = 400

PLAYERS = {
    "v13.5": {"kind": "v135", "path": str(HERE / "play/model_weights/v13_5/model_best.pt")},
    "v13.6": {"kind": "v136", "path": str(ROOT / "checkpoint_backups/v136_gaeterminal_2026-06-17/v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt")},
    "v15.2": {"kind": "v15", "path": str(ROOT / "checkpoint_backups/v152_gaeterminal_2026-06-19/v152_gaeterminal_BEST_eval84.65pct_2026-06-19.pt")},
    "MCTS":  {"kind": "mcts", "name": "MCTSExpectimaxPrior"},
    "v15.2-latest": {"kind": "v15", "path": str(ROOT / "checkpoint_backups/v152_gaeterminal_2026-06-19/v152_gaeterminal_LATEST_G418k_eval84.45pct_2026-06-19.pt")},
}


def build_picker(spec, device):
    """Return picker(state, legal, history). history is unused (all hist_len 0/1)."""
    kind = spec["kind"]
    if kind == "v135":
        from eval_v135_v15_v151_local import load_v135, make_picker_v135
        return make_picker_v135(load_v135(spec["path"], device), device)
    if kind == "v136":
        from eval_v135_v15_v151_local import make_picker_v135
        from h2h_v152gae_vs_v136best import load_v136
        return make_picker_v135(load_v136(spec["path"], device), device)
    if kind == "v15":
        from eval_v135_v15_v151_local import load_v15_any, make_picker_v15
        m = load_v15_any(spec["path"], device, d_model=128, n_heads=4,
                         n_layers=4, ffn_dim=256, history_len=1)
        return make_picker_v15(m, device, history_len=1)
    if kind == "mcts":
        from td_ludo_v15.rich.v15_bot_eval import get_bot
        bot = get_bot(spec["name"], player_id=None)
        return lambda state, legal, hist: bot.select_move(state, list(legal))
    raise ValueError(kind)


def run_game(pick0, pick2, seed):
    import td_ludo_cpp as cpp
    random.seed(seed)
    state = cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    while not state.is_terminal and mc < MAX_MOVES:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            continue
        if state.current_dice_roll == 0:
            d = random.randint(1, 6)
            if d == 6:
                csix[cp] += 1
                if csix[cp] >= 3:
                    csix[cp] = 0
                    n = (cp + 1) % 4
                    while not state.active_players[n]:
                        n = (n + 1) % 4
                    state.current_player = n
                    state.current_dice_roll = 0
                    continue
            else:
                csix[cp] = 0
            state.current_dice_roll = d
        legal = cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        action = pick0(state, list(legal), None) if cp == 0 else pick2(state, list(legal), None)
        state = cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        w = cpp.get_winner(state)
        if w == 0: return "0"
        if w == 2: return "2"
    return "draw"


def run_matchup(args):
    """Worker: name_a vs name_b, gpo games per orientation. Returns result dict."""
    import torch
    torch.set_num_threads(1)
    name_a, name_b, gpo = args
    device = torch.device("cpu")
    pa = build_picker(PLAYERS[name_a], device)
    pb = build_picker(PLAYERS[name_b], device)
    aw = bw = dr = 0
    # orient 1: a=player0, b=player2
    for i in range(gpo):
        r = run_game(pa, pb, seed=i)
        if r == "0": aw += 1
        elif r == "2": bw += 1
        else: dr += 1
    # orient 2: b=player0, a=player2
    for i in range(gpo):
        r = run_game(pb, pa, seed=10_000 + i)
        if r == "0": bw += 1
        elif r == "2": aw += 1
        else: dr += 1
    total = aw + bw + dr
    return {"a": name_a, "b": name_b, "a_wins": aw, "b_wins": bw, "draws": dr,
            "games": total, "a_wr": 100.0 * aw / total, "b_wr": 100.0 * bw / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpo", type=int, default=1000, help="games per orientation (x2 = per pair)")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    tourney = ["v13.5", "v13.6", "v15.2", "MCTS"]
    pairs = [(tourney[i], tourney[j]) for i in range(len(tourney)) for j in range(i + 1, len(tourney))]
    extra = [("v15.2", "v15.2-latest")]
    all_matchups = [(a, b, args.gpo) for (a, b) in pairs + extra]

    print(f"4-way tournament: {tourney}")
    print(f"{len(pairs)} round-robin pairs + {len(extra)} extra, {2*args.gpo} games each "
          f"({len(all_matchups)*2*args.gpo} total), {args.workers} workers\n", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_matchup, m): (m[0], m[1]) for m in all_matchups}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"  [done {len(results)}/{len(all_matchups)}] {r['a']} {r['a_wins']}-{r['b_wins']} {r['b']} "
                  f"(draws {r['draws']}) | {r['a']} {r['a_wr']:.1f}%  ({time.time()-t0:.0f}s elapsed)", flush=True)

    # Build win matrix + ranking over the 4 tourney players
    wr = {p: {} for p in tourney}
    for r in results:
        if r["a"] in tourney and r["b"] in tourney:
            wr[r["a"]][r["b"]] = r["a_wr"]
            wr[r["b"]][r["a"]] = r["b_wr"]
    avg = {p: sum(wr[p].values()) / len(wr[p]) for p in tourney}
    ranked = sorted(tourney, key=lambda p: -avg[p])

    print("\n" + "=" * 64)
    print("4-WAY ROUND ROBIN — win% of ROW vs COLUMN")
    hdr = "        " + "".join(f"{c:>10}" for c in tourney)
    print(hdr)
    for p in tourney:
        row = f"{p:>8}" + "".join(f"{wr[p].get(c, '--'):>10}" if isinstance(wr[p].get(c), str)
                                  else f"{wr[p].get(c, 0):>10.1f}" for c in tourney)
        print(row)
    print("\nRANKING (avg win% across the other 3):")
    for i, p in enumerate(ranked, 1):
        print(f"  {i}. {p:<14} {avg[p]:.1f}%")
    print("=" * 64)

    ex_res = next((r for r in results if set((r["a"], r["b"])) == {"v15.2", "v15.2-latest"}), None)
    if ex_res:
        a, b = ex_res["a"], ex_res["b"]
        se = (ex_res["a_wr"] * ex_res["b_wr"] / ex_res["games"]) ** 0.5
        print(f"\nV15.2 best vs latest: {a} {ex_res['a_wr']:.1f}% ({ex_res['a_wins']}) "
              f"vs {b} {ex_res['b_wr']:.1f}% ({ex_res['b_wins']}), ±{se:.1f}pp")

    out = HERE / "tournament_4way_results.json"
    out.write_text(json.dumps({"results": results, "avg": avg, "ranked": ranked,
                               "gpo": args.gpo, "elapsed_sec": time.time() - t0}, indent=2))
    print(f"\nsaved {out} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
