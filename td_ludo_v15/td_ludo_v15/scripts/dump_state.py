"""CLI: dump a V15 game state with side-by-side board + encoding.

Usage:
    python -m td_ludo_v15.scripts.dump_state                       # seed 42, 0 moves
    python -m td_ludo_v15.scripts.dump_state --seed 1 --moves 25
    python -m td_ludo_v15.scripts.dump_state --seed 7 --moves 50 --pov 2
    python -m td_ludo_v15.scripts.dump_state --output frame_xyz.txt --seed 3 --moves 80

Plays `--moves` random moves with a deterministic RNG seed, then renders
the resulting state as a side-by-side board + triplet encoding for visual
verification.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import td_ludo_v15_cpp as _cpp
from ..game.state import V15GameWrapper
from ..viz.board_viewer import render_side_by_side


def play_random_moves(seed: int, n_moves: int) -> V15GameWrapper:
    """Build a V15 game state by playing `n_moves` random moves from seed."""
    rng = random.Random(seed)
    g = V15GameWrapper.new_2p()
    moves_made = 0
    safety_iter_cap = n_moves * 50  # avoid infinite loop on degenerate rng
    iters = 0
    while moves_made < n_moves and not g.is_terminal and iters < safety_iter_cap:
        iters += 1
        d = rng.randint(1, 6)
        g.set_dice(d)
        if g.dice == 0:
            # Forfeit happened
            continue
        cells = g.get_legal_source_cells()
        if not cells:
            g.pass_turn()
            continue
        chosen = rng.choice(cells)
        g.apply_move_from_cell(*chosen)
        moves_made += 1
    return g


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for play (default 42)")
    p.add_argument("--moves", type=int, default=0, help="Number of random moves to play (default 0 = initial state)")
    p.add_argument("--pov", type=int, default=None,
                   help="POV player ID for encoding (default: current_player at time of dump)")
    p.add_argument("--output", type=str, default=None,
                   help="Write to this file instead of stdout")
    args = p.parse_args(argv)

    g = play_random_moves(args.seed, args.moves)
    pov = args.pov if args.pov is not None else int(g.current_player)
    out = render_side_by_side(g.state, pov_player=pov)

    # Add a header line with metadata
    header = (
        f"V15 STATE DUMP — seed={args.seed} moves={args.moves} "
        f"pov={pov} current_player={int(g.current_player)} "
        f"terminal={int(g.is_terminal)}"
    )
    full = header + "\n" + out

    if args.output:
        Path(args.output).write_text(full + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(full)


if __name__ == "__main__":
    main()
