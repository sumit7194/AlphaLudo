"""AlphaLudo v15.2 champion  vs  TypeSafe Jev (System One) via OpenRouter.

Jev is non-autoregressive: it takes program state + a typed question and returns
a Choice with per-option probabilities and a confidence, in one parallel pass.
That maps directly onto "pick one of these legal Ludo moves", so Jev plays
without ever being shown the game's rules in prose or asked to reason.

FAIRNESS: Jev cannot see the board tensor our model sees, so each legal move is
described by its FACTUAL CONSEQUENCES (from/to, capture, safety, home entry, and
how many opponent tokens could hit the landing square next turn). Those are
things a competent player reads off the board — not strategic advice. No option
is ever labelled good or bad, and the option order is shuffled each turn so
position carries no signal.

Game loop matches td_ludo's own training/eval loop exactly (engine owns turn
order, including the extra turn on a 6), so our model plays under the same rules
it was trained and evaluated on.
"""
import argparse, json, os, random, sys, time, urllib.request

sys.path.insert(0, "/Users/sumit/Github/AlphaLudo/td_ludo")
sys.path.insert(0, "/Users/sumit/Github/AlphaLudo/td_ludo_v15")

import numpy as np
import torch
import td_ludo_cpp as C
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.game.cells import NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov
from td_ludo_v15.models.v15 import V15GraphTransformer
from td_ludo.game.strong_bots import _is_safe, _absolute_pos

BASE, GOAL, TRACK = -1, 56, 52
OR_URL = "https://openrouter.ai/api/v1/systemone"
CHAMP = ("/Users/sumit/Github/AlphaLudo/checkpoint_backups/v152_gaeterminal_2026-06-19/"
         "v152_gaeterminal_BEST_eval84.65pct_2026-06-19.pt")


# ── our model ──────────────────────────────────────────────────────────────
def load_model(path):
    m = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256,
                            history_len=1)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    if any(k.endswith(".a") for k in sd):          # v16 two-property ckpt
        from td_ludo_v15.models.v16 import V16GraphTransformer
        m = V16GraphTransformer(d_model=128, n_heads=4, n_layers=4,
                                ffn_dim=256, history_len=1)
    m.load_state_dict(sd, strict=False)
    return m.eval()


def legal_cell_mask(g, cp, legal):
    mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
    c2t = {}
    for tok in legal:
        p = int(g.player_positions[cp][tok])
        idx = cell_to_index(*position_to_cell_in_pov(BASE if p == BASE else p, cp, cp))
        mask[idx] = 1.0
        c2t[idx] = tok
    return mask, c2t


@torch.no_grad()
def model_pick(model, g, cp, legal):
    if len(legal) == 1:
        return int(legal[0]), None
    mask, c2t = legal_cell_mask(g, cp, legal)
    x = torch.from_numpy(encode_frame(g, pov_player=cp)[None, None]).float()
    pol, _ = model(x, torch.from_numpy(mask).unsqueeze(0))
    pr = pol.squeeze(0).numpy() * mask
    return (int(c2t[int(pr.argmax())]) if pr.sum() > 0 else int(legal[0])), None


# ── describing the board to Jev ────────────────────────────────────────────
def where(p):
    if p == BASE:  return "in base (not yet on the board)"
    if p >= GOAL:  return "finished (home)"
    if p >= TRACK: return f"in the final home lane, {GOAL - p} steps from finishing"
    return f"on square {p} of 52, {GOAL - p} steps from finishing"


def threats_to(g, cp, landing_rel):
    """How many opponent tokens could land on `landing_rel` with a 1-6 roll."""
    opp = (cp + 2) % 4
    if landing_rel >= TRACK:
        return 0                                   # home lane is unreachable
    la = _absolute_pos(cp, landing_rel)
    if la is None:
        return 0
    n = 0
    for t in range(4):
        q = int(g.player_positions[opp][t])
        if not (0 <= q < TRACK):
            continue
        qa = _absolute_pos(opp, q)
        if qa is None:
            continue
        if 1 <= (la - qa) % TRACK <= 6:
            n += 1
    return n


def describe_move(g, cp, tok, dice):
    opp = (cp + 2) % 4
    p = int(g.player_positions[cp][tok])
    frm = where(p)
    if p == BASE:
        to_rel = 0
        head = f"Bring token {tok} out of base onto square 0."
    else:
        to_rel = p + dice
        if to_rel > GOAL:
            return None
        head = (f"Move token {tok} from square {p} to square {to_rel}."
                if to_rel < GOAL else
                f"Move token {tok} from square {p} HOME — it finishes and is safe forever.")
    bits = [head]
    if to_rel < GOAL:
        la = _absolute_pos(cp, to_rel) if to_rel < TRACK else None
        caps = []
        if la is not None and not _is_safe(cp, to_rel):
            for t in range(4):
                q = int(g.player_positions[opp][t])
                if 0 <= q < TRACK and _absolute_pos(opp, q) == la:
                    caps.append(t)
        if caps:
            bits.append(f"CAPTURES {len(caps)} opponent token(s), sending them back to base.")
        if to_rel >= TRACK:
            bits.append("Enters the final home lane, where it can never be captured.")
        elif _is_safe(cp, to_rel):
            bits.append("Lands on a safe square where it cannot be captured.")
        else:
            th = threats_to(g, cp, to_rel)
            bits.append(f"After landing, {th} opponent token(s) could capture it next turn."
                        if th else "No opponent token could reach it next turn.")
        bits.append(f"It would then be {GOAL - to_rel} steps from finishing.")
    return " ".join(bits), frm




def _rel_in_pov(g, owner, owner_pos, pov):
    """Where an `owner` token physically sits, in `pov`'s square numbering."""
    a = _absolute_pos(owner, owner_pos)
    if a is None:
        return None
    for r in range(TRACK):
        if _absolute_pos(pov, r) == a:
            return r
    return None


# ── full board rendering (Jev sees the whole board, as text) ───────────────
def render_board(g, cp):
    """15x15 ASCII board + a 52-square track map, both in the mover's POV.

    Everything is expressed in YOUR relative coordinates: square N is N steps
    along your own path, so "square 23 holds an opponent token" is directly
    actionable without the model having to reason about whose frame it is in.
    """
    opp = (cp + 2) % 4
    grid = [[" " for _ in range(15)] for _ in range(15)]

    # lay down the path: your route 0..51, then both home lanes
    for pos in range(TRACK):
        r, c = position_to_cell_in_pov(pos, cp, cp)
        grid[r][c] = "*" if _is_safe(cp, pos) else "."
    for pos in range(TRACK, GOAL):
        r, c = position_to_cell_in_pov(pos, cp, cp)
        grid[r][c] = "y"
        r, c = position_to_cell_in_pov(pos, opp, cp)
        grid[r][c] = "o"

    # tokens (stacks shown as a digit)
    def place(player, ch):
        for t in range(4):
            q = int(g.player_positions[player][t])
            if not (0 <= q < GOAL):
                continue
            r, c = position_to_cell_in_pov(q, player, cp)
            cur = grid[r][c]
            grid[r][c] = ch if cur in ".*yo" else ("2" if cur in "YO" else "3")
    place(cp, "Y"); place(opp, "O")

    rows = ["   " + "".join(f"{c%10}" for c in range(15))]
    rows += [f"{r:2d} " + "".join(grid[r]) for r in range(15)]
    ascii_board = "\n".join(rows)

    # linear track in YOUR coordinates — unambiguous, no frame reasoning needed
    track = []
    for pos in range(TRACK):
        a = _absolute_pos(cp, pos)
        ch = "*" if _is_safe(cp, pos) else "."
        for t in range(4):
            q = int(g.player_positions[cp][t])
            if 0 <= q < TRACK and _absolute_pos(cp, q) == a: ch = "Y"
        for t in range(4):
            q = int(g.player_positions[opp][t])
            if 0 <= q < TRACK and _absolute_pos(opp, q) == a: ch = "O" if ch != "Y" else "X"
        track.append(ch)
    return ascii_board, "".join(track)


def build_request(g, cp, legal, dice, model_slug, shuffle_rng):
    opp = (cp + 2) % 4
    mine = [int(g.player_positions[cp][t]) for t in range(4)]
    theirs = [int(g.player_positions[opp][t]) for t in range(4)]
    ascii_board, track = render_board(g, cp)
    state = {
        "game": ("Ludo, 2 players. You move along squares 0..51 of a shared loop, "
                 "then up your private home lane (52..55), then finish at 56. "
                 "Landing on an opponent token sends it back to base. Tokens on "
                 "safe squares and in home lanes cannot be captured. First player "
                 "to finish all 4 tokens wins."),
        "your_dice_roll": dice,
        "board_15x15": ascii_board,
        "board_legend": {
            "Y": "your token", "O": "opponent token",
            "2/3": "that many tokens stacked on one square",
            ".": "ordinary path square", "*": "safe square (no capture possible)",
            "y": "your home lane", "o": "opponent home lane",
            " ": "not part of the board",
        },
        "track_52_squares_from_your_start": track,
        "track_legend": ("Index i = square i counting from YOUR start. "
                         "Y=your token, O=opponent token, X=both, "
                         "*=empty safe square, .=empty ordinary square."),
        "your_tokens": [{"id": t, "square": (mine[t] if 0 <= mine[t] < GOAL else None),
                         "status": where(mine[t])} for t in range(4)],
        "your_tokens_finished": sum(1 for p in mine if p >= GOAL),
        # Opponent positions in BOTH frames. Their own square number measures
        # their progress; `on_your_track_square` is where they physically sit on
        # the loop in YOUR numbering, which is the frame the track map and every
        # move option uses. Giving only one frame forces a coordinate
        # translation that has nothing to do with playing well.
        "opponent_tokens": [{
            "id": t,
            "their_progress": where(theirs[t]),
            "on_your_track_square": (
                _rel_in_pov(g, opp, theirs[t], cp) if 0 <= theirs[t] < TRACK else None),
        } for t in range(4)],
        "opponent_tokens_finished": sum(1 for p in theirs if p >= GOAL),
    }
    opts, order = {}, list(legal)
    shuffle_rng.shuffle(order)                      # position carries no signal
    for tok in order:
        d = describe_move(g, cp, tok, dice)
        if d:
            opts[f"token_{tok}"] = d[0]
    return {
        "model": model_slug,
        "state": state,
        "questions": {"move": {
            "type": "choice",
            "instructions": ("You are playing Ludo and must move one token. "
                             "Choose the move that best improves your chance of "
                             "winning the game."),
            "criteria": opts,
        }},
    }


def jev_pick(g, cp, legal, dice, key, model_slug, rng, retries=3):
    if len(legal) == 1:
        return int(legal[0]), {"single_legal": True}
    body = build_request(g, cp, legal, dice, model_slug, rng)
    req = urllib.request.Request(
        OR_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            ans = out["answers"]["move"]
            tok = int(ans["choice"].split("_")[1])
            if tok not in legal:                    # never trust it blindly
                tok = int(max(legal, key=lambda t: ans["probabilities"].get(f"token_{t}", 0)))
            return tok, {"probs": ans["probabilities"], "conf": ans["confidence"],
                         "cost": out.get("usage", {}).get("cost", 0)}
        except Exception as e:
            if a == retries - 1:
                return int(rng.choice(list(legal))), {"error": str(e)[:80]}
            time.sleep(1.5 * (a + 1))


# ── the game ───────────────────────────────────────────────────────────────
def play(model, key, model_slug, model_seat, seed, max_steps=400, verbose=False):
    rng = random.Random(seed)
    g = C.create_initial_state_2p()
    jev_seat = 2 if model_seat == 0 else 0
    steps = 0
    jev_calls, jev_cost, jev_conf, errs = 0, 0.0, [], 0
    log = []
    while not g.is_terminal and steps < max_steps:
        cp = int(g.current_player)
        if not g.active_players[cp]:
            g.current_player = (cp + 2) % 4; continue
        if g.current_dice_roll == 0:
            g.current_dice_roll = rng.randint(1, 6)
        dice = int(g.current_dice_roll)
        legal = list(C.get_legal_moves(g))
        if not legal:
            g.current_player = (cp + 2) % 4; g.current_dice_roll = 0; continue
        if cp == model_seat:
            tok, _ = model_pick(model, g, cp, legal)
            who = "v15.2"
        else:
            tok, info = jev_pick(g, cp, legal, dice, key, model_slug, rng)
            who = "Jev"
            if not info.get("single_legal"):
                jev_calls += 1
                jev_cost += info.get("cost", 0) or 0
                if "conf" in info: jev_conf.append(info["conf"])
                if "error" in info: errs += 1
                if verbose:
                    pr = info.get("probs", {})
                    top = " ".join(f"{k.split('_')[1]}:{v:.2f}" for k, v in
                                   sorted(pr.items(), key=lambda x: -x[1])[:3])
                    print(f"    [Jev] d{dice} legal{legal} -> {tok}  "
                          f"({top}) conf {info.get('conf',0):.2f}", flush=True)
        log.append((who, dice, tok))
        g = C.apply_move(g, int(tok)); steps += 1
    sc = [sum(1 for t in range(4) if int(g.player_positions[p][t]) >= GOAL)
          for p in range(4)]
    winner = None
    if g.is_terminal:
        winner = model_seat if sc[model_seat] >= sc[jev_seat] else jev_seat
    return {"winner": winner, "model_seat": model_seat, "jev_seat": jev_seat,
            "scores": {"v15.2": sc[model_seat], "Jev": sc[jev_seat]},
            "steps": steps, "jev_calls": jev_calls, "jev_cost": jev_cost,
            "jev_mean_conf": (sum(jev_conf)/len(jev_conf)) if jev_conf else 0.0,
            "errors": errs, "terminal": bool(g.is_terminal)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--ckpt", default=CHAMP)
    ap.add_argument("--jev-model", default="typesafe/jev-1.13")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--key-file", required=True)
    a = ap.parse_args()

    key = open(a.key_file).read().strip()
    model = load_model(a.ckpt)
    print(f"  ours : {os.path.basename(a.ckpt)}")
    print(f"  jev  : {a.jev_model}\n")

    wins = {"v15.2": 0, "Jev": 0, "unfinished": 0}
    tot_cost, tot_calls, confs = 0.0, 0, []
    for i in range(a.games):
        seat = 0 if i % 2 == 0 else 2               # alternate seats
        t0 = time.time()
        r = play(model, key, a.jev_model, seat, a.seed + i, verbose=a.verbose)
        if not r["terminal"]:
            wins["unfinished"] += 1; w = "unfinished"
        else:
            w = "v15.2" if r["winner"] == r["model_seat"] else "Jev"
            wins[w] += 1
        tot_cost += r["jev_cost"]; tot_calls += r["jev_calls"]
        if r["jev_mean_conf"]: confs.append(r["jev_mean_conf"])
        print(f"  game {i+1}: {w:10s} tokens home  v15.2 {r['scores']['v15.2']} - "
              f"{r['scores']['Jev']} Jev | {r['steps']} plies | "
              f"{r['jev_calls']} Jev decisions, conf {r['jev_mean_conf']:.2f}"
              + (f", {r['errors']} errors" if r["errors"] else "")
              + f" | {time.time()-t0:.0f}s", flush=True)

    print(f"\n  RESULT  v15.2 {wins['v15.2']} - {wins['Jev']} Jev"
          + (f"  ({wins['unfinished']} unfinished)" if wins["unfinished"] else ""))
    print(f"  Jev: {tot_calls} decisions, mean confidence "
          f"{(sum(confs)/len(confs) if confs else 0):.3f}, total cost ${tot_cost:.5f}")


if __name__ == "__main__":
    main()
