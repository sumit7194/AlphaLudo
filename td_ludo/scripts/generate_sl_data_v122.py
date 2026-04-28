"""
V12.2 SL data generator.

Plays V12-latest (the trained 79.4% / 81% peak model from G=629K) against a
mixed opponent pool. For every state where V12 makes a decision, captures:

  - state33  : 33-channel V11 encoding (idle + streak channels included so
                V12.2 trains on the same encoder it'll use in RL)
  - policy   : V12's softmax output at this state (legal-masked)
  - legal_mask
  - won      : 1 if V12 (the captured-side player) eventually won, else 0
  - moves_remaining : own-side decisions left to game end
  - opp_id   : 0=V12 self-play / 1=Expert / 2=Heuristic / 3=Aggressive / 4=Defensive

Inference uses the V10 encoder (28ch) because V12-latest's conv_input is
trained on 28-channel inputs. Storage is the V11 encoding (33ch).

Game composition matches the V12.2 plan:
   SelfPlay 75% / Expert 15% / Heuristic 5% / Aggressive 3% / Defensive 2%

Usage:
    python scripts/generate_sl_data_v122.py \\
        --teacher play/model_weights/v12_final/model_latest.pt \\
        --target-states 500000 \\
        --batch-size 512
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

# Make project imports work regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import td_ludo_cpp as ludo_cpp
from td_ludo.models.v12_legacy import AlphaLudoV12Legacy
from td_ludo.game.heuristic_bot import get_bot

OPP_NAMES = {0: "SelfPlay", 1: "Expert", 2: "Heuristic", 3: "Aggressive", 4: "Defensive"}
OPP_ID_BY_NAME = {v: k for k, v in OPP_NAMES.items()}

# V12.2 game composition (must sum to 1.0)
DEFAULT_COMPOSITION = {
    "SelfPlay":   0.75,
    "Expert":     0.15,
    "Heuristic":  0.05,
    "Aggressive": 0.03,
    "Defensive":  0.02,
}

V12_PLAYER = 0   # V12 always plays as P0 (own captures come from cp == 0)
OPP_PLAYER = 2
MAX_TURNS = 400


def pick_opponent(composition):
    r = random.random()
    cum = 0.0
    for name, frac in composition.items():
        cum += frac
        if r <= cum:
            return name
    return list(composition.keys())[-1]


class V122DataGenerator:
    """Batched generation: V12 vs (V12 | Expert | Heuristic | Aggressive | Defensive).

    For each of `batch_size` parallel games:
      - At game start, draw an opponent type per the composition.
      - V12 plays as P0; opponent plays as P2.
      - On V12's turn, run V12 inference, capture (state33, policy, ...).
      - On opponent's turn:
          - SelfPlay: also run V12 inference, ALSO capture (we get 2x decisions).
          - Bot: bot.select_move() picks an action; nothing captured.
      - On terminal: finalize trajectories with won + moves_remaining.
    """

    def __init__(self, teacher_model, device, batch_size, composition):
        self.model = teacher_model
        self.device = device
        self.batch_size = batch_size
        self.composition = composition
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.opp_name = [pick_opponent(composition) for _ in range(batch_size)]
        self.opp_bots = [get_bot(name, player_id=OPP_PLAYER) if name != "SelfPlay"
                         else None for name in self.opp_name]
        # trajectories[i] is a list of (capturing_player, state33, policy, mask)
        self.trajectories = [[] for _ in range(batch_size)]
        self.step_count = np.zeros(batch_size, dtype=np.int32)

    def _reset_game(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        name = pick_opponent(self.composition)
        self.opp_name[i] = name
        self.opp_bots[i] = get_bot(name, player_id=OPP_PLAYER) if name != "SelfPlay" else None
        self.trajectories[i] = []
        self.step_count[i] = 0

    def _maybe_roll(self, i):
        """Roll dice for current player, handling triple-six. Returns True if a
        legal-move decision can be made this step, False if turn was passed."""
        game = self.games[i]
        cp = game.current_player
        if game.current_dice_roll == 0:
            roll = random.randint(1, 6)
            game.current_dice_roll = roll
            if roll == 6:
                self.consec_sixes[i, cp] += 1
            else:
                self.consec_sixes[i, cp] = 0
            if self.consec_sixes[i, cp] >= 3:
                # Triple 6 — turn lost
                nxt = (cp + 1) % 4
                while not game.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                game.current_player = nxt
                game.current_dice_roll = 0
                self.consec_sixes[i, cp] = 0
                return False

        legal = ludo_cpp.get_legal_moves(game)
        if not legal:
            cp = game.current_player
            nxt = (cp + 1) % 4
            while not game.active_players[nxt]:
                nxt = (nxt + 1) % 4
            game.current_player = nxt
            game.current_dice_roll = 0
            return False
        return True

    def _v12_capture_states(self):
        """Collect the V12-decision games — both V12 self-play turns and V12's
        own (P0) turns vs bot opponents. Returns a list of (i, cp, state33,
        state28, mask, legal) so we can run one batched inference."""
        decisions = []
        for i, game in enumerate(self.games):
            if game.is_terminal or self.step_count[i] >= MAX_TURNS:
                continue
            if not self._maybe_roll(i):
                # Turn was passed via no-moves or triple-six. Loop continues.
                continue
            cp = int(game.current_player)
            opp_is_v12 = (self.opp_name[i] == "SelfPlay")
            # V12 captures: always when cp == V12_PLAYER, and also when opponent
            # is also V12 (self-play). When opp is a bot and cp == OPP_PLAYER,
            # the bot will play instead — we don't capture.
            if cp == V12_PLAYER or opp_is_v12:
                legal = ludo_cpp.get_legal_moves(game)
                mask = np.zeros(4, dtype=np.float32)
                for m in legal:
                    mask[m] = 1.0
                state28 = np.array(ludo_cpp.encode_state_v10(game), dtype=np.float32)
                state33 = np.array(ludo_cpp.encode_state_v11(game), dtype=np.float32)
                decisions.append((i, cp, state33, state28, mask, legal))
        return decisions

    def _bot_play_step(self):
        """Bot turns (opp != SelfPlay, cp == OPP_PLAYER). Apply bot moves; no
        capture. Called BEFORE _v12_capture_states each tick to advance bot
        turns until it's V12's turn."""
        for i, game in enumerate(self.games):
            if game.is_terminal or self.step_count[i] >= MAX_TURNS:
                continue
            opp_is_v12 = (self.opp_name[i] == "SelfPlay")
            if opp_is_v12:
                continue
            # Loop while it's bot's turn
            while (not game.is_terminal and self.step_count[i] < MAX_TURNS
                   and int(game.current_player) == OPP_PLAYER):
                if not self._maybe_roll(i):
                    continue
                legal = ludo_cpp.get_legal_moves(game)
                if not legal:
                    continue
                action = self.opp_bots[i].select_move(game, legal)
                if action not in legal:
                    action = legal[0]
                self.games[i] = ludo_cpp.apply_move(game, action)
                game = self.games[i]
                self.step_count[i] += 1

    def play_step(self):
        """One outer tick: bot advances, then V12 batched inference + apply.
        Returns finalized samples for any games that ended this tick."""
        # Advance bot turns first (they don't need batched inference).
        self._bot_play_step()

        decisions = self._v12_capture_states()
        if not decisions:
            return self._collect_finals()

        # Batch V12 inference (single forward pass over all decisions)
        states28 = torch.from_numpy(
            np.stack([d[3] for d in decisions])
        ).to(self.device, dtype=torch.float32)
        masks = torch.from_numpy(
            np.stack([d[4] for d in decisions])
        ).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            policy, _win, _moves = self.model(states28, masks)
            probs = policy.cpu().numpy()
            # Sample stochastically (temperature 1.0) to inject decision diversity.
            actions = torch.multinomial(policy, num_samples=1).squeeze(1).cpu().numpy()

        # Apply moves + record trajectories
        for k, (i, cp, state33, _state28, mask, legal) in enumerate(decisions):
            action = int(actions[k])
            if mask[action] == 0:
                action = legal[0]
            self.trajectories[i].append({
                "capturing_player": cp,
                "state33": state33,
                "policy": probs[k].copy(),
                "legal_mask": mask.copy(),
            })
            self.games[i] = ludo_cpp.apply_move(self.games[i], action)
            self.step_count[i] += 1

        return self._collect_finals()

    def _collect_finals(self):
        completed = []
        for i in range(self.batch_size):
            game = self.games[i]
            if game.is_terminal:
                completed.extend(self._finalize(i))
                self._reset_game(i)
            elif self.step_count[i] >= MAX_TURNS:
                # Timeout — discard the trajectory (no winner).
                self._reset_game(i)
        return completed

    def _finalize(self, i):
        winner = int(ludo_cpp.get_winner(self.games[i]))
        if winner < 0:
            return []
        opp_id = OPP_ID_BY_NAME[self.opp_name[i]]

        # Per-player remaining-moves counts
        per_player_total = {}
        for step in self.trajectories[i]:
            p = step["capturing_player"]
            per_player_total[p] = per_player_total.get(p, 0) + 1
        cursor = {p: 0 for p in per_player_total}

        samples = []
        for step in self.trajectories[i]:
            p = step["capturing_player"]
            cursor[p] += 1
            samples.append({
                "state33": step["state33"],
                "policy": step["policy"],
                "legal_mask": step["legal_mask"],
                "won": 1 if p == winner else 0,
                "moves_remaining": per_player_total[p] - cursor[p],
                "opp_id": opp_id,
            })
        return samples


def main():
    ap = argparse.ArgumentParser(description="Generate V12.2 SL data using V12 as teacher")
    ap.add_argument("--teacher", default="play/model_weights/v12_final/model_latest.pt",
                    help="V12 checkpoint (28ch in_channels)")
    ap.add_argument("--output-dir", default="checkpoints/sl_data_v122")
    ap.add_argument("--target-states", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=10_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[V12.2 Data] Device: {device}")

    # Load V12 teacher (legacy architecture matches the trained checkpoint)
    print(f"[V12.2 Data] Loading teacher: {args.teacher}")
    teacher = AlphaLudoV12Legacy(num_res_blocks=4, num_channels=96, num_attn_layers=2,
                                 num_heads=4, ffn_ratio=4, dropout=0.0, in_channels=28)
    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    teacher.load_state_dict(sd)
    teacher.to(device).eval()
    print(f"[V12.2 Data] Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[V12.2 Data] Output:  {args.output_dir}")
    print(f"[V12.2 Data] Target:  {args.target_states:,} states")
    print(f"[V12.2 Data] Mix:     {DEFAULT_COMPOSITION}")

    gen = V122DataGenerator(teacher, device, args.batch_size, DEFAULT_COMPOSITION)

    all_states = []; all_policies = []; all_masks = []
    all_won = []; all_moves = []; all_opp = []
    chunk_idx = 0; total = 0
    opp_counts = {n: 0 for n in DEFAULT_COMPOSITION}
    t0 = time.time(); last_report = t0

    print(f"[V12.2 Data] Collecting...")
    while total < args.target_states:
        completed = gen.play_step()
        for s in completed:
            all_states.append(s["state33"])
            all_policies.append(s["policy"])
            all_masks.append(s["legal_mask"])
            all_won.append(s["won"])
            all_moves.append(s["moves_remaining"])
            all_opp.append(s["opp_id"])
            opp_counts[OPP_NAMES[s["opp_id"]]] += 1
            total += 1

        while len(all_states) >= args.chunk_size:
            path = os.path.join(args.output_dir, f"chunk_{chunk_idx:04d}.npz")
            np.savez_compressed(
                path,
                states=np.stack(all_states[:args.chunk_size]).astype(np.float32),
                policies=np.stack(all_policies[:args.chunk_size]).astype(np.float32),
                legal_masks=np.stack(all_masks[:args.chunk_size]).astype(np.float32),
                won=np.array(all_won[:args.chunk_size], dtype=np.int8),
                moves_remaining=np.array(all_moves[:args.chunk_size], dtype=np.int32),
                opp_id=np.array(all_opp[:args.chunk_size], dtype=np.int8),
            )
            for lst in (all_states, all_policies, all_masks, all_won, all_moves, all_opp):
                del lst[:args.chunk_size]
            chunk_idx += 1

        now = time.time()
        if now - last_report > 15:
            elapsed = now - t0
            rate = total / max(1e-3, elapsed)
            eta = (args.target_states - total) / max(1e-3, rate)
            mix_str = " ".join(f"{n[:3]}={c}" for n, c in opp_counts.items())
            print(f"  [{elapsed:.0f}s] {total:,}/{args.target_states:,} "
                  f"| {rate:.0f} states/s | ETA {eta:.0f}s | {mix_str}", flush=True)
            last_report = now

    # Final chunk
    if all_states:
        path = os.path.join(args.output_dir, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            path,
            states=np.stack(all_states).astype(np.float32),
            policies=np.stack(all_policies).astype(np.float32),
            legal_masks=np.stack(all_masks).astype(np.float32),
            won=np.array(all_won, dtype=np.int8),
            moves_remaining=np.array(all_moves, dtype=np.int32),
            opp_id=np.array(all_opp, dtype=np.int8),
        )
        chunk_idx += 1

    elapsed = time.time() - t0
    print(f"\n[V12.2 Data] DONE  {total:,} states in {chunk_idx} chunks "
          f"({elapsed:.0f}s, {total/elapsed:.0f} states/s)")
    print(f"[V12.2 Data] Mix actuals: {opp_counts}")


if __name__ == "__main__":
    main()
