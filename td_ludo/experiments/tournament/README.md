# Tournament — round-robin between Ludo agents

Round-robin tournament where every unordered pair of competitors plays
N games (default 2000). Seats rotate between games to control for
first-mover advantage. 2-player Ludo only.

## Competitors

Three competitor types, mixable on the CLI:

| Type | Backed by | Examples |
|---|---|---|
| `--hist`         | `OpponentRegistry` | V6_big, V6_1, V6_3, V10 |
| `--bots`         | `td_ludo.game.heuristic_bot` | Expert, Heuristic, Aggressive, Defensive, Racing, Random |
| `--add-model`    | Custom checkpoint + arch preset | V12.2, distilled-14ch, anything |

Architecture presets for `--add-model NAME:ARCH:PATH`:

- `v122` — V12.2 (3 ResBlocks × 128, V11 33ch encoder, attn)
- `v12_default` — V12 default (4 ResBlocks × 96, V11 33ch encoder, attn)
- `v10` — V10 (6 ResBlocks × 96, 28ch encoder)
- `v6_3` — V6.3 (10 ResBlocks × 128, 27ch encoder)
- `v6_1` — V6.1 (10 ResBlocks × 128, 24ch encoder)
- `v6_big` — V5-era (10 ResBlocks × 128, 17ch encoder)
- `v14_minimal` — distilled student (10 ResBlocks × 128, 14ch encoder)

## Examples

```bash
cd td_ludo

# All four historicals (defaults), 2000 games/pair
python -m experiments.tournament.run --hist all --games-per-pair 2000

# Add the current V12.2 + Expert bot
python -m experiments.tournament.run \
  --hist all \
  --bots Expert \
  --add-model V12_2:v122:play/model_weights/v12_2/model_latest.pt

# Full roster including the distilled student (when available)
python -m experiments.tournament.run \
  --hist all \
  --bots Expert,Random \
  --add-model V12_2:v122:play/model_weights/v12_2/model_latest.pt \
  --add-model Distill14:v14_minimal:experiments/distillation_14ch/student_14ch_final.pt \
  --games-per-pair 2000 \
  --output runs/tournament_full.json
```

## Output

- **Live progress** per pair (per-100-game updates, suppressible with `--quiet`).
- **Leaderboard**: aggregate WR across all pairs.
- **Head-to-head matrix**: row's win rate against col.
- **JSON dump** (with `--output`): per-pair stats including seat
  breakdown, suitable for downstream Bradley-Terry ELO computation.

## Notes

- Greedy (argmax) play for all agents. Reproducible given `--seed`.
- Single-game serial play. ~50–170 GPM depending on agent mix and
  hardware. For 6 competitors × 2000 games = 30K games this runs in
  ~3–10 hours on CPU. Acceptable for one-shot evaluations.
- 2-player only. 4-player tournament is a future variant (different
  pairings, different strategic dynamics).
- Token-attention models (V11.1, V12.x) are supported via
  `--add-model` with the `v122` or `v12_default` preset. The Hist_V11
  registry tag is intentionally absent because the V11 checkpoint we
  have uses non-default `attn_dim=64` and is deferred.
- V6.2 is not supported — it's a temporal transformer (sequence over
  K=16 past states) and needs per-game history tracking outside the
  agent abstraction. Deferred.
