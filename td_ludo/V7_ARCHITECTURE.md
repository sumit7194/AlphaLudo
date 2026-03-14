# AlphaLudo V7 — Sequence Transformer Architecture

## Why V7?

V6 was a CNN (15x15x17 spatial tensor, 3M params) that plateaued at 73-77% win rate. Mechanistic interpretability experiments revealed:

- The model is **purely reactive** — it reads the board, reacts to the current dice, and makes locally optimal moves
- **Zero temporal reasoning** — no velocity tracking, no threat persistence, no multi-turn planning
- The 15x15 grid is wasteful — Ludo is fundamentally a **1D circular track**

V7 replaces the CNN with a **Transformer over a context window of past turns**, enabling cross-turn reasoning, threat tracking, and strategic planning.

---

## Model Overview

| Property | V6 (CNN) | V7 (Transformer) |
|---|---|---|
| Input | 15x15x17 spatial tensor | 1D state vector per turn |
| Architecture | 10 ResBlocks, 128 channels | 4-layer Transformer, 128 dim |
| Parameters | ~3M | ~855K |
| Context | Single board state | Last 16 turns |
| Reasoning | Reactive (current state only) | Temporal (multi-turn history) |

```
File: td_ludo/src/model_v7.py
Class: AlphaLudoV7(context_length=16, embed_dim=128, num_heads=4, num_layers=4)
```

---

## Input Representation

### Per-Turn State (encoded by `state_encoder_1d.py`)

Each turn is represented as two components:

#### 1. Token Positions — `(8,)` int64

Integer indices into `nn.Embedding(59, 128)`:

```
Index 0-3: My 4 token positions
Index 4-7: Opponent's 4 token positions
```

Position encoding (0-58):
```
 0     = Locked in base
 1-51  = Main path (51 squares, player-relative)
 53-57 = Home stretch (5 squares)
 58    = Home (scored)
```

Note: Position 52 is unused (gap between path and home stretch).

The C++ engine uses different conventions internally:
```
C++ → V7 remapping:
  -1      → 0       (base)
  0-50    → 1-51    (path)
  51-55   → 53-57   (home stretch)
  99      → 58      (home)
```

#### 2. Continuous Features — `(9,)` float32

```
[0]   opp_locked_frac   — Fraction of opponent tokens in base (0.0 - 1.0)
[1]   my_locked_frac    — Fraction of my tokens in base (0.0 - 1.0)
[2]   score_diff        — (my_score - max_opp_score) / 4.0
[3-8] dice_onehot       — One-hot encoding of current dice roll (1-6)
```

#### 3. Historical Action — `(1,)` int64

The action taken on the **previous** turn, used as input to the current turn's encoding:
```
0-3 = Moved token 0/1/2/3
4   = Pass / no previous action (first turn)
```

---

## Context Window

The model sees the **last K=16 turns** of the current player's history. This is managed by `TurnHistory` in `game_player_v7.py`.

```
Turn sequence: [pad, pad, ..., turn_t-3, turn_t-2, turn_t-1, turn_t]
                |-- padding --|  |----------- valid turns ----------|
```

- Padding is placed at the **start** of the sequence (left-padded)
- A `seq_mask` tensor marks padded positions as `True`
- The model uses **causal attention** — each turn can only attend to itself and earlier turns
- A **learned pad token** replaces padded positions (not just zeros)

### What the context window enables

- **Velocity estimation**: "Is the opponent advancing fast or stuck?"
- **Threat persistence**: "That opponent token has been near mine for 3 turns"
- **Strategy tracking**: "I've been spawning tokens — should I switch to advancing?"
- **Opponent modeling**: "The opponent keeps capturing — play defensively"

---

## Model Architecture

### TurnEncoder

Encodes a single turn's state into a fixed-size embedding:

```
token_positions (B, 8) ──→ nn.Embedding(59, 128) ──→ (B, 8, 128)  ─┐
                                                                      │
continuous (B, 9) ────────→ Linear(9, 128) ─────────→ (B, 1, 128)  ─┤ concat → (B, 10, 128)
                                                                      │           │
action (B,) ──────────────→ nn.Embedding(5, 128) ──→ (B, 1, 128)  ─┘      mean pool
                                                                              │
                                                                        (B, 128)
                                                                              │
                                                                    Linear + LayerNorm
                                                                              │
                                                                   turn_embedding (B, 128)
```

### AlphaLudoV7 (Main Model)

```
Input per turn:
  token_positions: (B, K, 8)  int64
  continuous:      (B, K, 9)  float32
  actions:         (B, K)     int64
  seq_mask:        (B, K)     bool (True = padding)
  legal_mask:      (B, 4)     float32 (1.0 = legal move)

                    ┌─────────────────────────────────────────┐
                    │  For each of K turns:                    │
                    │    TurnEncoder(tok, cont, act) → (B, E) │
                    └──────────────┬──────────────────────────┘
                                   │
                           (B, K, 128) turn embeddings
                                   │
                         + temporal position embeddings
                         + learned pad token for masked positions
                                   │
                    ┌──────────────┴──────────────────────────┐
                    │  Transformer Encoder (4 layers)          │
                    │    - 4 attention heads                   │
                    │    - 512 FFN dim (4x embed)              │
                    │    - GELU activation                     │
                    │    - Pre-norm (LayerNorm before attn)    │
                    │    - Causal attention mask                │
                    │    - Dropout 0.1                         │
                    └──────────────┬──────────────────────────┘
                                   │
                          Extract last valid turn's output
                                   │
                              (B, 128)
                            ┌──────┴──────┐
                            │             │
                    ┌───────┴───────┐  ┌──┴──────────┐
                    │  Policy Head  │  │  Value Head  │
                    │  Linear(128)  │  │  Linear(128) │
                    │  ReLU         │  │  ReLU        │
                    │  Linear(4)    │  │  Linear(1)   │
                    │  + legal mask │  │              │
                    │  softmax      │  │              │
                    └───────┬───────┘  └──────┬───────┘
                            │                 │
                     policy (B, 4)      value (B, 1)
```

### Key Design Choices

- **Pre-norm**: LayerNorm before attention/FFN (not after) — more stable for RL training
- **Causal mask**: Turn `i` can only attend to turns `≤ i` — prevents information leakage from future turns
- **Mean pooling** within turns (not CLS token) — treats all 10 input slots equally
- **Last valid turn** extraction — handles variable-length sequences correctly
- **No tanh on value head** — unbounded value prediction (same as V6)

---

## Training Pipeline

### Files

| File | Purpose |
|---|---|
| `src/state_encoder_1d.py` | Converts GameState → 1D vector |
| `src/model_v7.py` | Transformer model (855K params) |
| `src/game_player_v7.py` | Batched game player with context window |
| `src/trainer_v7.py` | PPO trainer for sequence inputs |
| `evaluate_v7.py` | Evaluator (greedy policy vs bots) |
| `train_v7.py` | Main training entrypoint |

### PPO Training

Same algorithm as V6 with sequence-adapted data flow:

- **Buffer**: Collect 64 complete games before PPO update (configurable)
- **PPO epochs**: 3 passes over collected data
- **Mini-batch size**: 256 steps
- **Clip epsilon**: 0.2
- **Discount**: gamma = 0.999
- **Reward shaping**: Dense v1 rewards (spawn +0.05, forward +0.005, home stretch +0.10, score +0.40, capture +0.20, killed -0.20, win +1.0, loss -1.0)

### Trajectory Data

Each step in a trajectory stores the **full context window snapshot**:

```python
{
    'token_positions': (K, 8) int64,    # context window of token positions
    'continuous':      (K, 9) float32,  # context window of continuous features
    'actions_seq':     (K,)   int64,    # context window of past actions
    'seq_mask':        (K,)   bool,     # padding mask
    'action':          int,             # action taken this step
    'legal_mask':      (4,)   float32,  # legal moves
    'old_log_prob':    float,           # for PPO importance sampling
    'temperature':     float,           # exploration temperature
}
```

### Running Training

```bash
cd td_ludo

# Fresh start (random weights)
td_env/bin/python train_v7.py --no-dashboard

# Resume from checkpoint
td_env/bin/python train_v7.py --resume

# With SL warmstart (place model_sl.pt in checkpoint dir first)
td_env/bin/python train_v7.py

# Quick test run
TD_LUDO_MODE=TEST td_env/bin/python train_v7.py --games 50

# Evaluation only
td_env/bin/python train_v7.py --eval-only
```

Checkpoints save to `checkpoints/ac_v7_transformer/` (PROD) or `checkpoints_test/ac_v7_transformer/` (TEST).

---

## Planned Roadmap

1. **PPO RL from random** (current) — validate architecture, target 60%+ WR
2. **SL warmstart → PPO RL** — generate bot-vs-bot data, supervised pretrain, then RL fine-tune
3. **SAC experiment (V7.1)** — swap PPO for Discrete SAC with replay buffer once architecture is validated
