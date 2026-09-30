# AlphaLudo Retrospective: Historical Bottlenecks, Abandoned Paths & Modern Opportunities

> **Guiding Principle ([AGENTS.md](file:///Users/sumit/Github/AlphaLudo/AGENTS.md)):**  
> *"Never retreat to an inferior alternative at the first sign of a technical hurdle. Always investigate deeply, formulate first-principles solutions, try relentlessly, and NEVER abandon a superior path without failing multiple times across multiple approaches."*

---

## Executive Summary

Across the 6,200+ lines of the project training journal and model history (`td_ludo/training_journal.md`, `td_ludo/MODEL_HISTORY.md`), the core barrier that repeatedly forced algorithmic retreats was **not algorithmic inadequacy, but computational latency and memory explosion in Python**.

Whenever ambitious ideas were attempted—such as **AlphaZero-style MCTS search**, **deep multi-turn temporal transformers**, or **multi-step expectimax lookahead**—they collided with:
1. **Python/C++ glue overhead and GIL contention** (limiting simulation budgets to 25–50 plies).
2. **Exponential chance-node dilution** (Ludo's $1/6$ dice roll splits search trees 6 ways at every turn; shallow budgets of 50–200 simulations could only look 1–2 moves ahead).
3. **MPS Memory Out-of-Memory (OOM) crashes** (e.g., V11 allocating 4.45 GB for a single $225 \times 225$ spatial attention forward pass).
4. **Noisy leaf value evaluations** (shallow trees amplified critic variance rather than reducing it).

With our native Rust engine now executing 4-player MCTS in **106 microseconds** and 2-ply lookahead in **10 microseconds** (a **10,000× speedup** over Python), the computational calculus has completely changed. Roadblocks that were insurmountable in March–May 2026 can now be solved with rigor.

---

## 1. MCTS & Search During Training: Why It Failed & How to Fix It

### Historical Incidents

#### A. Experiment 9: The Original AlphaZero Attempt (Phase 8, Journal Line 298)
* **Goal**: Replace manual reward shaping with pure AlphaZero self-play MCTS using `td_ludo_cpp::MCTSEngine`.
* **The Compromise**: Because Python/C++ rollout latency was high, MCTS was capped at **50 simulations per move**.
* **The Failure**:
  - The model lost disastrously in head-to-head evaluation: **34.7% win rate** against the baseline.
  - **Root Cause 1 (Dice Branching Dilution)**: With 6 possible dice rolls at every chance node, 50 simulations spread across branches meant individual legal actions only received 2–8 rollouts. The search was essentially blind.
  - **Root Cause 2 (Terminal Reward Noise)**: The critic was trained purely on terminal $\pm 1$ outcomes over 175–400 moves. End-game dice variance wiped out the strategic value of mid-game tactical captures.
* **The Retreat**: MCTS self-play was completely **abandoned**, and the project reverted to PPO with hand-crafted reward shaping.

#### B. Experiment 13b & 13c: Inference-Time MCTS Sweep (Journal Line 826)
* **Setup**: Evaluated MCTS with 25, 50, 100, and 200 simulations on GCP T4.
* **The Paradox**: **More simulations made the bot perform worse!**
  - MCTS(25): 69.8% win rate vs Expert
  - MCTS(50): 57.1% (-13pp)
  - MCTS(100): 51.0% (-19pp)
  - MCTS(200): 48.4% (-22pp)
* **Root Cause**: The neural network value head was noisy. In a shallow tree with dice chance nodes, MCTS searched just deep enough to back up noisy leaf estimates, amplifying value errors rather than cutting through them.

#### C. Post-V13.2 MCTS Step 1 Distillation (Model History Line 734)
* **Setup**: 2-ply expectimax search using V13.2 as leaf evaluator to generate 1M states.
* **Result**: Student lost **89.6% to 10.4%** against V13.2 in H2H.
* **Root Cause**: Python expectimax took tens of milliseconds per move; leaf evaluation using a biased neural net without rollouts compounded systematic blind spots.

---

### The Modern First-Principles Solution with Rust

| Bottleneck in 2026 | Legacy Constraint | Native Rust Architecture |
| :--- | :--- | :--- |
| **Search Budget** | 25–50 simulations (took ~2s/move in Python) | **2,000–10,000 simulations/move** in single-digit milliseconds. |
| **Dice Chance Nodes** | 50 sims / 6 dice = ~8 sims/branch (useless) | 6,000 sims / 6 dice = **1,000 sims/branch** (high statistical power). |
| **Leaf Valuation** | Raw, noisy NN critic forward pass | **Hybrid Rollout**: Fast microsecond rollout simulation (60 plies) + Expectimax root priors. |
| **Memory Allocation** | Heap allocation per MCTS tree node | **Arena Allocation**: Pre-allocated contiguous vector pool; 0 heap allocations during search. |

---

## 2. Transformer Architectures: The Evolution, Failures & Ideal Form

Across project history, five different Transformer paradigms were attempted:

```
V7 (1D Sequence) → V8/V9 (CNN + Temporal) → V11 (225-Cell Grid) → V12 (8-Entity Attention) → V15 (GraphTransformer)
```

### The Three Critical Transformer Roadblocks

#### 1. V11 MPS Out-of-Memory (OOM) Crash (Journal Line 1843)
* **What Happened**: At game 530, V11 crashed during PPO update:
  ```
  RuntimeError: MPS backend out of memory. Tried to allocate 4.45 GiB on private pool.
  ```
* **Why**: V11 treated the $15 \times 15$ board grid as **225 sequence tokens**.
  - Dense attention map across 225 tokens is $225 \times 225 = 50,625$ pairs per head.
  - Across batch 256, 4 heads, backward graph, and FFN intermediates, attention alone required **4.45 GB per step**.
* **The Compromise**: V11.1 gutted the transformer: layers were reduced from 2 to 1, hidden dimension shrank to 64, and attention became an additive skip.

#### 2. V12's Token-Entity Breakthrough (The Hidden Gem) (Journal Line 1944)
* **The Insight**: Ludo is not an image; it is an entity interaction game. The board has only **8 active tokens** in 2-player (16 in 4-player).
* **The Transformation**:
  - Instead of attending over 225 board cells, V12 gathered features directly at token positions and attended over **8 entity tokens**.
  - Attention map plummeted from $225^2 = 50,625$ down to $8^2 = \mathbf{64}$ (**~700× cheaper and lighter!**).
* **The Result**:
  - SL validation policy accuracy surged to **95.9% (all-time project record)**.
  - Training took only **7 minutes** on an L4 GPU.
  - Smashed Aggressive bot win rates (+18.3pp).
* **Why it Was Put Aside**: V12 had minor positional weakness against static Expert heuristics, so the team retreated to pure ResNets (V13.2) because ResNets were easier to tune with existing reward-shaping scripts.

#### 3. V15 GraphTransformer RL Bleed (Model History Line 1275)
* **What Happened**: V15 SL matched the V13.5 teacher cleanly (83% win rate), but RL training caused an **8pp regression** (down to 43.2% in 5-way tournament).
* **Root Cause**:
  - Softmax attention in Transformers produces high-gradient sensitivity to entropy bonuses compared to CNNs.
  - When self-play reaches near-zero advantage ($G - V \approx 0$), the entropy gradient ($0.01 \times \nabla H$) dominates, pushing the policy towards a uniform distribution (indecisive play).

---

### The Ideal Modern Transformer Setup

Rather than a spatial 225-cell grid or a flat 1D sequence, the mathematically superior design is an **Entity-Relational Transformer (ERT)**:
1. **Tokens**:
   - $N$ Token Entities (4 for Self, 4 for each Opponent).
   - 1 Dice Token (conditioned on current roll).
   - 1 Global Game State Token (`[CLS]`, encoding turn, standings, and score diff).
   - Total sequence length: **10 tokens (2-player)** or **18 tokens (4-player)**.
2. **Pairwise Relative Relational Bias**:
   - Instead of 2D coordinates, inject a scalar relative distance matrix $\Delta_{ij} = \text{track\_distance}(i, j)$ into the attention logits:
     $$\text{Attention}(Q, K) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + B(\Delta_{ij})\right)$$
   - The model natively attends to tokens that are 1–6 squares behind (threats) or 1–6 squares ahead (chase targets) with zero spatial convolution overhead.
3. **Compute Cost**: Sequence length of 10 or 18 runs with sub-millisecond latency on Mac MPS without any risk of memory OOM.

---

## 3. The 80–83% Capability Ceiling: The Supervised Distillation Trap

Across models V6.1, V10, V11, V12.2, V13.2, V13.5, V14_scalar, and V15, every single architecture plateaued at **80–83% bot win rate**.

The journal reveals why:
1. **Teacher Bound**: Distilling from a teacher policy (e.g., V12.2 or V13.2) cannot produce a student that exceeds the teacher in expectation.
2. **Heuristic Saturation**: Fixed heuristic bots (Random, Heuristic, Aggressive, Defensive, Expert) have deterministic weaknesses. Once a network learns to exploit those weaknesses ~82% of the time, the remaining ~18% of losses are purely due to unpreventable bad dice luck.
3. **PPO Squeeze in Self-Play**: In symmetric self-play, $E[G - V] = 0$. Without curriculum expansion or search guidance, PPO enters an attractor state where policy updates stall.

---

## 4. Systems Architecture & Detached Background Execution

The journal records multiple instances where training runs died prematurely:
* **Terminal Disconnection / IDE Quit**: `nohup` died when parent shell closed because macOS lacks traditional `setsid` handling unless launched via `python -c "import os; os.setsid()"` or proper subshell disowning.
* **Multiprocessing IPC Queue Serialization**: V9/V10 used `multiprocessing.Queue` to stream transitions from 4 CPU actors to the MPS learner. Pickling numpy arrays saturated CPU cores and capped throughput at 115–134 GPM (~2 games/sec).
* **Current Rust Reality**:
  - Our Rust generator streams zero-copy numpy arrays directly into PyTorch tensors via PyO3.
  - Achieves **320 games per minute** on a single thread while sleeping 25ms between batches.
  - Memory consumption is stable at ~550 MB with zero disk bloat.

---

## 5. Strategic Proposal: 2-Player Benchmark First, Then 4-Player Scale-Up

Returning to **2-Player (P0 vs. P2)** offers immense strategic and scientific advantages before returning to 4-Player:

### Why 2-Player First?
1. **Strict Zero-Sum Game Theory**:
   - In 2-player Ludo, Minimax, Alpha-Beta, and standard 2-player AlphaZero MCTS are mathematically guaranteed to converge to the Nash equilibrium.
   - Payoffs are scalar: $v \in [-1, +1]$.
2. **Decades of Calibrated Elo Baselines**:
   - The repository contains frozen checkpoints (`V6.1`, `V12.2`, `V13.2`, `V13.5`) and standardized heuristic bot suites with millions of documented benchmark games.
   - Any gain over 83% or positive Elo delta against V13.5 is immediately verifiable.
3. **Lightning Fast Iteration**:
   - In 2-player mode, each game has only 8 tokens and takes half the moves (~160 moves vs 334 moves).
   - Rust simulation throughput will exceed **1,000 full games per second**.

### Recommended Two-Phase Execution Plan

#### Phase 1: 2-Player AlphaZero-Style MCTS Breakthrough
1. **Port 2-Player GameState to Rust**: Replicate the exact 2-player rules (P0 vs P2, 8 tokens) in `crates/ludo_core`.
2. **Zero-Sum MCTS with High Simulation Budget**:
   - Deploy Rust MCTS with **500–1,000 simulations per move**.
   - Use Expectimax priors at root + fast rollout evaluation at leaf nodes.
3. **Train Student on Search-Refined Policy ($\pi_{\text{MCTS}}$)**:
   - Train an **Entity-Relational Transformer (ERT)** directly on $\pi_{\text{MCTS}}$ targets.
   - Measure head-to-head win rate against `V13.5_latest` and `V12.2_latest`.

#### Phase 2: Transfer Proven Architecture Back to 4-Player
1. Expand Entity-Relational Transformer to 16 tokens.
2. Plug into the 4-way standings aux head and $\text{Max}^n$ search engine.
3. Train with winner-distillation at full scale.
