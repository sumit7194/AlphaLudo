# AlphaLudo

> **AlphaZero-style AI for the board game Ludo**

An implementation of AlphaZero techniques for learning to play Ludo, featuring:
- **Monte Carlo Tree Search (MCTS)** with neural network guidance
- **Deep Neural Networks** for policy and value estimation
- **Self-Play** for training data generation
- **Population-Based Training (PBT)** for hyperparameter optimization

---

## Features

| Feature | Description |
|---------|-------------|
| **C++ Game Engine** | High-performance Ludo implementation with Python bindings |
| **Batched MCTS** | Process multiple games simultaneously in C++ |
| **MPS/CUDA Support** | GPU acceleration for neural network inference |
| **Real-time Visualizer** | WebSocket-based game visualization in browser |
| **Elo Tracking** | Track model improvement over training |
| **Ghost System** | Train against previous versions of itself |
| **Multi-Heuristic Bots** | Aggressive, Defensive, and Racing bot variants |

---

## Installation

```bash
# Clone and enter directory
cd AlphaLudo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy websockets pybind11

# Build C++ extension
pip install -e . --no-build-isolation
```

---

## Training

### Normal Training (Recommended)

The main training script uses batched self-play with ghost opponents:

```bash
# Run with specific number of iterations
python train_mastery.py --run-name my_run --iterations 100

# Run continuously until Ctrl+C
python train_mastery.py --run-name my_run --continuous

# Resume training from checkpoint
python train_mastery.py --run-name my_run --iterations 50
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--run-name` | `mastery_v1` | Name for this training run (checkpoints saved in `checkpoints_mastery/{run-name}/`) |
| `--iterations` | `1000` | Number of training iterations (ignored if `--continuous`) |
| `--continuous` | `false` | Run indefinitely until interrupted with Ctrl+C |

**Training Details:**
- 16 games per iteration with C++ batched MCTS
- 200 MCTS simulations per move
- Temperature scheduling: τ=1.0 for first 30 moves, then τ=0.1
- Ghost snapshots saved every 10 iterations
- Checkpoints saved after each iteration

---

### Population-Based Training (PBT)

Trains multiple agents in parallel with hyperparameter evolution:

```bash
# Run PBT with default settings
python train_pbt.py --population-size 4 --generations 100

# Run PBT continuously until Ctrl+C
python train_pbt.py --population-size 4 --continuous

# Custom configuration
python train_pbt.py --population-size 8 --generations 200 --iterations-per-gen 30 --checkpoint-dir checkpoints_pbt_custom
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--population-size` | `4` | Number of agents in the population |
| `--generations` | `100` | Number of PBT generations (ignored if `--continuous`) |
| `--iterations-per-gen` | `20` | Training iterations per generation |
| `--checkpoint-dir` | `checkpoints_pbt` | Directory for PBT checkpoints |
| `--continuous` | `false` | Run indefinitely until interrupted with Ctrl+C |

**PBT Details:**
- Agents play real games against each other for evaluation
- Evolution every 5 generations (worst replaced by mutated best)
- Hyperparameters mutated: learning rate, temperature, MCTS simulations, c_puct

---

### Running Continuously

Use the `--continuous` flag to run training indefinitely until you press `Ctrl+C`:

```bash
# Normal training - runs forever until stopped
python train_mastery.py --run-name my_run --continuous

# PBT training - runs forever until stopped
python train_pbt.py --continuous
```

Both scripts automatically save checkpoints after each iteration/generation, so you can safely stop and resume anytime. When interrupted:
- All progress is saved to the checkpoint
- Training can be resumed by running the same command again


---

## Visualization

1. Start training with visualization enabled (default)
2. Open `visualizer.html` in a web browser
3. Connect to `ws://localhost:8765`

The visualizer shows:
- Real-time game boards for all concurrent games
- Training statistics (iteration, loss, buffer size)
- Elo ratings and history
- Ghost and heuristic game indicators

---

## Project Structure

```
AlphaLudo/
├── train_mastery.py         # Main training script (recommended)
├── train_pbt.py             # Population-Based Training script
├── visualizer.html          # Browser-based game visualizer
├── index.html               # Training dashboard
├── setup.py                 # C++ extension build configuration
│
├── src/                     # Source code
│   ├── game.cpp/h           # C++ Ludo game engine
│   ├── mcts.cpp/h           # C++ batched MCTS engine
│   ├── bindings.cpp         # pybind11 Python bindings
│   ├── model_mastery.py     # Neural network (12-channel, 128 filters)
│   ├── vector_league.py     # Batched game worker
│   ├── tensor_utils_mastery.py  # State → tensor conversion
│   ├── training_utils.py    # Temperature, augmentation, Elo
│   ├── pbt_manager.py       # PBT population manager
│   └── heuristic_bot.py     # Heuristic bot variants
│
├── tests/                   # Unit tests
├── checkpoints_mastery/     # Training checkpoints
└── checkpoints_pbt/         # PBT checkpoints
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_augmentation.py -v
```

---

## Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## License

MIT License

---

## Recent Updates (Jan 2026)

### Critical Logic Fixes
*   **Rotation Alignment**: Fixed a severe mismatch where C++ MCTS inference used Counter-Clockwise rotation while Python training used Clockwise. Both now use **Clockwise** rotation, ensuring the network sees the board correctly.
*   **MCTS Adversarial Logic**: Corrected `ucb_score` to **negate values** for opponent nodes, fixing a "cooperative play" bug where the model mistakenly maximized the opponent's winning chances.
*   **Ground Truth Terminals**: Search now uses exact game rules (1.0/-1.0) for terminal states instead of noisy network predictions.

### Input Representation Enhancements
*   **Score Advantage**: Repurposed unused Channel 11 to encode global game context: `(MyScore - MaxOpponentScore) / 4.0`. This gives the model a direct sense of "winning" vs "losing".
*   **Density Inputs**: Upgraded spatial channels from Binary (0/1) to **Density Counts** (Accumulation). The model can now "see" stacks (blockades) and count exact scores at home visually.

### Operational Improvements
*   **Ghost Management**: Implemented auto-cleanup to keep only the **Top 10 Elo** ghost models, saving significant disk space.
*   **Data Hygiene**: Purged the Replay Buffer to remove corrupted data from the pre-fix era.
*   **Persistence**: Switched to SQLite (`training_history.db`) for robust winrate tracking and fixed dashboard persistence issues.

