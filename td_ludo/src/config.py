"""
TD-Ludo Configuration — Actor-Critic Edition

Modes:
- PROD: Full training, saves to checkpoints/ac_v5/
- TEST: Fast iteration for debugging, saves to checkpoints_test/ac_test/

Set via: export TD_LUDO_MODE=TEST
"""

import os
import json

MODE = os.environ.get("TD_LUDO_MODE", "PROD").upper()

print(f"[TD-Ludo Config] Mode: {MODE}")

# =============================================================================
# Project Paths (always relative to td_ludo/ root)
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained")
KICKSTART_PATH = os.path.join(PRETRAINED_DIR, "model_kickstart_11ch.pt")

# =============================================================================
# Default Configurations — PROD vs TEST fully isolated
# =============================================================================
DEFAULT_CONFIGS = {
    "PROD": {
        # === Actor-Critic Training ===
        "LEARNING_RATE": 0.00001,        # Adam LR (lowered from 5e-5 to prevent long-term drift)
        "WEIGHT_DECAY": 1e-4,            # L2 regularization
        "MAX_GRAD_NORM": 1.0,            # Gradient clipping
        "ENTROPY_COEFF": 0.005,          # Entropy bonus coefficient (lowered to respect SL bounds)
        "VALUE_LOSS_COEFF": 0.5,         # Value head loss weight
        "GRAD_ACCUM_GAMES": 8,           # (Legacy, unused by PPO trainer)

        # === PPO (Proximal Policy Optimization) ===
        "CLIP_EPSILON": 0.2,             # PPO clipping range (limits policy change per update)
        "PPO_EPOCHS": 3,                 # Number of passes over collected data per PPO update
        "PPO_BUFFER_GAMES": 64,          # Collect this many games before running PPO update
        "PPO_MINIBATCH_SIZE": 256,       # Steps per mini-batch during PPO training

        # === Exploration (Temperature-based) ===
        "TEMPERATURE_START": 1.1,        # Policy sampling temperature (lowered to avoid off-policy blunders)
        "TEMPERATURE_END": 0.95,         # Final temperature (slightly deterministic to exploit known win-paths)
        "TEMPERATURE_DECAY_GAMES": 20000, # Decay faster to rely on the stabilized Value head
        "SELFPLAY_GHOST_FRACTION": 0.50, # In self-play, use a past ghost this often when snapshots exist
        "SELFPLAY_GHOST_STRATEGY": "matched", # Ghost selection strategy: matched/adversarial/random

        # === Game Settings ===
        "BATCH_SIZE": 512,               # Run N games in parallel in C++
        "MAX_MOVES_PER_GAME": 10000,
        "GAME_COMPOSITION": {
            "SelfPlay": 0.40,            # Deep self-correcting equilibrium
            "Expert": 0.25,              # The newly optimized elite bot
            "Heuristic": 0.15,           # General logic grounding
            "Aggressive": 0.10,          # Forced to learn survival/evasion
            "Defensive": 0.10,           # Forced to learn active breaching
        },
        "NUM_ACTIVE_PLAYERS": 2,         # 2-Player Mode (P0 vs P2)

        # === Evaluation ===
        # 2026-04-24: bumped from 2000/500 to 10000/2000 (SE 1.9pp → 1.0pp).
        # 2026-04-29 (V12.2 era): bumped to 100000/5000.
        #   - V12.2 plateau-broke at G=40K and stayed in 80–82% band; 10K
        #     intervals were too noisy at this WR level (1pp SE on 2K games)
        #     to distinguish real improvement from variance.
        #   - 5K games/eval drops SE to ~0.6pp; 100K interval gives the model
        #     ~3.5h of training between evals so each eval reflects a real
        #     policy shift, not minute-to-minute noise.
        "EVAL_INTERVAL": 100000,        # Games between evaluations
        "EVAL_GAMES": 5000,             # Games per evaluation round
        "EARLY_STOP_PATIENCE": 100,     # Stop training if eval WR drops for N consecutive evals

        # === Checkpointing ===
        "SAVE_INTERVAL": 300,           # Seconds between checkpoint saves
        "GHOST_SAVE_INTERVAL": 2000,    # Save ghost snapshot every N games
        "MAX_GHOSTS": 20,               # Keep at most N ghost snapshots

        # === Hardware ===
        "USE_FLOAT16": False,

        # === Paths (fully isolated per mode) ===
        "CHECKPOINT_BASE": "checkpoints",
        "RUN_NAME": "ac_v6_big",
    },

    "TEST": {
        "LEARNING_RATE": 0.0003,
        "WEIGHT_DECAY": 1e-4,
        "MAX_GRAD_NORM": 1.0,
        "ENTROPY_COEFF": 0.01,
        "VALUE_LOSS_COEFF": 0.5,
        "GRAD_ACCUM_GAMES": 4,
        "CLIP_EPSILON": 0.2,
        "PPO_EPOCHS": 2,
        "PPO_BUFFER_GAMES": 8,
        "PPO_MINIBATCH_SIZE": 64,
        "TEMPERATURE_START": 1.5,
        "TEMPERATURE_END": 1.0,
        "TEMPERATURE_DECAY_GAMES": 100,
        "SELFPLAY_GHOST_FRACTION": 0.50,
        "SELFPLAY_GHOST_STRATEGY": "matched",
        "BATCH_SIZE": 4,
        "MAX_MOVES_PER_GAME": 500,
        "GAME_COMPOSITION": {
            "SelfPlay": 0.50,
            "Heuristic": 0.50,
        },
        "NUM_ACTIVE_PLAYERS": 2,
        "EVAL_INTERVAL": 20,
        "EVAL_GAMES": 10,
        "EARLY_STOP_PATIENCE": 3,
        "SAVE_INTERVAL": 60,
        "GHOST_SAVE_INTERVAL": 50,
        "MAX_GHOSTS": 5,
        "USE_FLOAT16": False,
        "CHECKPOINT_BASE": "checkpoints_test",
        "RUN_NAME": "ac_test",
    }
}

# =============================================================================
# Config Loading
# =============================================================================
CONFIGS = {k: dict(v) for k, v in DEFAULT_CONFIGS.items()}

def load_config_from_json():
    """Load overrides from td_config.json if it exists."""
    global CONFIGS
    try:
        json_path = os.path.join(PROJECT_ROOT, "td_config.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                overrides = json.load(f)
            for mode_key in CONFIGS:
                if mode_key in overrides:
                    for k, v in overrides[mode_key].items():
                        CONFIGS[mode_key][k.upper()] = v
            return True
    except Exception as e:
        print(f"[Config] Failed to load td_config.json: {e}")
    return False

load_config_from_json()

# Select active config
CONF = CONFIGS.get(MODE, CONFIGS["PROD"])

# =============================================================================
# Expose individual variables for direct import
# =============================================================================
LEARNING_RATE = CONF["LEARNING_RATE"]
WEIGHT_DECAY = CONF["WEIGHT_DECAY"]
MAX_GRAD_NORM = CONF["MAX_GRAD_NORM"]
ENTROPY_COEFF = CONF["ENTROPY_COEFF"]
VALUE_LOSS_COEFF = CONF["VALUE_LOSS_COEFF"]
GRAD_ACCUM_GAMES = CONF["GRAD_ACCUM_GAMES"]
CLIP_EPSILON = CONF["CLIP_EPSILON"]
PPO_EPOCHS = CONF["PPO_EPOCHS"]
PPO_BUFFER_GAMES = CONF["PPO_BUFFER_GAMES"]
PPO_MINIBATCH_SIZE = CONF["PPO_MINIBATCH_SIZE"]
TEMPERATURE_START = CONF["TEMPERATURE_START"]
TEMPERATURE_END = CONF["TEMPERATURE_END"]
TEMPERATURE_DECAY_GAMES = CONF["TEMPERATURE_DECAY_GAMES"]
SELFPLAY_GHOST_FRACTION = CONF["SELFPLAY_GHOST_FRACTION"]
SELFPLAY_GHOST_STRATEGY = CONF["SELFPLAY_GHOST_STRATEGY"]
BATCH_SIZE = CONF["BATCH_SIZE"]
MAX_MOVES_PER_GAME = CONF["MAX_MOVES_PER_GAME"]
GAME_COMPOSITION = CONF["GAME_COMPOSITION"]
NUM_ACTIVE_PLAYERS = CONF.get("NUM_ACTIVE_PLAYERS", 2)
EVAL_INTERVAL = CONF["EVAL_INTERVAL"]
EVAL_GAMES = CONF["EVAL_GAMES"]
EARLY_STOP_PATIENCE = CONF["EARLY_STOP_PATIENCE"]
SAVE_INTERVAL = CONF["SAVE_INTERVAL"]
GHOST_SAVE_INTERVAL = CONF["GHOST_SAVE_INTERVAL"]
MAX_GHOSTS = CONF["MAX_GHOSTS"]
USE_FLOAT16 = CONF["USE_FLOAT16"]

# =============================================================================
# Paths — Fully Isolated per Mode
# =============================================================================
RUN_NAME = os.environ.get("TD_LUDO_RUN_NAME", CONF["RUN_NAME"])
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, CONF["CHECKPOINT_BASE"], RUN_NAME)
GHOSTS_DIR = os.path.join(CHECKPOINT_DIR, "ghosts")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GHOSTS_DIR, exist_ok=True)

MAIN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "model_best.pt")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "training_metrics.json")
STATS_PATH = os.path.join(CHECKPOINT_DIR, "live_stats.json")
ELO_PATH = os.path.join(CHECKPOINT_DIR, "elo_ratings.json")
GAME_DB_PATH = os.path.join(CHECKPOINT_DIR, "game_history.db")

print(f"[TD-Ludo Config] Run: {RUN_NAME} | Dir: {CHECKPOINT_DIR}")
print(f"[TD-Ludo Config] LR: {LEARNING_RATE} | Entropy: {ENTROPY_COEFF} | Temp: {TEMPERATURE_START}→{TEMPERATURE_END}")
print(f"[TD-Ludo Config] Ghosts: every {GHOST_SAVE_INTERVAL} games | Batch: {BATCH_SIZE}")
