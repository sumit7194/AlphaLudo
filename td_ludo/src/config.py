"""
TD-Ludo Configuration

Modes:
- PROD: Full training with kickstart weights, saves to checkpoints/td_prod/
- TEST: Fast iteration for debugging, saves to checkpoints/td_test/

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
KICKSTART_PATH = os.path.join(PRETRAINED_DIR, "model_kickstart.pt")

# =============================================================================
# Default Configurations — PROD vs TEST fully isolated
# =============================================================================
DEFAULT_CONFIGS = {
    "PROD": {
        # === TD Learning ===
        "TD_GAMMA": 0.995,           # Discount factor (high for long Ludo games)
        "EPSILON_START": 0.10,        # Initial exploration rate
        "EPSILON_END": 0.02,          # Final exploration rate
        "EPSILON_DECAY_GAMES": 50000, # Games over which ε decays linearly

        # === Neural Network Training ===
        "LEARNING_RATE": 0.0005,      # Adam optimizer LR
        "WEIGHT_DECAY": 1e-4,         # L2 regularization
        "GRAD_ACCUM_STEPS": 4,        # Accumulate gradients over N moves before stepping
        "MAX_GRAD_NORM": 1.0,         # Gradient clipping

        # === Reward Shaping ===
        "REWARD_SHAPING": True,       # Enable PBRS

        # === Experience Buffer (optional replay) ===
        "USE_EXPERIENCE_BUFFER": True, # Store recent experiences for replay
        "BUFFER_SIZE": 200000,          # Max transitions in buffer
        "REPLAY_BATCH_SIZE": 512,       # Batch size for replay training
        "REPLAY_EVERY_N_GAMES": 10,    # Do a replay pass every N games
        "REPLAY_STEPS": 32,            # Number of mini-batches per replay pass

        # === Game Settings ===
        "BATCH_SIZE": 2048,
        "MAX_MOVES_PER_GAME": 1000,
        "GAME_COMPOSITION": {
            "SelfPlay": 0.50,
            "Heuristic": 0.15,
            "Aggressive": 0.10,
            "Defensive": 0.10,
            "Racing": 0.10,
            "Random": 0.05,
        },
        "NUM_ACTIVE_PLAYERS": 4,       # Standard 4-player
        "TERMINAL_LOSS_REWARD": -0.33, # -1/3 for losing against 3 opponents

        # === Evaluation ===
        "EVAL_INTERVAL": 500,         # Games between evaluations
        "EVAL_GAMES": 200,            # Games per evaluation round

        # === Checkpointing ===
        "SAVE_INTERVAL": 300,         # Seconds between checkpoint saves
        "GHOST_SAVE_INTERVAL": 2000,  # Save ghost snapshot every N games
        "MAX_GHOSTS": 20,             # Keep at most N ghost snapshots

        # === Hardware ===
        "USE_FLOAT16": False,

        # === Paths (fully isolated per mode) ===
        "CHECKPOINT_BASE": "checkpoints",
        "RUN_NAME": "td_prod",
    },

    "TEST": {
        "TD_GAMMA": 0.995,
        "EPSILON_START": 0.10,
        "EPSILON_END": 0.02,
        "EPSILON_DECAY_GAMES": 100,
        "LEARNING_RATE": 0.001,
        "WEIGHT_DECAY": 1e-4,
        "GRAD_ACCUM_STEPS": 1,
        "MAX_GRAD_NORM": 1.0,
        "REWARD_SHAPING": True,
        "USE_EXPERIENCE_BUFFER": False,
        "BUFFER_SIZE": 1000,
        "REPLAY_BATCH_SIZE": 16,
        "REPLAY_EVERY_N_GAMES": 5,
        "REPLAY_STEPS": 4,
        "BATCH_SIZE": 4,
        "MAX_MOVES_PER_GAME": 500,
        "GAME_COMPOSITION": {
            "SelfPlay": 0.50,
            "Heuristic": 0.50,
        },
        "NUM_ACTIVE_PLAYERS": 2,       # 2-player for testing
        "TERMINAL_LOSS_REWARD": -1.0,  # Zero-sum for 1v1
        "EVAL_INTERVAL": 20,
        "EVAL_GAMES": 10,
        "SAVE_INTERVAL": 60,
        "GHOST_SAVE_INTERVAL": 50,
        "MAX_GHOSTS": 5,
        "USE_FLOAT16": False,
        "CHECKPOINT_BASE": "checkpoints_test",
        "RUN_NAME": "td_test",
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
TD_GAMMA = CONF["TD_GAMMA"]
EPSILON_START = CONF["EPSILON_START"]
EPSILON_END = CONF["EPSILON_END"]
EPSILON_DECAY_GAMES = CONF["EPSILON_DECAY_GAMES"]
LEARNING_RATE = CONF["LEARNING_RATE"]
WEIGHT_DECAY = CONF["WEIGHT_DECAY"]
GRAD_ACCUM_STEPS = CONF["GRAD_ACCUM_STEPS"]
MAX_GRAD_NORM = CONF["MAX_GRAD_NORM"]
REWARD_SHAPING = CONF["REWARD_SHAPING"]
USE_EXPERIENCE_BUFFER = CONF["USE_EXPERIENCE_BUFFER"]
BUFFER_SIZE = CONF["BUFFER_SIZE"]
REPLAY_BATCH_SIZE = CONF["REPLAY_BATCH_SIZE"]
REPLAY_EVERY_N_GAMES = CONF["REPLAY_EVERY_N_GAMES"]
REPLAY_STEPS = CONF["REPLAY_STEPS"]
BATCH_SIZE = CONF["BATCH_SIZE"]
MAX_MOVES_PER_GAME = CONF["MAX_MOVES_PER_GAME"]
GAME_COMPOSITION = CONF["GAME_COMPOSITION"]
EVAL_INTERVAL = CONF["EVAL_INTERVAL"]
EVAL_GAMES = CONF["EVAL_GAMES"]
SAVE_INTERVAL = CONF["SAVE_INTERVAL"]
GHOST_SAVE_INTERVAL = CONF["GHOST_SAVE_INTERVAL"]
MAX_GHOSTS = CONF["MAX_GHOSTS"]
USE_FLOAT16 = CONF["USE_FLOAT16"]
USE_FLOAT16 = CONF["USE_FLOAT16"]
NUM_ACTIVE_PLAYERS = CONF.get("NUM_ACTIVE_PLAYERS", 4)
TERMINAL_LOSS_REWARD = CONF.get("TERMINAL_LOSS_REWARD", -0.33)
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
BUFFER_PATH = os.path.join(CHECKPOINT_DIR, "experience_buffer.pt")
STATS_PATH = os.path.join(CHECKPOINT_DIR, "live_stats.json")
ELO_PATH = os.path.join(CHECKPOINT_DIR, "elo_ratings.json")
GAME_DB_PATH = os.path.join(CHECKPOINT_DIR, "game_history.db")

print(f"[TD-Ludo Config] Run: {RUN_NAME} | Dir: {CHECKPOINT_DIR}")
print(f"[TD-Ludo Config] LR: {LEARNING_RATE} | γ: {TD_GAMMA} | ε: {EPSILON_START}→{EPSILON_END}")
print(f"[TD-Ludo Config] Buffer: {'ON' if USE_EXPERIENCE_BUFFER else 'OFF'} | Ghosts: every {GHOST_SAVE_INTERVAL} games")
