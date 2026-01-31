import json
import os

# Mode Selection: PROD, BACKGROUND, or TEST
# Set via environment variable: export ALPHALUDO_MODE=TEST
MODE = os.environ.get("ALPHALUDO_MODE", "PROD").upper()

print(f"[Config] Loading Configuration for Mode: {MODE}")

# Default Configs (Fallback)
DEFAULT_CONFIGS = {
    "PROD": {
        # === MCTS Settings (v3: Improved Exploration) ===
        "MCTS_SIMS": 400,              # Reduced from 800 (more games > deeper search)
        "MCTS_PARALLEL_SIMS": 8,       # Virtual Loss parallelism per game
        "C_PUCT": 3.0,                 # Increased for stochastic game exploration
        "DIRICHLET_ALPHA": 0.3,        # Root exploration noise
        "DIRICHLET_EPS": 0.25,         # Noise mixing weight
        
        # === Batch Settings (Hardware-optimized for M4) ===
        "ACTOR_BATCH_SIZE": 16,        # Reduced for less straggler waiting
        "INFERENCE_BATCH_SIZE": 128,   # Batch inference for MPS efficiency
        "TRAINING_BATCH_SIZE": 512,    # Reduced for more frequent updates
        
        # === Training Settings (v3: Improved Learning) ===
        "TRAIN_STEPS": 100,            # Gradient updates per learning iteration
        "NUM_ACTORS": 2,               # 2 actors for overnight training
        "GHOST_SAVE_FREQ": 100,        # Save ghost every N iterations
        "LEARNING_RATE": 0.001,        # Increased from 0.0001
        "LR_WARMUP_STEPS": 1000,       # Learning rate warmup
        "BUFFER_SIZE_LIMIT": 200000,   # Replay buffer max size
        
        # === TD(λ) Training (v3) ===
        "TD_LAMBDA": 0.95,             # TD(λ) discount
        "TD_GAMMA": 0.99,              # Value discount factor
        "AUX_LOSS_WEIGHT": 0.5,        # Auxiliary safety head weight
        
        # === Hardware Optimization ===
        "USE_FLOAT16": True,           # Half precision for 2x speedup on M4
    },
    
    "BACKGROUND": {
        "MCTS_SIMS": 400, "MCTS_PARALLEL_SIMS": 4, "C_PUCT": 3.0, 
        "DIRICHLET_ALPHA": 0.3, "DIRICHLET_EPS": 0.25,
        "ACTOR_BATCH_SIZE": 8, "INFERENCE_BATCH_SIZE": 32, "TRAINING_BATCH_SIZE": 256,
        "TRAIN_STEPS": 50, "NUM_ACTORS": 1, "GHOST_SAVE_FREQ": 200,
        "LEARNING_RATE": 0.001, "LR_WARMUP_STEPS": 1000, "BUFFER_SIZE_LIMIT": 200000,
        "TD_LAMBDA": 0.95, "TD_GAMMA": 0.99, "AUX_LOSS_WEIGHT": 0.5, "USE_FLOAT16": True,
    },
    
    "TEST": {
        "MCTS_SIMS": 10, "MCTS_PARALLEL_SIMS": 1, "C_PUCT": 3.0,
        "DIRICHLET_ALPHA": 0.3, "DIRICHLET_EPS": 0.25,
        "ACTOR_BATCH_SIZE": 16, "INFERENCE_BATCH_SIZE": 32, "TRAINING_BATCH_SIZE": 16,
        "TRAIN_STEPS": 5, "NUM_ACTORS": 1, "GHOST_SAVE_FREQ": 100,
        "LEARNING_RATE": 0.001, "LR_WARMUP_STEPS": 100, "BUFFER_SIZE_LIMIT": 1000,
        "TD_LAMBDA": 0.95, "TD_GAMMA": 0.99, "AUX_LOSS_WEIGHT": 0.5, "USE_FLOAT16": False,
    }
}

CONFIGS = DEFAULT_CONFIGS.copy()

def load_config_from_json():
    """Reloads configuration from config.json if it exists."""
    global CONFIGS
    try:
        # Check if project root config.json exists
        # Assuming config.py is in src/, json is in root
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(root_dir, "config.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_conf = json.load(f)
                
            # Update CONFIGS with JSON values (keys are upper-cased automatically in JSON usually? 
            # My created JSON has lowercase keys "mcts_sims". Need to handle mapping.)
            
            for mode_key in ["PROD", "BACKGROUND", "TEST"]:
                if mode_key in json_conf:
                    for k, v in json_conf[mode_key].items():
                        # Map lowercase json key to uppercase config key
                        CONFIGS[mode_key][k.upper()] = v
                        
            # print(f"[Config] Reloaded from JSON. C_PUCT={CONFIGS[MODE]['C_PUCT']}")
            return True
    except Exception as e:
        print(f"[Config] Failed to load config.json: {e}")
        return False

# Initial Load
load_config_from_json()

# Select Config
CONF = CONFIGS.get(MODE, CONFIGS["PROD"])

# Expose individual variables for direct import (NOTE: These are STATIC copies!)
# To get dynamic values, users should read CONFIGS[MODE]["KEY"]
MCTS_SIMS = CONF["MCTS_SIMS"]
MCTS_PARALLEL_SIMS = CONF.get("MCTS_PARALLEL_SIMS", 1)
C_PUCT = CONF.get("C_PUCT", 3.0)
DIRICHLET_ALPHA = CONF.get("DIRICHLET_ALPHA", 0.3)
DIRICHLET_EPS = CONF.get("DIRICHLET_EPS", 0.25)


ACTOR_BATCH_SIZE = CONF["ACTOR_BATCH_SIZE"]
INFERENCE_BATCH_SIZE = CONF.get("INFERENCE_BATCH_SIZE", 64)
TRAINING_BATCH_SIZE = CONF["TRAINING_BATCH_SIZE"]

TRAIN_STEPS = CONF["TRAIN_STEPS"]
NUM_ACTORS = CONF["NUM_ACTORS"]
GHOST_SAVE_FREQ = CONF["GHOST_SAVE_FREQ"]
LEARNING_RATE = CONF["LEARNING_RATE"]
LR_WARMUP_STEPS = CONF.get("LR_WARMUP_STEPS", 1000)
BUFFER_SIZE_LIMIT = CONF["BUFFER_SIZE_LIMIT"]

TD_LAMBDA = CONF.get("TD_LAMBDA", 0.95)
TD_GAMMA = CONF.get("TD_GAMMA", 0.99)
AUX_LOSS_WEIGHT = CONF.get("AUX_LOSS_WEIGHT", 0.5)

USE_FLOAT16 = CONF.get("USE_FLOAT16", False)

# Run Name (for forking experiments)
# Set via: export ALPHALUDO_RUN_NAME=mastery_v3
# BACKGROUND mode uses same data as PROD by default
if MODE == "BACKGROUND":
    RUN_NAME = os.environ.get("ALPHALUDO_RUN_NAME", "mastery_v3")  # Shares PROD data
elif MODE == "PROD":
    RUN_NAME = os.environ.get("ALPHALUDO_RUN_NAME", "mastery_v3")
else:
    RUN_NAME = os.environ.get("ALPHALUDO_RUN_NAME", "test_v3")

# Data Paths (using RUN_NAME)
CHECKPOINT_DIR = f"checkpoints_mastery/{RUN_NAME}"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MAIN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
GHOSTS_DIR = os.path.join(CHECKPOINT_DIR, "ghosts")
ELOS_PATH = os.path.join(CHECKPOINT_DIR, "elo_ratings.json")
WC_STATS_PATH = os.path.join(CHECKPOINT_DIR, "wc_stats.json")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "training_metrics.json")

print(f"[Config] Run Name: {RUN_NAME} | MCTS Sims: {MCTS_SIMS} | c_puct: {C_PUCT} | Float16: {USE_FLOAT16}")
