
# Async Shared Configuration

import os

# Config
# Config imported from src.config
try:
    from src.config import ACTOR_BATCH_SIZE as BATCH_SIZE, NUM_ACTORS, TRAINING_BATCH_SIZE as SAMPLE_BATCH_SIZE, TRAIN_STEPS as TRAIN_STEPS_PER_LOOP
except ImportError:
    # Fallback if run directly or path issue
    from config import ACTOR_BATCH_SIZE as BATCH_SIZE, NUM_ACTORS, TRAINING_BATCH_SIZE as SAMPLE_BATCH_SIZE, TRAIN_STEPS as TRAIN_STEPS_PER_LOOP

QUEUE_MAX_SIZE = 50 # Max batches in queue before blocking actors
SYNC_INTERVAL = 30 # Seconds between actor weight syncs

# Paths (Dynamically loaded from config to match RUN_NAME)
try:
    from src.config import CHECKPOINT_DIR, MAIN_CKPT_PATH, ELOS_PATH, WC_STATS_PATH
except ImportError:
    from config import CHECKPOINT_DIR, MAIN_CKPT_PATH, ELOS_PATH, WC_STATS_PATH

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ensure directories (Done in config, but safe to keep)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
