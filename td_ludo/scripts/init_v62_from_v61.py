#!/usr/bin/env python3
"""Entry-point shim — delegates to the legacy implementation.

The actual logic still lives in td_ludo/init_v62_from_v61.py and will
move into a td_ludo.training.init_v62 module in a future cron tick.

Usage from td_ludo/:
    python3 scripts/init_v62_from_v61.py [v61_checkpoint_path]
"""
import os
import runpy
import sys

# Locate the legacy script (one directory up from scripts/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY = os.path.normpath(os.path.join(_HERE, "..", "init_v62_from_v61.py"))

if not os.path.exists(_LEGACY):
    print(f"ERROR: legacy implementation not found at {_LEGACY}", file=sys.stderr)
    sys.exit(1)

# Run the legacy file as if it were invoked directly
runpy.run_path(_LEGACY, run_name="__main__")
