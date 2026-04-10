#!/usr/bin/env python3
"""Entry-point shim — delegates to the legacy implementation.

The actual logic still lives in td_ludo/train_v6_2_fast.py and will move
into the td_ludo package in a future cron tick (post-Stage-B9).

Usage from td_ludo/:
    python3 scripts/train_v6_2_fast.py [args]
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY = os.path.normpath(os.path.join(_HERE, "..", "train_v6_2_fast.py"))

if not os.path.exists(_LEGACY):
    print(f"ERROR: legacy implementation not found at {_LEGACY}", file=sys.stderr)
    sys.exit(1)

runpy.run_path(_LEGACY, run_name="__main__")
