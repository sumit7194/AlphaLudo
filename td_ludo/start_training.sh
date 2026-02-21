#!/bin/bash
export TD_LUDO_RUN_NAME=td_v2_11ch
export TD_LUDO_MODE=PROD
td_env/bin/python train.py --resume
