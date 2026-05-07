"""Daemonize Mac pipeline — survives Claude Code closure / shell exit."""
import os
import sys

if os.fork() > 0:
    print("Mac pipeline daemonized")
    sys.exit(0)
os.setsid()
if os.fork() > 0:
    sys.exit(0)

os.chdir("/Users/sumit/Github/AlphaLudo/td_ludo")
devnull_r = os.open(os.devnull, os.O_RDONLY)
log_w = os.open("/tmp/mcts_v1_mac_pipeline.stdout", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(devnull_r, 0)
os.dup2(log_w, 1)
os.dup2(log_w, 2)
os.close(devnull_r)
os.close(log_w)

os.execv("/bin/bash", ["/bin/bash", "/tmp/mcts_v1_mac_pipeline.sh"])
