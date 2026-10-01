#!/bin/bash
# Cleanly triggers graceful stop for V15.3 generator and trainer.
touch stop checkpoints/v15_3/stop
echo "🛑 Created 'stop' files. Generator and Trainer will finish current iteration, save checkpoints, and cleanly exit."
