#!/bin/zsh
cd /Users/sumit/Github/AlphaLudo
if pgrep -f confirm_routing > /dev/null; then echo "already running (pid $(pgrep -f confirm_routing|head -1))"; exit 0; fi
nohup env PYTHONPATH=td_ludo:td_ludo_v15 td_ludo/td_env/bin/python -c "
import subprocess,sys
runs=[('64','128','confirm'),('128','256','confirm_d128')]
for d,f,od in runs:
    subprocess.run([sys.executable,'-m','experiments.twosignal.confirm_routing',
      '--seeds','2,3,4,5','--coeffs','1.0','--d-model',d,'--ffn-dim',f,
      '--out-dir',od,'--epochs','6','--device','mps'])
" >> td_ludo/experiments/twosignal/power_run.log 2>&1 &
echo "resumed, pid $!"
