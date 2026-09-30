#!/bin/zsh
cd /Users/sumit/Github/AlphaLudo
echo "processes:"
pgrep -fl "train_twosignal_rl|make_dashboard" | sed 's/^/  /' || echo "  none running"
echo
td_ludo/td_env/bin/python -c "
import json
from pathlib import Path
T=330_000
for tag in ('routed_long','mixed_long'):
    f=Path('td_ludo/experiments/twosignal/rl_long')/tag/'metrics.json'
    if not f.exists(): print(f'  {tag:12s} not started'); continue
    h=json.load(open(f))
    g=h[-1].get('total_games',0) if h else 0
    ev=[(r.get('total_games',0),r.get('win_rate_vs_bots',r.get('win_rate_vs_random')),'B' if 'win_rate_vs_bots' in r else 'r') for r in h if ('win_rate_vs_bots' in r or 'win_rate_vs_random' in r)]
    last=' '.join(f'{a//1000}k:{b:.3f}{t}' for a,b,t in ev[-5:])
    print(f'  {tag:12s} {g:>7,}/{T:,} ({100*g/T:4.1f}%)  gate {h[-1].get(\"gate_abs_mean\",0):.4f}')
    print(f'  {\"\":12s} recent evals: {last}')
"
