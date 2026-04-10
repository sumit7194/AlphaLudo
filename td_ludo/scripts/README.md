# td_ludo/scripts/

Canonical entry-point scripts for the td_ludo project.

This directory is the new home for invocation scripts. Each script here
is currently a thin runpy wrapper around the legacy implementation at
`td_ludo/<name>.py`. In a future cron tick (post-Stage-B9) the legacy
files will be retired and these wrappers will become full
`from td_ludo.X import main; main()` shims.

During the refactor (Stage B8/B9 of `discussion/REFACTOR_PLAN.md`) the
originals at `td_ludo/<name>.py` keep working untouched, so production
callers (GCP sweep, training jobs, deploy scripts) are not disrupted.

## Current contents

| Script                       | Status        | Backing impl                                 |
|------------------------------|---------------|----------------------------------------------|
| init_v62_from_v61.py         | runpy wrapper | td_ludo/init_v62_from_v61.py (legacy)        |
| train_v6_1.py                | runpy wrapper | td_ludo/train_v6_1.py (legacy)               |
| train_v6_1_fast.py           | runpy wrapper | td_ludo/train_v6_1_fast.py (legacy)          |
| train_v6_2_fast.py           | runpy wrapper | td_ludo/train_v6_2_fast.py (legacy)          |
| train_sl_v6_1.py             | runpy wrapper | td_ludo/train_sl_v6_1.py (legacy)            |
| generate_sl_data_v6_1.py     | runpy wrapper | td_ludo/generate_sl_data_v6_1.py (legacy)    |
| evaluate_v6_1.py             | runpy wrapper | td_ludo/evaluate_v6_1.py (legacy)            |
| evaluate_v6_2.py             | runpy wrapper | td_ludo/evaluate_v6_2.py (legacy)            |
| check_v62_parity.py          | runpy wrapper | td_ludo/check_v62_parity.py (legacy)         |
| debug_gameplay.py            | runpy wrapper | td_ludo/debug_gameplay.py (legacy)           |
| tune_heuristic.py            | runpy wrapper | td_ludo/tune_heuristic.py (legacy)           |

## NOT in scripts/ (intentional)

- **mcts_eval_sweep.py** — sweep is currently running on GCP. The file is
  in the hard sweep skip list and will be wrapped only after the sweep
  completes.

## Usage

```bash
cd td_ludo
./td_env/bin/python3 scripts/train_v6_1.py --resume
./td_env/bin/python3 scripts/evaluate_v6_1.py --num-games 500
```

The wrappers behave identically to invoking the legacy script directly.
