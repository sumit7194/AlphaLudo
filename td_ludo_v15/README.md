# td_ludo_v15

V15 — Graph Transformer + per-cell triplet encoder + fresh cell-based engine.

**Status**: under construction. Code + tests phase. No real training yet — that happens on the VM later when V13.5 RL finishes.

This is a **completely separate** parallel pipeline from `../td_ludo/`. The old engine and all V12.x/V13.x/V14_scalar/V13.5 code is untouched.

## Architecture (locked, see `../td_ludo/V15_DESIGN_PLAN.md`)

- **Engine**: brand new cpp module `td_ludo_v15_cpp`. **Cell-based API only** — token-IDs are an internal implementation detail, never exposed.
- **Input**: per-cell triplet `(a, b, c)` over 15×15 board, stacked across 8 chronological frames.
- **Output**: per-source-cell policy (225 logits) + sigmoid value scalar. No aux heads.
- **Backbone**: Graph Transformer, 4 layers, d=192, 8 heads, edge-biased attention. Hand-rolled (no PyG dependency).

## Quickstart

```sh
cd td_ludo_v15
pip install -e .                # builds td_ludo_v15_cpp ext
pytest -q                       # fast tests
pytest -q -m slow               # parity (10K games) + POC learning check
python -m td_ludo_v15.scripts.dump_state --seed 42 --moves 35
```

## Folder layout

```
td_ludo_v15/
├── src/                         # cpp engine sources
├── td_ludo_v15/                 # python package
│   ├── game/                    # cells, state wrapper, encoder, graph
│   ├── models/                  # GraphTransformer blocks + V15 model
│   ├── viz/                     # board+encoding side-by-side viewer
│   └── scripts/                 # dump_state, poc_learn CLIs
└── tests/                       # pytest suite + golden files
```
