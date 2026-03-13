# AlphaLudo Client App

This folder is a minimal export for building a standalone gameplay app around the AlphaLudo model.

## Included

- `play/model_weights/model.pt`
  - Copied from `model_latest_323k_shaped.pt`
  - This is the main gameplay checkpoint used by the exported server
- `weights/model_latest_323k_shaped.pt`
  - Original backup filename preserved for reference
- `weights/model_best_323k_shaped.pt`
  - Best-eval backup preserved for reference
- `play/server.py`
  - Flask gameplay server for human-vs-AI play
- `play/model.py`
  - Inference-time PyTorch model definition
- `play/static/`
  - Lightweight web UI assets
- `src/game.cpp`, `src/game.h`, `src/bindings.cpp`
  - C++ game engine and Python bindings
- `src/mcts.cpp`, `src/mcts.h`
  - Included because the extension build references them
- `setup.py`, `pyproject.toml`, `requirements.txt`
  - Build/install files for the `td_ludo_cpp` extension

## Model Architecture

The client app uses `AlphaLudoV5` from `play/model.py`.

- Input: `(B, 17, 15, 15)`
- Backbone: `10` residual blocks
- Width: `128` channels
- Policy output: `4` token logits
- Value output: `1` scalar in `[-1, 1]`

For gameplay, the server masks illegal actions and chooses among token indices `0..3`.

## Input Tensor

The server uses the C++ encoder through `td_ludo_cpp.encode_state(...)`.

Channel layout:

- `0-3`: current player's tokens
- `4`: opponent token density
- `5`: safe zones
- `6`: current player's home path
- `7`: opponent home paths
- `8`: broadcast score difference
- `9`: broadcast current-player locked fraction
- `10`: broadcast opponent locked fraction
- `11-16`: one-hot dice roll channels

The board is rotated into the acting player's perspective before inference.

## Outputs

The model returns:

- `policy`: probability distribution over the four token slots
- `value`: scalar position value for the current player

The gameplay server uses the policy head for move selection and can log the value head for debugging.

## Quick Start

Build the extension and install dependencies:

```bash
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

Run the local gameplay server:

```bash
cd play
python server.py
```

By default, the exported server expects:

- Human player: `P0`
- AI player: `P2`
- Model path: `play/model_weights/model.pt`

## Important Notes

- The exported client app is intentionally just a copied baseline, not a cleaned production app yet.
- The exact input conversion path used at runtime is the C++ binding, not a Python encoder.
- The copied `weights/model_best_323k_shaped.pt` is preserved for comparison, but the default server points at the latest `323k` shaped PPO snapshot.
