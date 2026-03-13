# AlphaLudo MechInterp

This folder is a minimal export of the AlphaLudo model snapshot and the code needed to inspect how the model sees game states.

## Included

- `weights/model_latest_323k_shaped.pt`
  - PPO shaped-reward checkpoint at `total_games = 323186`
- `weights/model_best_323k_shaped.pt`
  - Best-eval checkpoint backup from the same experiment family
  - Note: the checkpoint metadata reports `total_games = 262269`
- `src/model.py`
  - Main PyTorch model definitions
- `src/tensor_utils.py`
  - Python reference encoder for the board tensor
- `src/game.cpp`, `src/game.h`, `src/bindings.cpp`
  - C++ game engine and Python bindings
- `src/mcts.cpp`, `src/mcts.h`
  - Copied for completeness because the extension build includes them
- `setup.py`, `pyproject.toml`, `requirements.txt`
  - Build/install files for the `td_ludo_cpp` extension

## Model Architecture

The exported checkpoint is compatible with `AlphaLudoV5` in `src/model.py`.

- Input: `(B, 17, 15, 15)`
- Backbone: `10` residual blocks
- Width: `128` channels
- Policy head: `4` logits, one per token index
- Value head: `1` scalar in `[-1, 1]`
- Aux head: `4` safety values, used during training only

The policy is not a 225-cell board policy. It directly predicts which token to move.

## Input Tensor

The active encoder is a `17-channel` state tensor:

- `0-3`: current player's four tokens, one channel per token
- `4`: opponent token density (`0.25` per opponent token)
- `5`: safe-zone mask
- `6`: current player's home path
- `7`: opponent home paths
- `8`: broadcast score difference
- `9`: broadcast fraction of current player's locked/base tokens
- `10`: broadcast fraction of opponent locked/base tokens
- `11-16`: one-hot dice roll channels for rolls `1..6`

Spatial channels are rotated into the current player's perspective before being fed to the model.

## Outputs

For a single position, the model returns:

- `policy`: probabilities over token indices `[0, 1, 2, 3]`
- `value`: expected outcome for the current player, scaled to `[-1, 1]`
- `aux_safety`: optional token-wise safety scores

## Quick Start

Build the C++ extension:

```bash
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

Load the model:

```python
import torch
from src.model import AlphaLudoV5

ckpt = torch.load("weights/model_latest_323k_shaped.pt", map_location="cpu")
model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=17)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Where The Real Encoder Lives

If you are tracing the exact production input path, the key entry points are:

- `src/game.cpp`
  - `write_state_tensor(...)`
- `src/bindings.cpp`
  - `encode_state(...)`
  - `VectorGameState.get_state_tensor(...)`

`src/tensor_utils.py` is useful as a readable Python reference, but the live gameplay/training path uses the C++ encoder.
