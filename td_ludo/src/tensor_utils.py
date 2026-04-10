"""DEPRECATED: this module has moved to td_ludo.game.tensor_utils.

Existing imports (``from src.tensor_utils import get_board_coords``)
keep working via the re-export below. Will be removed after Step B9.
"""
import warnings as _warnings

_warnings.warn(
    "src.tensor_utils is deprecated; "
    "import from td_ludo.game.tensor_utils instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.game.tensor_utils import *  # noqa: F401,F403
from td_ludo.game.tensor_utils import (  # noqa: F401
    get_board_coords,
    get_safe_mask,
    get_home_path_mask,
    get_home_run_masks,
    state_to_tensor_mastery,
    BOARD_SIZE,
    NUM_PLAYERS,
    NUM_TOKENS,
    HOME_POS,
    BASE_POS,
    SAFE_INDICES,
    SAFE_MASK,
    HOME_PATH_MASK,
    HOME_RUN_MASKS,
)
