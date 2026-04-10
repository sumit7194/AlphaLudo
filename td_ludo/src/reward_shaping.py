"""DEPRECATED: this module has moved to td_ludo.game.reward_shaping.

Existing imports (``from src.reward_shaping import compute_shaped_reward``)
keep working via the re-export below. Will be removed after Step B9.
"""
import warnings as _warnings

_warnings.warn(
    "src.reward_shaping is deprecated; "
    "import from td_ludo.game.reward_shaping instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.game.reward_shaping import *  # noqa: F401,F403
from td_ludo.game.reward_shaping import (  # noqa: F401
    compute_shaped_reward,
    get_terminal_reward,
    HOME_STRETCH_START,
    SCORE_POSITION,
    SAFE_SQUARES,
)
