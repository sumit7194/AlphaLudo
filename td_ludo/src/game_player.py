"""DEPRECATED: this module has moved to td_ludo.game.players.base.

Existing imports keep working via the re-export below. Will be removed
after Step B9.
"""
import warnings as _warnings

_warnings.warn(
    "src.game_player is deprecated; "
    "import from td_ludo.game.players.base instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.game.players.base import *  # noqa: F401,F403
from td_ludo.game.players.base import VectorACGamePlayer  # noqa: F401
