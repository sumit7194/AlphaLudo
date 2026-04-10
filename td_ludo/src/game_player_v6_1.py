"""DEPRECATED: this module has moved to td_ludo.game.players.v6_1.

Existing imports keep working via the re-export below. Will be removed
after Step B9.
"""
import warnings as _warnings

_warnings.warn(
    "src.game_player_v6_1 is deprecated; "
    "import from td_ludo.game.players.v6_1 instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.game.players.v6_1 import *  # noqa: F401,F403
from td_ludo.game.players.v6_1 import VectorACGamePlayer  # noqa: F401
