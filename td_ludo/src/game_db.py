"""DEPRECATED: this module has moved to td_ludo.data.game_db.

This shim re-exports the public symbols so existing imports
(``from src.game_db import GameDB``) keep working until all
callers are updated to use the new path.

Will be removed after Step B9.
"""
import warnings as _warnings

_warnings.warn(
    "src.game_db is deprecated; "
    "import from td_ludo.data.game_db instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.data.game_db import *  # noqa: F401,F403
from td_ludo.data.game_db import GameDB  # noqa: F401
