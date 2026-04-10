"""DEPRECATED: this module has moved to td_ludo.eval.elo_tracker.

This shim re-exports the public symbols so existing imports
(``from src.elo_tracker import EloTracker``) keep working until
all callers are updated to use the new path.

Will be removed after Step B9 (one full re-eval cycle confirms safety).
"""
import warnings as _warnings

_warnings.warn(
    "src.elo_tracker is deprecated; "
    "import from td_ludo.eval.elo_tracker instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.eval.elo_tracker import *  # noqa: F401,F403
from td_ludo.eval.elo_tracker import EloTracker  # noqa: F401
