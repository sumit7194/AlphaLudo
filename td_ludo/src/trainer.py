"""DEPRECATED: this module has moved to td_ludo.training.trainer.

Existing imports (``from src.trainer import ActorCriticTrainer``)
keep working via the re-export below. Will be removed after Step B9.
"""
import warnings as _warnings

_warnings.warn(
    "src.trainer is deprecated; "
    "import from td_ludo.training.trainer instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.training.trainer import *  # noqa: F401,F403
from td_ludo.training.trainer import ActorCriticTrainer  # noqa: F401
