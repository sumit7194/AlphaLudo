"""DEPRECATED: this module has moved to td_ludo.models.v6_2.

Existing imports (``from src.model_v6_2 import AlphaLudoV62``)
keep working via the re-export below. Will be removed after Step B9.
"""
import warnings as _warnings

_warnings.warn(
    "src.model_v6_2 is deprecated; "
    "import from td_ludo.models.v6_2 instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.models.v6_2 import *  # noqa: F401,F403
from td_ludo.models.v6_2 import (  # noqa: F401
    AlphaLudoV62,
    ResidualBlock,
)
