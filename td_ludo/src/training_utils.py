"""DEPRECATED: this module has moved to td_ludo.training.utils.

Existing imports keep working via the re-export below. Will be
removed after Step B9. Note: as of A7 there are zero Python callers
in the repo (deploy.sh references the path but doesn't import it).
"""
import warnings as _warnings

_warnings.warn(
    "src.training_utils is deprecated; "
    "import from td_ludo.training.utils instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.training.utils import *  # noqa: F401,F403
from td_ludo.training.utils import (  # noqa: F401
    get_temperature,
    rotate_state_tensor,
    rotate_policy,
    rotate_token_indices,
    rotate_channels,
    augment_training_sample,
    augment_batch,
    TrainingMetrics,
)
