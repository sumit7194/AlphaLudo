"""DEPRECATED: moved to td_ludo.training.fast_learner."""
import warnings as _w
_w.warn("src.fast_learner is deprecated; import from td_ludo.training.fast_learner instead", DeprecationWarning, stacklevel=2)
from td_ludo.training.fast_learner import *  # noqa
from td_ludo.training.fast_learner import (FastLearner, learner_worker)  # noqa
