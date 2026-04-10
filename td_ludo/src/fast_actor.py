"""DEPRECATED: moved to td_ludo.training.fast_actor."""
import warnings as _w
_w.warn("src.fast_actor is deprecated; import from td_ludo.training.fast_actor instead", DeprecationWarning, stacklevel=2)
from td_ludo.training.fast_actor import *  # noqa
from td_ludo.training.fast_actor import (TurnHistory, FastActor, actor_worker, actor_worker_gpu)  # noqa
