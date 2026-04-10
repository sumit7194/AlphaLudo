"""DEPRECATED: moved to td_ludo.training.fast_actor_v62."""
import warnings as _w
_w.warn("src.fast_actor_v62 is deprecated; import from td_ludo.training.fast_actor_v62 instead", DeprecationWarning, stacklevel=2)
from td_ludo.training.fast_actor_v62 import *  # noqa
from td_ludo.training.fast_actor_v62 import (TurnHistory, FastActor, actor_worker, actor_worker_gpu)  # noqa
