"""DEPRECATED: moved to td_ludo.training.inference_server."""
import warnings as _w
_w.warn("src.inference_server is deprecated; import from td_ludo.training.inference_server instead", DeprecationWarning, stacklevel=2)
from td_ludo.training.inference_server import *  # noqa
from td_ludo.training.inference_server import inference_server_worker  # noqa
