"""DEPRECATED: moved to td_ludo.game.heuristic_bot."""
import warnings as _w
_w.warn("src.heuristic_bot is deprecated; import from td_ludo.game.heuristic_bot instead", DeprecationWarning, stacklevel=2)
from td_ludo.game.heuristic_bot import *  # noqa
from td_ludo.game.heuristic_bot import (HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, RandomBot, ExpertBot, get_bot, BOT_REGISTRY)  # noqa
