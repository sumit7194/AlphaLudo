"""Smoke tests for the heuristic bots.

NOTE: src/heuristic_bot.py is currently in the sweep skip list (the
running GCP MCTS sweep imports from `src.heuristic_bot`). It has not
been moved to td_ludo.game.heuristic_bot yet, so this test imports
from the legacy path.

Once Stage B1 unblocks (after the sweep finishes) and the move + shim
land, the same import will resolve through the shim and these tests
will keep passing without modification.

Run from inside td_ludo/:
    cd td_ludo
    python3 -m unittest tests.test_heuristic_bot -v
"""
import unittest

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


def _setup_state_with_dice(dice_roll: int):
    """Build a 2-player initial state with a known dice roll.

    Returns (state, legal_moves) so each test can pick a fresh state
    rather than mutating a shared one.
    """
    import td_ludo_cpp
    s = td_ludo_cpp.create_initial_state_2p()
    s.current_dice_roll = dice_roll
    legal = td_ludo_cpp.get_legal_moves(s)
    return s, legal


class TestHeuristicBots(unittest.TestCase):
    def test_imports(self):
        from src.heuristic_bot import (
            HeuristicLudoBot,
            AggressiveBot,
            DefensiveBot,
            RacingBot,
            RandomBot,
            ExpertBot,
            get_bot,
            BOT_REGISTRY,
        )
        self.assertIn("Expert", BOT_REGISTRY)
        self.assertIn("Heuristic", BOT_REGISTRY)
        self.assertIn("Aggressive", BOT_REGISTRY)
        self.assertIn("Defensive", BOT_REGISTRY)

    def test_expert_bot_select_move_dice_6(self):
        """ExpertBot must return a legal action when given a real
        game state with at least one legal move."""
        from src.heuristic_bot import ExpertBot
        bot = ExpertBot(player_id=0)
        s, legal = _setup_state_with_dice(6)
        self.assertGreater(len(legal), 0)
        action = bot.select_move(s, legal)
        self.assertIn(action, legal,
                      "ExpertBot returned an action that's not in the legal-move list")

    def test_random_bot_select_move(self):
        from src.heuristic_bot import RandomBot
        bot = RandomBot(player_id=0)
        s, legal = _setup_state_with_dice(6)
        action = bot.select_move(s, legal)
        self.assertIn(action, legal)

    def test_all_registered_bots_return_legal_action(self):
        """Every bot in BOT_REGISTRY must return a legal action on a
        normal game state. This catches regressions where a bot's
        evaluator returns -1 / None / out-of-range index."""
        from src.heuristic_bot import BOT_REGISTRY, get_bot
        s, legal = _setup_state_with_dice(6)
        for name in BOT_REGISTRY:
            with self.subTest(bot=name):
                bot = get_bot(name, player_id=0)
                action = bot.select_move(s, legal)
                self.assertIn(action, legal,
                              f"{name} returned non-legal action {action}")


if __name__ == "__main__":
    unittest.main()
