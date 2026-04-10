"""Smoke tests for the C++ game engine (td_ludo_cpp).

These verify the native extension loads and the basic gameplay
primitives work. They do NOT depend on any moved Python modules,
so they're safe to run during the refactor.

Run from inside td_ludo/:
    cd td_ludo
    python3 -m unittest tests.test_game_engine -v
"""
import unittest


class TestGameEngine(unittest.TestCase):
    def test_cpp_extension_imports(self):
        import td_ludo_cpp
        self.assertTrue(hasattr(td_ludo_cpp, "create_initial_state_2p"))
        self.assertTrue(hasattr(td_ludo_cpp, "create_initial_state"))
        self.assertTrue(hasattr(td_ludo_cpp, "get_legal_moves"))
        self.assertTrue(hasattr(td_ludo_cpp, "apply_move"))
        self.assertTrue(hasattr(td_ludo_cpp, "get_winner"))
        self.assertTrue(hasattr(td_ludo_cpp, "encode_state_v6"))

    def test_initial_state_2p(self):
        import td_ludo_cpp
        s = td_ludo_cpp.create_initial_state_2p()
        self.assertFalse(s.is_terminal)
        self.assertIn(s.current_player, (0, 1, 2, 3))
        # 2-player setup: P0 and P2 active, P1 and P3 inactive
        self.assertTrue(s.active_players[0])
        self.assertTrue(s.active_players[2])

    def test_legal_moves_dice_6_initial(self):
        import td_ludo_cpp
        s = td_ludo_cpp.create_initial_state_2p()
        s.current_dice_roll = 6
        moves = td_ludo_cpp.get_legal_moves(s)
        self.assertIsInstance(moves, list)
        # On dice 6 from initial state (all tokens in base) at least
        # one base token can spawn
        self.assertGreaterEqual(len(moves), 1)
        for m in moves:
            self.assertIn(m, range(0, 4))  # token index 0..3

    def test_legal_moves_dice_3_initial(self):
        import td_ludo_cpp
        s = td_ludo_cpp.create_initial_state_2p()
        s.current_dice_roll = 3
        moves = td_ludo_cpp.get_legal_moves(s)
        self.assertIsInstance(moves, list)
        # All tokens at base, dice != 6 → no legal moves
        self.assertEqual(len(moves), 0)

    def test_apply_move_after_spawn(self):
        import td_ludo_cpp
        s = td_ludo_cpp.create_initial_state_2p()
        s.current_dice_roll = 6
        moves = td_ludo_cpp.get_legal_moves(s)
        if moves:
            s2 = td_ludo_cpp.apply_move(s, moves[0])
            # State must transition meaningfully after a legal move
            self.assertFalse(s2.is_terminal)
            self.assertEqual(s2.current_dice_roll, 0,
                             "dice should reset after move")

    def test_encode_state_v6_shape(self):
        import td_ludo_cpp
        s = td_ludo_cpp.create_initial_state_2p()
        tensor = td_ludo_cpp.encode_state_v6(s)
        # 24-channel V6.1 strategic encoding, 15x15 board
        self.assertEqual(tensor.shape, (24, 15, 15))


if __name__ == "__main__":
    unittest.main()
