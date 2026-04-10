"""Smoke test for td_ludo package skeleton.

Run from inside td_ludo/ so the cwd-implicit sys.path resolves the
inner td_ludo/ package:

    cd td_ludo
    python3 -m unittest tests.test_package_imports
"""
import unittest


class TestPackageImports(unittest.TestCase):
    def test_root_package(self):
        import td_ludo
        self.assertTrue(hasattr(td_ludo, "__file__"))

    def test_subpackages_exist(self):
        import td_ludo.models
        import td_ludo.game
        import td_ludo.training
        import td_ludo.eval
        import td_ludo.data
        import td_ludo.play
        import td_ludo._native


if __name__ == "__main__":
    unittest.main()
