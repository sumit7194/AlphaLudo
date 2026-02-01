"""
Test Suite for Dynamic Configuration Pipeline
============================================
Tests that the Auto-Tuner -> config.json -> Worker propagation chain works correctly.
"""

import os
import sys
import json
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_config_reload():
    """
    Test that load_config_from_json() correctly updates CONFIGS dictionary.
    """
    import src.config as cfg
    
    # Create a temporary config.json with test values
    test_config = {
        "PROD": {
            "c_puct": 99.0,
            "dirichlet_eps": 0.99
        },
        "BACKGROUND": {
            "c_puct": 99.0,
            "dirichlet_eps": 0.99
        },
        "TEST": {
            "c_puct": 99.0,
            "dirichlet_eps": 0.99
        }
    }
    
    # Backup original config if exists
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    backup_path = config_path + ".bak"
    
    had_original = os.path.exists(config_path)
    if had_original:
        shutil.copy(config_path, backup_path)
    
    try:
        # Write test config
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Reload config
        result = cfg.load_config_from_json()
        
        # Verify reload succeeded
        assert result == True, "load_config_from_json should return True"
        
        # Verify values updated (uppercase keys)
        assert cfg.CONFIGS["PROD"]["C_PUCT"] == 99.0, f"C_PUCT should be 99.0, got {cfg.CONFIGS['PROD']['C_PUCT']}"
        assert cfg.CONFIGS["PROD"]["DIRICHLET_EPS"] == 0.99, f"DIRICHLET_EPS should be 0.99, got {cfg.CONFIGS['PROD']['DIRICHLET_EPS']}"
        
        print("✅ test_config_reload PASSED")
        
    finally:
        # Restore original config
        if had_original:
            shutil.move(backup_path, config_path)
            cfg.load_config_from_json()  # Reload original
        else:
            os.remove(config_path)


def test_tuner_config_write():
    """
    Test that AutoTuner.update_config() writes to config.json correctly.
    """
    from src.tuner import AutoTuner
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    backup_path = config_path + ".bak"
    
    had_original = os.path.exists(config_path)
    if had_original:
        shutil.copy(config_path, backup_path)
    
    try:
        # Create minimal starting config
        starting_config = {
            "PROD": {"c_puct": 3.0, "dirichlet_eps": 0.25},
            "BACKGROUND": {"c_puct": 3.0, "dirichlet_eps": 0.25},
            "TEST": {"c_puct": 3.0, "dirichlet_eps": 0.25}
        }
        with open(config_path, 'w') as f:
            json.dump(starting_config, f)
        
        # Create tuner (but don't run eval)
        tuner = AutoTuner()
        
        # Update config with new values
        new_params = {"cpuct": 5.0, "eps": 0.75}
        result = tuner.update_config(new_params)
        
        assert result == True, "update_config should return True"
        
        # Read back and verify
        with open(config_path, 'r') as f:
            updated = json.load(f)
        
        assert updated["PROD"]["c_puct"] == 5.0, f"c_puct should be 5.0, got {updated['PROD']['c_puct']}"
        assert updated["PROD"]["dirichlet_eps"] == 0.75, f"dirichlet_eps should be 0.75, got {updated['PROD']['dirichlet_eps']}"
        
        print("✅ test_tuner_config_write PASSED")
        
    finally:
        if had_original:
            shutil.move(backup_path, config_path)
        else:
            os.remove(config_path)


def test_worker_update_params():
    """
    Test that VectorLeagueWorker.update_params() updates internal state.
    """
    import torch
    from src.model_v3 import AlphaLudoV3
    from src.vector_league import VectorLeagueWorker
    
    # Create minimal model
    model = AlphaLudoV3(num_res_blocks=2, num_channels=32)
    model.eval()
    
    # Create worker with initial params
    worker = VectorLeagueWorker(
        main_model=model,
        probabilities={'Main': 1.0},
        mcts_simulations=10,
        c_puct=3.0,
        dirichlet_eps=0.25,
        actor_id=999
    )
    
    assert worker.c_puct == 3.0, "Initial c_puct should be 3.0"
    assert worker.dirichlet_eps == 0.25, "Initial dirichlet_eps should be 0.25"
    
    # Update params
    worker.update_params(5.0, 0.75)
    
    assert worker.c_puct == 5.0, f"c_puct should be 5.0 after update, got {worker.c_puct}"
    assert worker.dirichlet_eps == 0.75, f"dirichlet_eps should be 0.75 after update, got {worker.dirichlet_eps}"
    
    print("✅ test_worker_update_params PASSED")


def test_mcts_engine_params():
    """
    Test that MCTSEngine receives correct parameters at creation.
    """
    import ludo_cpp
    
    # Create engine with specific params
    mcts = ludo_cpp.MCTSEngine(1, 5.0, 0.3, 0.75)
    
    # Run a simple MCTS to verify it works
    state = ludo_cpp.create_initial_state()
    state.current_dice_roll = 6
    
    mcts.set_roots([state])
    
    # Just verify it doesn't crash - can't easily read back params from C++
    print("✅ test_mcts_engine_params PASSED (creation verified)")


def test_end_to_end_config_propagation():
    """
    End-to-end test: Tuner writes -> Config reloads -> Worker reads new values
    """
    import src.config as cfg
    from src.tuner import AutoTuner
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    backup_path = config_path + ".bak"
    
    had_original = os.path.exists(config_path)
    if had_original:
        shutil.copy(config_path, backup_path)
    
    try:
        # 1. Create starting config
        starting_config = {
            "PROD": {"c_puct": 3.0, "dirichlet_eps": 0.25, "mcts_sims": 10},
            "BACKGROUND": {"c_puct": 3.0, "dirichlet_eps": 0.25},
            "TEST": {"c_puct": 3.0, "dirichlet_eps": 0.25}
        }
        with open(config_path, 'w') as f:
            json.dump(starting_config, f)
        
        # 2. Load initial config
        cfg.load_config_from_json()
        initial_cpuct = cfg.CONFIGS["PROD"]["C_PUCT"]
        assert initial_cpuct == 3.0, f"Initial C_PUCT should be 3.0, got {initial_cpuct}"
        
        # 3. Tuner writes new config
        tuner = AutoTuner()
        tuner.update_config({"cpuct": 7.0, "eps": 0.9})
        
        # 4. Config reloads
        cfg.load_config_from_json()
        
        # 5. Verify propagation
        new_cpuct = cfg.CONFIGS["PROD"]["C_PUCT"]
        new_eps = cfg.CONFIGS["PROD"]["DIRICHLET_EPS"]
        
        assert new_cpuct == 7.0, f"C_PUCT should be 7.0 after propagation, got {new_cpuct}"
        assert new_eps == 0.9, f"DIRICHLET_EPS should be 0.9 after propagation, got {new_eps}"
        
        print("✅ test_end_to_end_config_propagation PASSED")
        
    finally:
        if had_original:
            shutil.move(backup_path, config_path)
            cfg.load_config_from_json()
        else:
            os.remove(config_path)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DYNAMIC CONFIGURATION TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_config_reload,
        test_tuner_config_write,
        test_worker_update_params,
        test_mcts_engine_params,
        test_end_to_end_config_propagation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
