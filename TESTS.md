# AlphaLudo Test Suite Documentation

This document catalogs all test and evaluation scripts in the AlphaLudo project.

---

## Tests Directory (`/tests/`)

| File | Purpose |
|------|---------|
| `test_augmentation.py` | Tests data augmentation for training samples |
| `test_cpp_mcts.py` | Tests C++ MCTS engine bindings |
| `test_dynamic_config.py` | **NEW** Tests dynamic configuration propagation pipeline |
| `test_engine.py` | Tests game engine rules and state transitions |
| `test_hybrid_tensor.py` | Tests hybrid tensor representations |
| `test_mcts.py` | Tests MCTS search algorithm |
| `test_model.py` | Tests neural network model forward pass |
| `test_specialist.py` | Tests specialist training components |
| `test_training.py` | Tests training loop components |
| `test_vector.py` | Tests vectorized operations |
| `verify_pbt_cpuct.py` | Verifies Population-Based Training c_puct |

### Debug/Repro Scripts
| File | Purpose |
|------|---------|
| `debug_cut.py` | Debugging cut/capture logic |
| `debug_single_game.py` | Debug single game execution |
| `repro_game_reset.py` | Reproduce game reset issues |
| `repro_issue.py` | General issue reproduction |
| `repro_skip.py` | Reproduce skip/pass issues |

---

## Source Directory Tests (`/src/`)

| File | Purpose |
|------|---------|
| `test_advanced.py` | Advanced integration tests (MCTS, model, training) |
| `test_edge_cases.py` | Edge case handling for game logic |
| `test_pure_model.py` | **EVAL** Pure model evaluation vs heuristic bots |
| `test_tensor_consistency.py` | Tensor encoding/decoding consistency |
| `test_training_flow.py` | End-to-end training flow tests |
| `audit_pipeline.py` | **AUDIT** Full pipeline audit script |

---

## Running Tests

### Quick Validation
```bash
# Run dynamic config tests
source .venv/bin/activate
python tests/test_dynamic_config.py

# Run model evaluation
python src/test_pure_model.py
```

### Full Test Suite
```bash
# Run all tests in tests/ directory
python -m pytest tests/ -v

# Run specific test
python tests/test_mcts.py
```

---

## Dynamic Configuration Tests

The `test_dynamic_config.py` file tests the following chain:

```
Auto-Tuner → config.json → Config Reload → Worker → MCTSEngine
```

### Tests Included:
1. **test_config_reload** - Verifies `load_config_from_json()` updates CONFIGS dict
2. **test_tuner_config_write** - Verifies `AutoTuner.update_config()` writes to JSON
3. **test_worker_update_params** - Verifies `VectorLeagueWorker.update_params()` updates state
4. **test_mcts_engine_params** - Verifies MCTSEngine creation with params
5. **test_end_to_end_config_propagation** - Full chain test

---

## Model Evaluation

Use `src/test_pure_model.py` to evaluate model performance:

```bash
# Evaluate default model
python src/test_pure_model.py

# Evaluate specific run
ALPHALUDO_RUN_NAME=mastery_v3_prod python src/test_pure_model.py
```

### Expected Metrics:
- **Win Rate**: Should be >25% (random baseline)
- **vs Heuristic Bots**: Mix of Aggressive, Defensive, Racing, Random

---

## Adding New Tests

1. Create test file in `tests/` directory
2. Follow naming convention: `test_<component>.py`
3. Include docstrings explaining test purpose
4. Update this document
