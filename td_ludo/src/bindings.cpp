#include "game.h"
#include "mcts.h"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class VectorGameState {
public:
  VectorGameState(int batch_size, bool two_player_mode_in)
      : batch_size(batch_size), two_player_mode(two_player_mode_in) {
    games.reserve(batch_size);
    reset();
  }

  void reset() {
    games.clear();
    for (int i = 0; i < batch_size; ++i) {
      if (two_player_mode) {
        games.push_back(create_initial_state_2p());
      } else {
        games.push_back(create_initial_state());
      }
    }
  }

  void reset_game(int i) {
    if (i < 0 || i >= batch_size)
      return;
    if (two_player_mode) {
      games[i] = create_initial_state_2p();
    } else {
      games[i] = create_initial_state();
    }
  }

  // Returns (next_states, rewards, dones, info_list)
  // actions: vector of token indices. If -1, no op.
  py::tuple step(const std::vector<int> &actions) {
    if (actions.size() != (size_t)batch_size) {
      throw std::runtime_error("Action batch size mismatch");
    }

    std::vector<float> rewards(batch_size, 0.0f);
    std::vector<uint8_t> dones(batch_size, 0); // uint8_t for numpy bool
    py::list infos;

    for (int i = 0; i < batch_size; ++i) {
      int action = actions[i];

      // If game is already terminal, we don't step it, but we might define
      // behavior
      if (!games[i].is_terminal && action >= 0) {
        // Apply move
        games[i] = apply_move(games[i], action);

        if (games[i].is_terminal) {
          dones[i] = 1;
        }
      } else {
        // Game over or no-op
        if (games[i].is_terminal) {
          dones[i] = 1;
        }
      }

      // Info dict
      py::dict info;
      info["is_terminal"] = games[i].is_terminal;
      if (games[i].is_terminal) {
        info["winner"] = get_winner(games[i]);
      } else {
        info["winner"] = -1;
      }
      info["current_player"] = games[i].current_player;
      info["current_dice_roll"] = games[i].current_dice_roll;
      infos.append(info);
    }

    // Get next states tensor
    py::array_t<float> next_states = get_state_tensor();

    return py::make_tuple(
        next_states, py::array_t<float>(batch_size, rewards.data()),
        py::array_t<bool>(batch_size, (bool *)dones.data()), infos);
  }

  py::array_t<float> get_state_tensor() {
    // Shape: (B, 17, 15, 15)
    size_t channel_size = BOARD_SIZE * BOARD_SIZE;
    size_t sample_size = 17 * channel_size;
    // size_t total_size = batch_size * sample_size; // unused

    auto result = py::array_t<float>({batch_size, 17, BOARD_SIZE, BOARD_SIZE});
    auto buffer = result.mutable_data();

    // Parallelize? No, GIL is held. Just sequential optimized writing.
    for (int i = 0; i < batch_size; ++i) {
      write_state_tensor(games[i], buffer + (i * sample_size));
    }
    return result;
  }

  // Returns list of lists
  std::vector<std::vector<int>> get_legal_moves() {
    std::vector<std::vector<int>> batch_moves;
    batch_moves.reserve(batch_size);
    for (const auto &g : games) {
      batch_moves.push_back(::get_legal_moves(g));
    }
    return batch_moves;
  }

  // Get raw games (for Python inspection if needed)
  GameState &get_game(int index) {
    if (index < 0 || index >= batch_size)
      throw std::out_of_range("Index error");
    return games[index];
  }

private:
  int batch_size;
  bool two_player_mode;
  std::vector<GameState> games;
};

PYBIND11_MODULE(td_ludo_cpp, m) {
  m.doc() = "AlphaLudo C++ Engine (Isolated for TD Learning)";

  py::class_<GameState>(m, "GameState")
      .def(py::init<>())
      .def_property(
          "board",
          [](GameState &s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>(
                {BOARD_SIZE, BOARD_SIZE},
                {BOARD_SIZE * sizeof(int8_t), sizeof(int8_t)}, &s.board[0][0],
                py::cast(s));
          },
          [](GameState &s, py::array_t<int8_t> array) {
            std::memcpy(&s.board[0][0], array.data(),
                        BOARD_SIZE * BOARD_SIZE * sizeof(int8_t));
          })
      .def_property(
          "player_positions",
          [](GameState &s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>(
                {NUM_PLAYERS, NUM_TOKENS},
                {NUM_TOKENS * sizeof(int8_t), sizeof(int8_t)},
                &s.player_positions[0][0], py::cast(s));
          },
          [](GameState &s, py::array_t<int8_t> array) {
            std::memcpy(&s.player_positions[0][0], array.data(),
                        NUM_PLAYERS * NUM_TOKENS * sizeof(int8_t));
          })
      .def_property(
          "scores",
          [](GameState &s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>({NUM_PLAYERS}, {sizeof(int8_t)},
                                       &s.scores[0], py::cast(s));
          },
          [](GameState &s, py::array_t<int8_t> array) {
            std::memcpy(&s.scores[0], array.data(),
                        NUM_PLAYERS * sizeof(int8_t));
          })
      .def_readwrite("current_player", &GameState::current_player)
      .def_readwrite("current_dice_roll", &GameState::current_dice_roll)
      .def_readwrite("is_terminal", &GameState::is_terminal)
      .def_property(
          "active_players",
          [](GameState &s) -> py::array_t<bool> {
            return py::array_t<bool>({NUM_PLAYERS}, {sizeof(bool)},
                                     s.active_players.data(), py::cast(s));
          },
          [](GameState &s, py::array_t<bool> array) {
            auto buf = array.unchecked<1>();
            for (int i = 0; i < NUM_PLAYERS; ++i)
              s.active_players[i] = buf(i);
          })
      // V12.1 (encoder v11): per-token idleness + same-token streak.
      .def_property(
          "idle_counter",
          [](GameState &s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>(
                {NUM_PLAYERS, NUM_TOKENS},
                {NUM_TOKENS * sizeof(int8_t), sizeof(int8_t)},
                &s.idle_counter[0][0], py::cast(s));
          },
          [](GameState &s, py::array_t<int8_t> array) {
            std::memcpy(&s.idle_counter[0][0], array.data(),
                        NUM_PLAYERS * NUM_TOKENS * sizeof(int8_t));
          })
      .def_property(
          "last_moved_token",
          [](GameState &s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>({NUM_PLAYERS}, {sizeof(int8_t)},
                                       &s.last_moved_token[0], py::cast(s));
          },
          [](GameState &s, py::array_t<int8_t> array) {
            std::memcpy(&s.last_moved_token[0], array.data(),
                        NUM_PLAYERS * sizeof(int8_t));
          })
      .def_property(
          "streak",
          [](GameState &s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>({NUM_PLAYERS}, {sizeof(int8_t)},
                                       &s.streak[0], py::cast(s));
          },
          [](GameState &s, py::array_t<int8_t> array) {
            std::memcpy(&s.streak[0], array.data(),
                        NUM_PLAYERS * sizeof(int8_t));
          });

  m.def("get_legal_moves", &get_legal_moves,
        "Get legal moves for current state");
  m.def("apply_move", &apply_move, "Apply a move to the state");

  m.def("encode_state", [](const GameState &state) {
    // Return shape (17, 15, 15) - Single 17 Channel Stack
    py::array_t<float> result({17, BOARD_SIZE, BOARD_SIZE});
    auto buf = result.mutable_data();
    write_state_tensor(state, buf);
    return result;
  });

  m.def("encode_state_v6", [](const GameState &state) {
    py::array_t<float> result({24, BOARD_SIZE, BOARD_SIZE});
    auto buf = result.mutable_data();
    write_state_tensor_v6(state, buf);
    return result;
  });

  m.def("encode_state_v6_3",
        [](const GameState &state, int consecutive_sixes) {
          // Return shape (27, 15, 15) - V6.3 27 Channel Stack
          py::array_t<float> result({27, BOARD_SIZE, BOARD_SIZE});
          auto buf = result.mutable_data();
          write_state_tensor_v6_3(state, buf, consecutive_sixes);
          return result;
        },
        py::arg("state"), py::arg("consecutive_sixes") = 0);

  m.def("encode_state_v10", [](const GameState &state) {
    // Return shape (28, 15, 15) - V10 Strategic Stack
    py::array_t<float> result({28, BOARD_SIZE, BOARD_SIZE});
    auto buf = result.mutable_data();
    write_state_tensor_v10(state, buf);
    return result;
  });

  m.def("encode_state_v11", [](const GameState &state) {
    // Return shape (33, 15, 15) - V11 = V10 + per-own-token idle (4) + streak (1)
    py::array_t<float> result({33, BOARD_SIZE, BOARD_SIZE});
    auto buf = result.mutable_data();
    write_state_tensor_v11(state, buf);
    return result;
  });

  m.def("encode_state_v9", [](const GameState &state) {
    // Return shape (14, 15, 15) - V9 14 Channel Stack
    py::array_t<float> result({14, BOARD_SIZE, BOARD_SIZE});
    auto buf = result.mutable_data();
    write_state_tensor_v9(state, buf);
    return result;
  });

  m.def("encode_state_v14_minimal", [](const GameState &state) {
    // Return shape (14, 15, 15) - V14 Minimal Distillation Stack
    py::array_t<float> result({14, BOARD_SIZE, BOARD_SIZE});
    auto buf = result.mutable_data();
    write_state_tensor_v14_minimal(state, buf);
    return result;
  });

  // V14_scalar: non-spatial encoder for the DeepSets model. Returns a dict
  // of numpy arrays + scalars. See V14ScalarEncoding in game.h for layout.
  m.def("encode_state_v14_scalar", [](const GameState &state) {
    V14ScalarEncoding enc;
    write_state_v14_scalar(state, enc);
    py::dict out;

    // Per-own-token features (NUM_TOKENS = 4 each)
    auto own_pos = py::array_t<int8_t>({NUM_TOKENS});
    std::memcpy(own_pos.mutable_data(), enc.own_pos_emb,
                NUM_TOKENS * sizeof(int8_t));
    out["own_pos"] = own_pos;

    auto own_in_danger = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(own_in_danger.mutable_data(), enc.own_in_danger,
                NUM_TOKENS * sizeof(bool));
    out["own_in_danger"] = own_in_danger;

    auto own_can_capture = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(own_can_capture.mutable_data(), enc.own_can_capture,
                NUM_TOKENS * sizeof(bool));
    out["own_can_capture"] = own_can_capture;

    auto own_can_score = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(own_can_score.mutable_data(), enc.own_can_score,
                NUM_TOKENS * sizeof(bool));
    out["own_can_score"] = own_can_score;

    auto own_can_land_safe = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(own_can_land_safe.mutable_data(), enc.own_can_land_safe,
                NUM_TOKENS * sizeof(bool));
    out["own_can_land_safe"] = own_can_land_safe;

    auto own_is_safe = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(own_is_safe.mutable_data(), enc.own_is_safe,
                NUM_TOKENS * sizeof(bool));
    out["own_is_safe"] = own_is_safe;

    auto own_at_base = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(own_at_base.mutable_data(), enc.own_at_base,
                NUM_TOKENS * sizeof(bool));
    out["own_at_base"] = own_at_base;

    auto own_at_home = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(own_at_home.mutable_data(), enc.own_at_home,
                NUM_TOKENS * sizeof(bool));
    out["own_at_home"] = own_at_home;

    auto own_idle = py::array_t<float>({NUM_TOKENS});
    std::memcpy(own_idle.mutable_data(), enc.own_idle_count,
                NUM_TOKENS * sizeof(float));
    out["own_idle_count"] = own_idle;

    // Per-opp-token features
    auto opp_pos = py::array_t<int8_t>({NUM_TOKENS});
    std::memcpy(opp_pos.mutable_data(), enc.opp_pos_emb,
                NUM_TOKENS * sizeof(int8_t));
    out["opp_pos"] = opp_pos;

    auto opp_in_my_danger = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(opp_in_my_danger.mutable_data(), enc.opp_in_my_danger,
                NUM_TOKENS * sizeof(bool));
    out["opp_in_my_danger"] = opp_in_my_danger;

    auto opp_threatens_me = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(opp_threatens_me.mutable_data(), enc.opp_threatens_me,
                NUM_TOKENS * sizeof(bool));
    out["opp_threatens_me"] = opp_threatens_me;

    auto opp_is_safe = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(opp_is_safe.mutable_data(), enc.opp_is_safe,
                NUM_TOKENS * sizeof(bool));
    out["opp_is_safe"] = opp_is_safe;

    auto opp_at_base = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(opp_at_base.mutable_data(), enc.opp_at_base,
                NUM_TOKENS * sizeof(bool));
    out["opp_at_base"] = opp_at_base;

    auto opp_at_home = py::array_t<bool>({NUM_TOKENS});
    std::memcpy(opp_at_home.mutable_data(), enc.opp_at_home,
                NUM_TOKENS * sizeof(bool));
    out["opp_at_home"] = opp_at_home;

    // Globals (Python-side scalars; Python wrapper will pack into a vector)
    out["dice"] = (int)enc.dice;
    out["same_token_streak"] = enc.same_token_streak;
    out["my_locked_frac"] = enc.my_locked_frac;
    out["opp_locked_frac"] = enc.opp_locked_frac;
    out["score_diff"] = enc.score_diff;
    out["leader_progress"] = enc.leader_progress;
    out["non_home_tokens_frac"] = enc.non_home_tokens_frac;
    out["bonus_turn_flag"] = enc.bonus_turn_flag;

    return out;
  });

  m.def("get_winner", &get_winner, "Get winner (-1 if none)");
  m.def("create_initial_state", &create_initial_state,
        "Create initial 4-player game state");
  m.def("create_initial_state_2p", &create_initial_state_2p,
        "Create initial 2-player game state (P0 vs P2)");

  // Vector Env Bindings
  py::class_<VectorGameState>(m, "VectorGameState")
      .def(py::init<int, bool>(), py::arg("batch_size"),
           py::arg("two_player_mode") = false)
      .def("reset", &VectorGameState::reset)
      .def("reset_game", &VectorGameState::reset_game)
      .def("step", &VectorGameState::step, py::arg("actions"))
      .def("get_state_tensor", &VectorGameState::get_state_tensor)
      .def("get_legal_moves", &VectorGameState::get_legal_moves)
      .def("get_game", &VectorGameState::get_game,
           py::return_value_policy::reference);

  // MCTS Bindings (Reserved for compatibility)
  py::class_<MCTSEngine>(m, "MCTSEngine")
      .def(py::init<int, float, float, float>(), py::arg("batch_size"),
           py::arg("c_puct") = 3.0f, py::arg("dirichlet_alpha") = 0.3f,
           py::arg("dirichlet_eps") = 0.25f)
      .def("set_roots", &MCTSEngine::set_roots)
      .def("select_leaves", &MCTSEngine::select_leaves,
           py::arg("parallel_sims") = 1,
           py::call_guard<py::gil_scoped_release>())
      .def("expand_and_backprop", &MCTSEngine::expand_and_backprop,
           py::call_guard<py::gil_scoped_release>())
      .def("get_action_probs", &MCTSEngine::get_action_probs,
           py::call_guard<py::gil_scoped_release>())
      .def("get_root_stats", &MCTSEngine::get_root_stats)
      .def("get_leaf_tensors", [](MCTSEngine &self) {
        std::vector<float> data = self.get_leaf_tensors();
        // Return shape (batch, 24, 15, 15) - V6.1 strategic 24 Channel Stack
        int n_batch = data.size() / (24 * BOARD_SIZE * BOARD_SIZE);
        return py::array_t<float>({n_batch, 24, BOARD_SIZE, BOARD_SIZE},
                                  {24 * BOARD_SIZE * BOARD_SIZE * sizeof(float),
                                   BOARD_SIZE * BOARD_SIZE * sizeof(float),
                                   BOARD_SIZE * sizeof(float), sizeof(float)},
                                  data.data());
      });
}
