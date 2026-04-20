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

  m.def("encode_state_v9", [](const GameState &state) {
    // Return shape (14, 15, 15) - V9 14 Channel Stack
    py::array_t<float> result({14, BOARD_SIZE, BOARD_SIZE});
    auto buf = result.mutable_data();
    write_state_tensor_v9(state, buf);
    return result;
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
