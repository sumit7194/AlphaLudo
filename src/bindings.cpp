#include "game.h"
#include "mcts.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(ludo_cpp, m) {
  m.doc() = "AlphaLudo C++ Engine";

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
      .def_readwrite("is_terminal", &GameState::is_terminal);

  m.def("get_legal_moves", &get_legal_moves,
        "Get legal moves for current state");
  m.def("apply_move", &apply_move, "Apply a move to the state");
  m.def("get_winner", &get_winner, "Get winner (-1 if none)");
  m.def("create_initial_state", &create_initial_state,
        "Create initial game state");

  // MCTS Bindings
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
        // Return shape (batch, 21, 15, 15) - 21 Channel Stack
        int n_batch = data.size() / (21 * BOARD_SIZE * BOARD_SIZE);
        return py::array_t<float>({n_batch, 21, BOARD_SIZE, BOARD_SIZE},
                                  {21 * BOARD_SIZE * BOARD_SIZE * sizeof(float),
                                   BOARD_SIZE * BOARD_SIZE * sizeof(float),
                                   BOARD_SIZE * sizeof(float), sizeof(float)},
                                  data.data());
      });
}
