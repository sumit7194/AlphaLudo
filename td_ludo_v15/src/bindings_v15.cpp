// pybind11 bindings for the V15 engine (`td_ludo_v15_cpp`).
//
// Exposes the cell-based public API plus read-only state inspection helpers
// (`player_positions`, `scores`, etc.) used by tests and by the Python-side
// encoder iteration.
//
// Module name `td_ludo_v15_cpp` deliberately differs from legacy `td_ludo_cpp`
// so both can be imported into the same Python process without symbol clash.

#include "game_v15.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace v15;

PYBIND11_MODULE(td_ludo_v15_cpp, m) {
  m.doc() = "AlphaLudo V15 engine — cell-based API, fresh build.";

  // Constants
  m.attr("BOARD_SIZE") = BOARD_SIZE;
  m.attr("NUM_PLAYERS") = NUM_PLAYERS;
  m.attr("NUM_TOKENS") = NUM_TOKENS;
  m.attr("PATH_LENGTH") = PATH_LENGTH;
  m.attr("HOME_RUN_LENGTH") = HOME_RUN_LENGTH;
  m.attr("HOME_POS") = HOME_POS;
  m.attr("BASE_POS") = BASE_POS;

  py::class_<GameState>(m, "GameState")
      .def(py::init<>())
      .def_property_readonly(
          "board",
          [](GameState& s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>(
                {BOARD_SIZE, BOARD_SIZE},
                {BOARD_SIZE * sizeof(int8_t), sizeof(int8_t)},
                &s.board[0][0], py::cast(s));
          })
      .def_property_readonly(
          "player_positions",
          [](GameState& s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>(
                {NUM_PLAYERS, NUM_TOKENS},
                {NUM_TOKENS * sizeof(int8_t), sizeof(int8_t)},
                &s.player_positions[0][0], py::cast(s));
          })
      .def_property_readonly(
          "scores",
          [](GameState& s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>({NUM_PLAYERS}, {sizeof(int8_t)},
                                       &s.scores[0], py::cast(s));
          })
      .def_property_readonly(
          "active_players",
          [](GameState& s) -> py::array_t<bool> {
            return py::array_t<bool>({NUM_PLAYERS}, {sizeof(bool)},
                                     s.active_players.data(), py::cast(s));
          })
      .def_property_readonly(
          "consecutive_sixes",
          [](GameState& s) -> py::array_t<int8_t> {
            return py::array_t<int8_t>({NUM_PLAYERS}, {sizeof(int8_t)},
                                       &s.consecutive_sixes[0], py::cast(s));
          })
      .def_readonly("current_player", &GameState::current_player)
      .def_readonly("current_dice_roll", &GameState::current_dice_roll)
      .def_readonly("is_terminal", &GameState::is_terminal);

  // ─── Construction ──────────────────────────────────────────────────────
  m.def("create_initial_state", &create_initial_state, "4-player initial state");
  m.def("create_initial_state_2p", &create_initial_state_2p,
        "2-player initial state (P0 vs P2)");

  // ─── Dice ──────────────────────────────────────────────────────────────
  m.def("set_dice", &set_dice,
        "Set dice value; atomically handles 3-six forfeit (turn passes, dice=0).",
        py::arg("state"), py::arg("dice_value"));

  m.def("pass_turn", &pass_turn,
        "Advance turn to next active player (use when no legal moves). "
        "Resets dice=0 and current player's consecutive_sixes.");

  // ─── Cell-based API ────────────────────────────────────────────────────
  m.def("get_legal_source_cells", &get_legal_source_cells,
        "Return unique (row, col) cells where the current player has a movable token.");
  m.def("apply_move_from_cell", &apply_move_from_cell,
        "Apply a move whose source is (row, col). Engine picks deterministic slot internally.",
        py::arg("state"), py::arg("row"), py::arg("col"));

  // ─── Position queries (no token-IDs) ───────────────────────────────────
  m.def("get_own_positions", &get_own_positions,
        "Return list of positions of the current player's tokens (multiplicity preserved).");
  m.def("get_opp_positions", &get_opp_positions,
        "Return list of positions of opponent player's tokens.",
        py::arg("state"), py::arg("opp_player"));
  m.def("position_to_cell", &position_to_cell,
        "Convert path position to (row, col) cell in given player's POV.",
        py::arg("pos"), py::arg("player"));

  // ─── Terminal / winner ─────────────────────────────────────────────────
  m.def("get_winner", &get_winner, "Return winner player-id or -1.");

  // ─── Internal helpers (testing / parity only — NOT for V15 inference) ──
  m.def("_internal_get_legal_token_slots", &_internal_get_legal_token_slots,
        "[INTERNAL] Returns slot indices (0..3) of legally-movable tokens. "
        "Used by parity tests against legacy engine. V15 production code must "
        "use get_legal_source_cells instead.");
  m.def("_internal_apply_move_by_slot", &_internal_apply_move_by_slot,
        "[INTERNAL] Apply move by slot index. For parity tests only.",
        py::arg("state"), py::arg("slot"));
}
