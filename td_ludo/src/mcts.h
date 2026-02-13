#ifndef MCTS_H
#define MCTS_H

#include "game.h"
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

// MCTS Node
struct MCTSNode {
  GameState state;

  int visit_count = 0;
  float value_sum = 0.0f;
  float prior = 0.0f;

  // Edges
  // Map move_index (0-3 usually, but wait, ApplyMove takes int token_index)
  // Actually, get_legal_moves returns vector<int>.
  // So children is map<int, MCTSNode*>
  // We use raw pointers for simplicity, owned by the node (or by a pool).
  // Let's use unique_ptr for ownership.
  std::vector<std::unique_ptr<MCTSNode>> children;
  std::vector<int> legal_moves; // Cached legal moves
  std::vector<float> move_priors;

  bool is_expanded = false;
  bool is_chance = false; // Is this a stochastic node (dice roll)?

  // For backprop
  MCTSNode *parent = nullptr;
  int move_from_parent = -1;

  MCTSNode(const GameState &s) : state(s) {}

  float get_value() const {
    if (visit_count == 0)
      return 0.0f;
    return value_sum / visit_count;
  }
};

class MCTSEngine {
public:
  MCTSEngine(int batch_size, float c_puct = 3.0f, float dirichlet_alpha = 0.3f,
             float dirichlet_eps = 0.25f);

  // Reset trees for new states
  void set_roots(const std::vector<GameState> &states);

  // Step 1: Selection
  // Returns list of leaf nodes (states) that need evaluation
  // and a mapping ID to identify them in backprop step.
  // Actually, we can just store the active leaf pointers internally.
  // Step 1: Selection
  // Returns list of leaf nodes (states) that need evaluation
  // argument parallel_sims: Number of leaf nodes to select per game (Virtual
  // Loss)
  std::vector<GameState> select_leaves(int parallel_sims = 1);

  // Step 2: Expansion & Backprop
  // policies: [batch_size, 4] (flat or whatever)
  // values: [batch_size]
  void expand_and_backprop(const std::vector<std::vector<float>> &policies,
                           const std::vector<float> &values);

  // Final Result
  std::vector<std::vector<float>> get_action_probs(float temperature);

  // Get root estimates (for debug/stats)
  // returns vector of {visit, value}
  std::vector<std::pair<int, float>> get_root_stats();

  // Phase 6: Get Tensors for current leaves (Batch x 18 x 15 x 15)
  // Returns flattened vector [Batch * 4050]
  std::vector<float> get_leaf_tensors();

private:
  int batch_size;
  float c_puct; // Exploration constant (default 3.0 for stochastic games)
  float dirichlet_alpha; // Dirichlet noise alpha parameter
  float dirichlet_eps;   // Dirichlet noise mixing weight

  // One root per game in the batch
  std::vector<std::unique_ptr<MCTSNode>> roots;

  // Current active leaves for the ongoing simulation step
  // Indices correspond to the game index (0..batch_size-1)
  // If a game is terminal, its pointer might be null or point to terminal node?
  std::vector<MCTSNode *> current_leaves;
  std::vector<bool>
      thread_active; // If a game is finished processing (e.g. terminal), skip

  // Helpers
  MCTSNode *select_child(MCTSNode *node);
  float ucb_score(MCTSNode *parent, MCTSNode *child, float prior);
};

#endif // MCTS_H
