#include "mcts.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

// Constants
// Constants
const float EPSILON = 1e-8f;

MCTSEngine::MCTSEngine(int batch_size, float c_puct, float dirichlet_alpha,
                       float dirichlet_eps)
    : batch_size(batch_size), c_puct(c_puct), dirichlet_alpha(dirichlet_alpha),
      dirichlet_eps(dirichlet_eps) {
  roots.resize(batch_size);
  current_leaves.resize(batch_size, nullptr);
  thread_active.resize(batch_size, true);
}

void MCTSEngine::set_roots(const std::vector<GameState> &states) {
  if (states.size() != batch_size) {
    // Just assume correct for now
  }

  for (int i = 0; i < batch_size; ++i) {
    roots[i] = std::make_unique<MCTSNode>(states[i]);
    thread_active[i] = true;

    // Check immediate terminal or no moves
    std::vector<int> moves = get_legal_moves(states[i]);
    if (states[i].is_terminal || moves.empty()) {
      // Effectively done for MCTS purposes if root has no moves
      // But usually we handle this before calling MCTS.
      // If it happens, mark as expanded with no children.
      roots[i]->is_expanded = true;
      roots[i]->legal_moves = moves;
    }
  }
}

float MCTSEngine::ucb_score(MCTSNode *parent, MCTSNode *child, float prior) {
  float pb_c =
      std::log((float(parent->visit_count) + c_puct + 1.0f) / c_puct) + c_puct;
  pb_c *= prior;
  pb_c /= (float(child->visit_count) + 1.0f);

  float val = child->get_value();

  // CORRECTION: Adversarial Perspective Flip
  if (child->state.current_player != parent->state.current_player) {
    val = -val;
  }

  // Normalize value from [-1, 1] to [0, 1] for UCB integration
  float val_norm = (val + 1.0f) / 2.0f;

  return val_norm + pb_c;
}

MCTSNode *MCTSEngine::select_child(MCTSNode *node) {
  if (node->children.empty())
    return nullptr;

  // [NEW] Chance Node Logic
  if (node->is_chance) {
    // Uniform random selection for Ludo dice (1-6)
    // We want to explore all outcomes.
    // Simple strategy: Pick uniform random.
    // Over many simulations, this approximates the distribution.
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0,
                                                    node->children.size() - 1);
    int idx = distribution(generator);
    return node->children[idx].get();
  }

  // Action Node Logic (UCB)
  MCTSNode *best_child = nullptr;
  float best_score = -1e9f;

  for (size_t i = 0; i < node->children.size(); ++i) {
    MCTSNode *child = node->children[i].get();
    float prior = node->move_priors[i];

    float score = ucb_score(node, child, prior);
    if (score > best_score) {
      best_score = score;
      best_child = child;
    }
  }

  return best_child;
}

std::vector<GameState> MCTSEngine::select_leaves(int parallel_sims) {
  int total_leaves = batch_size * parallel_sims;
  std::vector<GameState> leaf_states;
  leaf_states.reserve(total_leaves);

  // Resize current_leaves to hold all selection pointers
  current_leaves.resize(total_leaves, nullptr);

  for (int i = 0; i < batch_size; ++i) {
    // If thread is inactive (e.g. game finished), fill slots with nullptr or
    // skip? We usually just skip, but then indexing gets messy. Let's rely on
    // thread_active[i] and just fill nulls if inactive.

    for (int k = 0; k < parallel_sims; ++k) {
      int leaf_idx = i * parallel_sims + k;

      if (!thread_active[i]) {
        current_leaves[leaf_idx] = nullptr;
        continue;
      }

      MCTSNode *node = roots[i].get();

      while (node->is_expanded) {
        if (node->children.empty()) {
          break;
        }

        // Selection Policy
        node = select_child(node);

        // VIRTUAL LOSS: Increment visit count immediately to discourage
        // subsequent threads from picking this same path.
        node->visit_count++;
      }

      // Also increment root visit count? usually no, root is implicit.
      // But select_child increments child.
      // We need to mark the leaf itself as visited too?
      // If we stop at `node`, `select_child` didn't happen for `node`.
      // So we must increment `node->visit_count` explicitly?
      // In AlphaZero: Virtual Loss is applied to edges traversed.
      // If we are at the leaf, we haven't traversed an edge *from* it yet
      // (unless expanded). But we will expand it. Let's increment leaf visit
      // count too, so other threads avoid it? Yes, "Virtual Visit".
      node->visit_count++;

      current_leaves[leaf_idx] = node;
      leaf_states.push_back(node->state);
    }
  }

  return leaf_states;
}

void MCTSEngine::expand_and_backprop(
    const std::vector<std::vector<float>> &policies,
    const std::vector<float> &values) {
  // Determine parallelism from input size
  // total inputs = batch_size * parallel_sims
  // parallel_sims = values.size() / batch_size

  if (batch_size == 0)
    return;
  int total_inputs = values.size();
  int parallel_sims = total_inputs / batch_size;

  int input_idx = 0;

  for (int i = 0; i < batch_size; ++i) {

    for (int k = 0; k < parallel_sims; ++k) {
      int leaf_idx = i * parallel_sims + k;

      if (!thread_active[i] || current_leaves[leaf_idx] == nullptr) {
        input_idx++;
        continue;
      }

      MCTSNode *leaf = current_leaves[leaf_idx];

      if (input_idx >= (int)values.size())
        break;

      float value = values[input_idx];
      const std::vector<float> &policy = policies[input_idx];
      input_idx++;

      // EXPANSION
      if (!leaf->is_expanded) {
        // [NEW] Check for Chance Node (Waiting for Roll)
        if (!leaf->state.is_terminal && leaf->state.current_dice_roll == 0) {
          leaf->is_expanded = true;
          leaf->is_chance = true;
          leaf->children.resize(6);
          for (int r = 1; r <= 6; ++r) {
            GameState next_s = leaf->state;
            next_s.current_dice_roll = r;
            auto child = std::make_unique<MCTSNode>(next_s);
            child->parent = leaf;
            child->move_from_parent = r;
            leaf->children[r - 1] = std::move(child);
          }
        } else {
          std::vector<int> legal_moves = get_legal_moves(leaf->state);
          int winner = get_winner(leaf->state);

          if (winner != -1) {
            leaf->is_expanded = true;
            leaf->legal_moves = {};
            if (winner == leaf->state.current_player)
              value = 1.0f;
            else
              value = -1.0f;

          } else if (legal_moves.empty()) {
            leaf->is_expanded = true;
            value = 0.0f;
          } else {
            leaf->is_expanded = true;
            leaf->legal_moves = legal_moves;
            leaf->move_priors.resize(legal_moves.size());

            float sum_prob = 0.0f;
            int current_p = leaf->state.current_player;

            for (size_t m = 0; m < legal_moves.size(); ++m) {
              int token_idx = legal_moves[m];

              // v3: Policy is now 4-dimensional softmax probabilities (one per
              // token) Direct index into policy vector - NO exp() needed,
              // already probabilities
              float p = 0.0f;
              if (token_idx >= 0 && token_idx < (int)policy.size()) {
                p = policy[token_idx]; // Already a probability, don't exp()!
              }

              leaf->move_priors[m] = p;
              sum_prob += p;
            }

            if (sum_prob > EPSILON) {
              for (float &p : leaf->move_priors)
                p /= sum_prob;
            } else {
              float uni = 1.0f / legal_moves.size();
              for (float &p : leaf->move_priors)
                p = uni;
            }

            // v3: Apply Dirichlet noise at ROOT node for exploration
            // Check if this leaf is the root (parent is nullptr)
            if (leaf->parent == nullptr && dirichlet_eps > 0.0f) {
              static thread_local std::mt19937 gen(std::random_device{}());
              std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);

              std::vector<float> noise(leaf->move_priors.size());
              float noise_sum = 0.0f;
              for (size_t n = 0; n < noise.size(); ++n) {
                noise[n] = gamma(gen);
                noise_sum += noise[n];
              }
              if (noise_sum > EPSILON) {
                for (size_t n = 0; n < noise.size(); ++n) {
                  noise[n] /= noise_sum; // Normalize to get Dirichlet sample
                  // Mix prior with noise
                  leaf->move_priors[n] =
                      (1.0f - dirichlet_eps) * leaf->move_priors[n] +
                      dirichlet_eps * noise[n];
                }
              }
            }

            leaf->children.resize(legal_moves.size());
            for (size_t m = 0; m < legal_moves.size(); ++m) {
              int move = legal_moves[m];
              GameState next_s = apply_move(leaf->state, move);

              auto child = std::make_unique<MCTSNode>(next_s);
              child->parent = leaf;
              child->move_from_parent = move;
              leaf->children[m] = std::move(child);
            }
          }
        } // End of Action Node Expansion
      } // End of Chance/Action check

      // BACKPROPAGATION with Expecti-MCTS for Chance Nodes
      MCTSNode *node = leaf;
      float curr_val = value;

      while (node != nullptr) {
        // For Chance Nodes (dice rolls), compute the AVERAGE of all children
        // This implements Expectimax: V(chance) = (1/6) * sum(V(child))
        if (node->is_chance && !node->children.empty()) {
          float avg_child_value = 0.0f;
          int valid_children = 0;
          for (const auto &child : node->children) {
            if (child && child->visit_count > 0) {
              avg_child_value += child->get_value();
              valid_children++;
            }
          }
          if (valid_children > 0) {
            // Use the expected value (average) instead of the sampled value
            curr_val = avg_child_value / valid_children;
          }
          // Update the chance node's value as the average
          node->value_sum = curr_val * node->visit_count;
        } else {
          // Standard backprop for Action Nodes
          node->value_sum += curr_val;
        }

        // Flip value for opponent perspective
        if (node->parent) {
          if (node->parent->state.current_player !=
              node->state.current_player) {
            curr_val = -curr_val;
          }
        }
        node = node->parent;
      }
    }
  }
}

std::vector<std::vector<float>>
MCTSEngine::get_action_probs(float temperature) {
  std::vector<std::vector<float>> results;
  results.reserve(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    MCTSNode *root = roots[i].get();
    std::vector<float> probs(4, 0.0f); // Size 4 fixed

    if (root->children.empty()) {
      // Should not happen unless no moves
      // Uniform valid moves?
      if (!root->legal_moves.empty()) {
        for (int m : root->legal_moves)
          probs[m] = 1.0f / root->legal_moves.size();
      } else {
        probs[0] = 1.0f; // placeholder
      }
    } else {
      // Calculate based on visits
      float sum_visits = 0.0f;

      if (temperature == 0.0f) {
        // Argmax
        int best_visits = -1;
        int best_idx = -1;
        for (size_t k = 0; k < root->children.size(); ++k) {
          int v = root->children[k]->visit_count;
          if (v > best_visits) {
            best_visits = v;
            best_idx = k;
          }
        }
        int move = root->legal_moves[best_idx];
        probs[move] = 1.0f;
      } else {
        // Exponentiate
        // visits^(1/temp)
        std::vector<float> temp_visits;
        for (size_t k = 0; k < root->children.size(); ++k) {
          float v = std::pow(float(root->children[k]->visit_count),
                             1.0f / temperature);
          temp_visits.push_back(v);
          sum_visits += v;
        }

        for (size_t k = 0; k < root->children.size(); ++k) {
          int move = root->legal_moves[k];
          if (sum_visits > 1e-8f) {
            probs[move] = temp_visits[k] / sum_visits;
          } else {
            // Fallback: uniform over legal moves
            probs[move] = 1.0f / root->children.size();
          }
        }
      }
    }

    results.push_back(probs);
  }

  return results;
}

std::vector<std::pair<int, float>> MCTSEngine::get_root_stats() {
  std::vector<std::pair<int, float>> stats;
  for (int i = 0; i < batch_size; ++i) {
    stats.push_back({roots[i]->visit_count, roots[i]->get_value()});
  }
  return stats;
}

std::vector<float> MCTSEngine::get_leaf_tensors() {
  // 17 channels (V5 Architecture) * 15 * 15 = 3825 floats per leaf
  int tensor_size = 17 * BOARD_SIZE * BOARD_SIZE;

  // Note: current_leaves was resized in select_leaves to (batch_size *
  // parallel_sims)
  int total_leaves = current_leaves.size();

  std::vector<float> buffer(total_leaves * tensor_size, 0.0f);

  // Iterate over All leaves
  // Note: 'thread_active' is per GAME (batch_size), but we flatten leaves.
  // We must handle the mapping carefully, or just check null.
  // In select_leaves, we put nullptr if thread inactive.

  for (int i = 0; i < total_leaves; ++i) {
    MCTSNode *leaf = current_leaves[i];
    if (leaf) {
      float *ptr = buffer.data() + (i * tensor_size);
      write_state_tensor(leaf->state, ptr);
    }
  }
  return buffer;
}
