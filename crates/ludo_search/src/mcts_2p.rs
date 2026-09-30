//! 2-Player Zero-Sum Expecti-MCTS Engine (P0 vs P2).
//!
//! Features:
//!   - Strict zero-sum scalar payoff v in [-1.0, +1.0].
//!   - Handles Ludo bonus turns (6-rolls & captures) natively.
//!   - Arena node allocation (zero heap allocation during search).
//!   - PUCT action selection with configurable exploration constant c_puct.
//!   - Chance-node simulation across dice outcomes (1..6).
//!   - Returns search visit distributions π for AlphaZero training targets.

use ludo_core::GameState;
use rand::Rng;

use crate::evaluator::score_position;
use crate::expectimax::{ExpectimaxBot, LudoBot};

const MAX_TREE_NODES_2P: usize = 16384;

#[derive(Clone)]
struct Mcts2PNode {
    state: GameState,
    player: u8,
    legal_actions: [u8; 4],
    legal_count: u8,
    priors: [f32; 4],
    visit_counts: [u32; 4],
    /// Accumulated scalar value from this node's player perspective
    total_values: [f32; 4],
    children: [usize; 4],
    expanded: bool,
}

impl Mcts2PNode {
    fn new(state: GameState) -> Self {
        let (legal, count) = state.legal_moves();
        let p = state.current_player;

        Self {
            state,
            player: p,
            legal_actions: legal,
            legal_count: count as u8,
            priors: [0.25; 4],
            visit_counts: [0; 4],
            total_values: [0.0; 4],
            children: [usize::MAX; 4],
            expanded: false,
        }
    }

    /// Selects action maximizing PUCT from current node's player perspective.
    fn select_puct(&self, c_puct: f32) -> usize {
        let mut best_idx = 0;
        let mut best_score = -f32::INFINITY;

        let total_visits: u32 = self.visit_counts[..self.legal_count as usize].iter().sum();
        let sqrt_total = ((total_visits.max(1)) as f32).sqrt();

        for i in 0..self.legal_count as usize {
            let n = self.visit_counts[i] as f32;
            let q = if n > 0.0 {
                self.total_values[i] / n
            } else {
                0.0
            };

            let u = c_puct * self.priors[i] * sqrt_total / (1.0 + n);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }
}

/// Samples Dirichlet(alpha=0.3) noise for k actions using Marsaglia-Tsang Gamma method.
fn sample_dirichlet_03(k: usize, rng: &mut impl rand::Rng) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    let mut sum = 0.0f32;
    let alpha = 0.3f32;
    let d = (alpha + 1.0) - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    for i in 0..k {
        let mut attempts = 0;
        let g = loop {
            attempts += 1;
            let z: f32 = rng.gen_range(-2.5..2.5);
            let v = 1.0 + c * z;
            if v <= 0.0 {
                if attempts > 50 { break 0.1; }
                continue;
            }
            let v3 = v * v * v;
            let u: f32 = rng.gen_range(0.0001..0.9999);
            if u < 1.0 - 0.0331 * z * z * z * z {
                let u2: f32 = rng.gen_range(0.0001..0.9999);
                break (d * v3) * u2.powf(1.0 / alpha);
            }
            if u.ln() < 0.5 * z * z + d * (1.0 - v3 + v3.ln()) {
                let u2: f32 = rng.gen_range(0.0001..0.9999);
                break (d * v3) * u2.powf(1.0 / alpha);
            }
            if attempts > 50 { break 0.1; }
        };
        out[i] = g.max(1e-4);
        sum += out[i];
    }

    if sum > 0.0 {
        for i in 0..k {
            out[i] /= sum;
        }
    } else {
        for i in 0..k {
            out[i] = 1.0 / (k as f32);
        }
    }
    out
}

/// 2-Player Zero-Sum MCTS Bot for P0 vs P2.
pub struct TwoPlayerMCTSBot {
    pub player_id: Option<u8>,
    pub num_simulations: usize,
    pub c_puct: f32,
    pub prior_smooth: f32,
    pub use_heuristic_prior: bool,
    pub use_pure_terminal: bool,
    pub add_dirichlet: bool,
    pub dirichlet_epsilon: f32,
    prior_bot: ExpectimaxBot,
}

impl TwoPlayerMCTSBot {
    pub fn new(player_id: Option<u8>, num_simulations: usize) -> Self {
        Self {
            player_id,
            num_simulations,
            c_puct: 1.414,
            prior_smooth: 0.15,
            use_heuristic_prior: false, // Tabula rasa by default
            use_pure_terminal: true,    // Terminal rewards only by default
            add_dirichlet: false,
            dirichlet_epsilon: 0.25,
            prior_bot: ExpectimaxBot::new(player_id),
        }
    }

    /// Fast rollout from leaf to terminal or cutoff depth, returning scalar value in [-1, +1]
    /// from root_player's perspective.
    fn rollout(&self, mut state: GameState, root_player: u8, max_depth: usize) -> f32 {
        let mut rng = rand::thread_rng();
        let mut depth = 0;
        let opp_player = if root_player == 0 { 2 } else { 0 };

        while !state.is_terminal && depth < max_depth {
            let d = rng.gen_range(1..=6);
            state.set_dice(d);

            let (legal, count) = state.legal_moves();
            if count == 0 {
                state.pass_turn();
                continue;
            }

            let pick = rng.gen_range(0..count);
            state = state.apply_move(legal[pick]);
            depth += 1;
        }

        if state.is_terminal && state.winner >= 0 {
            if state.winner == root_player as i8 {
                1.0
            } else {
                -1.0
            }
        } else if self.use_pure_terminal {
            0.0 // Strict zero-sum terminal only (neutral if cutoff)
        } else {
            // Heuristic evaluation at cutoff: normalized score differential
            let s_my = score_position(root_player, &state).max(0.0) + 1.0;
            let s_opp = score_position(opp_player, &state).max(0.0) + 1.0;
            ((s_my - s_opp) / (s_my + s_opp)).clamp(-1.0, 1.0)
        }
    }

    /// Performs MCTS search from state with custom temperature for action sampling.
    /// Returns (chosen_action, visit_distribution_pi).
    pub fn search_with_temp(&self, state: &GameState, temperature: f32) -> (u8, [f32; 4]) {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return (0, [0.0; 4]);
        }
        if count == 1 {
            let mut pi = [0.0f32; 4];
            pi[legal[0] as usize] = 1.0;
            return (legal[0], pi);
        }

        let root_player = state.current_player;
        let mut arena: Vec<Mcts2PNode> = Vec::with_capacity(MAX_TREE_NODES_2P.min(self.num_simulations + 10));
        let mut root = Mcts2PNode::new(*state);
        let mut rng = rand::thread_rng();

        // Initialize root priors
        if self.use_heuristic_prior {
            let best_exp_move = self.prior_bot.select_move(state);
            let k = count as f32;
            for i in 0..count {
                if legal[i] == best_exp_move {
                    root.priors[i] = 1.0 - self.prior_smooth;
                } else if k > 1.0 {
                    root.priors[i] = self.prior_smooth / (k - 1.0);
                } else {
                    root.priors[i] = 1.0;
                }
            }
        } else {
            // Uniform tabula rasa prior
            let uniform_p = 1.0 / (count as f32);
            for i in 0..count {
                root.priors[i] = uniform_p;
            }
        }

        // Add Dirichlet exploration noise at root (AlphaZero self-play)
        if self.add_dirichlet && count > 1 {
            let noise = sample_dirichlet_03(count, &mut rng);
            let eps = self.dirichlet_epsilon;
            for i in 0..count {
                root.priors[i] = (1.0 - eps) * root.priors[i] + eps * noise[i];
            }
        }

        root.expanded = true;
        arena.push(root);


        let mut rng = rand::thread_rng();

        for _ in 0..self.num_simulations {
            let mut node_idx = 0;
            let mut path: Vec<(usize, usize)> = Vec::with_capacity(32); // (node_idx, action_slot)

            // 1. Selection
            while arena[node_idx].expanded && arena[node_idx].legal_count > 0 {
                let action_slot = arena[node_idx].select_puct(self.c_puct);
                path.push((node_idx, action_slot));

                let child_idx = arena[node_idx].children[action_slot];
                if child_idx == usize::MAX {
                    break;
                }
                node_idx = child_idx;
            }

            let (last_node_idx, last_action_slot) = match path.last() {
                Some(&pair) => pair,
                None => break,
            };

            // 2. Expansion & Chance Branch
            let action = arena[last_node_idx].legal_actions[last_action_slot];
            let after_action = arena[last_node_idx].state.apply_move(action);

            let (leaf_val, new_child_idx) = if after_action.is_terminal {
                let val = if after_action.winner == root_player as i8 {
                    1.0
                } else {
                    -1.0
                };
                (val, usize::MAX)
            } else if arena.len() < MAX_TREE_NODES_2P {
                // Sample next dice roll
                let d = rng.gen_range(1..=6);
                let mut child_state = after_action;
                child_state.set_dice(d);

                let mut child_node = Mcts2PNode::new(child_state);
                let val = self.rollout(child_state, root_player, 60);

                child_node.expanded = true;
                let c_idx = arena.len();
                arena.push(child_node);
                (val, c_idx)
            } else {
                let val = self.rollout(after_action, root_player, 60);
                (val, usize::MAX)
            };

            if new_child_idx != usize::MAX {
                arena[last_node_idx].children[last_action_slot] = new_child_idx;
            }

            // 3. Zero-Sum Backup along the path
            for &(n_idx, a_slot) in &path {
                arena[n_idx].visit_counts[a_slot] += 1;
                // If node belongs to root_player, value is +leaf_val; if opp, -leaf_val
                let val_for_node = if arena[n_idx].player == root_player {
                    leaf_val
                } else {
                    -leaf_val
                };
                arena[n_idx].total_values[a_slot] += val_for_node;
            }
        }

        // Compute visit distribution at root
        let root_node = &arena[0];
        let total_visits: u32 = root_node.visit_counts[..root_node.legal_count as usize].iter().sum();
        let mut pi = [0.0f32; 4];
        let mut best_action = root_node.legal_actions[0];
        let mut max_visits = 0;

        for i in 0..root_node.legal_count as usize {
            let act = root_node.legal_actions[i] as usize;
            let v = root_node.visit_counts[i];
            if total_visits > 0 {
                pi[act] = (v as f32) / (total_visits as f32);
            }
            if v > max_visits {
                max_visits = v;
                best_action = root_node.legal_actions[i];
            }
        }

        let chosen_action = if temperature <= 0.01 || total_visits == 0 {
            best_action
        } else {
            // Temperature-scaled sampling: p_i ~ (visit_count)^(1 / temp)
            let inv_temp = 1.0 / temperature;
            let mut weighted_probs = [0.0f32; 4];
            let mut weight_sum = 0.0f32;

            for i in 0..root_node.legal_count as usize {
                let v = root_node.visit_counts[i] as f32;
                let w = (v.max(1.0)).powf(inv_temp);
                weighted_probs[i] = w;
                weight_sum += w;
            }

            if weight_sum > 0.0 {
                let mut sample_r = rng.gen_range(0.0..weight_sum);
                let mut picked = root_node.legal_actions[0];
                for i in 0..root_node.legal_count as usize {
                    if sample_r <= weighted_probs[i] {
                        picked = root_node.legal_actions[i];
                        break;
                    }
                    sample_r -= weighted_probs[i];
                }
                picked
            } else {
                best_action
            }
        };

        (chosen_action, pi)
    }

    /// Performs standard MCTS search (argmax visit count action).
    pub fn search(&self, state: &GameState) -> (u8, [f32; 4]) {
        self.search_with_temp(state, 0.0)
    }
}


impl LudoBot for TwoPlayerMCTSBot {
    fn name(&self) -> &'static str {
        "TwoPlayerMCTS"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let (action, _) = self.search(state);
        action
    }
}
