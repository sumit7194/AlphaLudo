//! 4-Player Multi-Agent Max^n Monte Carlo Tree Search (MCTS) with Expectimax Prior.
//!
//! Implements game-theoretic Max^n tree search for 4-player Ludo:
//!   - Every node maintains a 4-dimensional payoff vector V = [v0, v1, v2, v3].
//!   - At each decision depth, player `p` selects the action that maximizes their OWN utility Q_p(a).
//!   - Root expansion is guided by an Expectimax prior to direct search down high-value branches.
//!   - Fast rollout simulations with arena-allocated node pools (0 heap allocations during search).

use ludo_core::{GameState, NUM_PLAYERS};
use rand::Rng;

use crate::evaluator::score_position;
use crate::expectimax::{ExpectimaxBot, LudoBot};

const MAX_TREE_NODES: usize = 4096;

/// A single node in the Max^n search tree.
#[derive(Clone)]
struct MctsNode {
    state: GameState,
    player: u8,
    legal_actions: [u8; 4],
    legal_count: u8,
    priors: [f32; 4],
    visit_counts: [u32; 4],
    /// Accumulated 4-dimensional payoff vectors: total_payoffs[action][player]
    total_payoffs: [[f32; 4]; 4],
    /// Children node indices in the arena. None = usize::MAX
    children: [usize; 4],
    expanded: bool,
}

impl MctsNode {
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
            total_payoffs: [[0.0; 4]; 4],
            children: [usize::MAX; 4],
            expanded: false,
        }
    }

    /// Selects best action using PUCT over player p's own coordinate in the payoff vector.
    fn select_puct(&self, c_puct: f32) -> usize {
        let mut best_idx = 0;
        let mut best_score = -f32::INFINITY;
        let p = self.player as usize;

        let total_visits: u32 = self.visit_counts[..self.legal_count as usize].iter().sum();
        let sqrt_total = ((total_visits.max(1)) as f32).sqrt();

        for i in 0..self.legal_count as usize {
            let n = self.visit_counts[i] as f32;
            let q = if n > 0.0 {
                self.total_payoffs[i][p] / n
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

/// 4-Player Max^n MCTS Bot.
pub struct MaxNMCTSBot {
    pub player_id: Option<u8>,
    pub num_simulations: usize,
    pub c_puct: f32,
    pub prior_smooth: f32,
    prior_bot: ExpectimaxBot,
}

impl MaxNMCTSBot {
    pub fn new(player_id: Option<u8>, num_simulations: usize) -> Self {
        Self {
            player_id,
            num_simulations,
            c_puct: 1.414,
            prior_smooth: 0.15,
            prior_bot: ExpectimaxBot::new(player_id),
        }
    }

    /// Executes rollouts from a state until terminal or max_depth, returning 4-way payoffs.
    fn rollout(&self, mut state: GameState, max_depth: usize) -> [f32; 4] {
        let mut rng = rand::thread_rng();
        let mut depth = 0;

        while !state.is_terminal && depth < max_depth {
            let d = rng.gen_range(1..=6);
            state.set_dice(d);

            let (legal, count) = state.legal_moves();
            if count == 0 {
                state.pass_turn();
                continue;
            }

            // Fast random rollout
            let pick = rng.gen_range(0..count);
            state = state.apply_move(legal[pick]);
            depth += 1;
        }

        let mut payoffs = [0.0f32; 4];
        if state.is_terminal && state.winner >= 0 {
            payoffs[state.winner as usize] = 1.0;
        } else {
            // Heuristic evaluation at cutoff
            let mut scores = [0.0f32; 4];
            let mut sum = 0.0;
            for p in 0..NUM_PLAYERS {
                if state.active_players[p] {
                    let sc = score_position(p as u8, &state).max(0.0) + 1.0;
                    scores[p] = sc;
                    sum += sc;
                }
            }
            if sum > 0.0 {
                for p in 0..NUM_PLAYERS {
                    payoffs[p] = scores[p] / sum;
                }
            }
        }

        payoffs
    }
}

impl LudoBot for MaxNMCTSBot {
    fn name(&self) -> &'static str {
        "MaxNMCTS"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return 0;
        }
        if count == 1 {
            return legal[0];
        }

        let mut arena: Vec<MctsNode> = Vec::with_capacity(MAX_TREE_NODES);
        let mut root = MctsNode::new(*state);

        // Seed root priors using Expectimax evaluation
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
        root.expanded = true;
        arena.push(root);

        let mut rng = rand::thread_rng();

        // Run MCTS simulations
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

            let (leaf_payoffs, new_child_idx) = if after_action.is_terminal {
                let mut payoffs = [0.0f32; 4];
                if after_action.winner >= 0 {
                    payoffs[after_action.winner as usize] = 1.0;
                }
                (payoffs, usize::MAX)
            } else if arena.len() < MAX_TREE_NODES {
                // Sample next dice roll
                let d = rng.gen_range(1..=6);
                let mut child_state = after_action;
                child_state.set_dice(d);

                let mut child_node = MctsNode::new(child_state);
                let payoffs = self.rollout(child_state, 60);

                child_node.expanded = true;
                let c_idx = arena.len();
                arena.push(child_node);
                (payoffs, c_idx)
            } else {
                // Arena full: rollout without expanding
                let payoffs = self.rollout(after_action, 60);
                (payoffs, usize::MAX)
            };

            if new_child_idx != usize::MAX {
                arena[last_node_idx].children[last_action_slot] = new_child_idx;
            }

            // 3. Max^n Vector Backup along the path
            for &(n_idx, a_slot) in &path {
                arena[n_idx].visit_counts[a_slot] += 1;
                for p in 0..NUM_PLAYERS {
                    arena[n_idx].total_payoffs[a_slot][p] += leaf_payoffs[p];
                }
            }
        }

        // Return action with most visits at root
        let root_node = &arena[0];
        let mut best_action_idx = 0;
        let mut max_visits = 0;

        for i in 0..root_node.legal_count as usize {
            if root_node.visit_counts[i] > max_visits {
                max_visits = root_node.visit_counts[i];
                best_action_idx = i;
            }
        }

        root_node.legal_actions[best_action_idx]
    }
}
