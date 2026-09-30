//! Multi-threaded game simulation and training data batch generation.

use ludo_core::{
    cell_to_index, encode_frame_4p_into, encode_state_v17_into, position_to_cell_in_pov,
    GameState, BOARD_SIZE, NUM_BOARD_CELLS, V17_CHANNELS, V17_FRAME_SIZE,
};
use ludo_search::{
    AggressiveExpectimaxBot, Depth2ExpectimaxBot, ExpectimaxBot, LudoBot, MaxNMCTSBot,
    TwoPlayerMCTSBot,
};
use numpy::{PyArray1, PyArray2, PyArray4, PyArrayMethods};
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;


pub struct StepRecord {
    pub frame: [i8; BOARD_SIZE * BOARD_SIZE * 5],
    pub legal_mask: [f32; NUM_BOARD_CELLS],
    pub target_cell: i64,
    pub standings: [f32; 4],
    pub mover: u8,
}

pub fn create_bot(bot_type: &str, player_id: u8) -> Box<dyn LudoBot> {
    match bot_type {
        "MCTS" | "MaxNMCTS" => Box::new(MaxNMCTSBot::new(Some(player_id), 40)),
        "Depth2" | "Depth2Expectimax" => Box::new(Depth2ExpectimaxBot::new(Some(player_id))),
        "Aggressive" | "AggressiveExpectimax" => Box::new(AggressiveExpectimaxBot::new(Some(player_id))),
        _ => Box::new(ExpectimaxBot::new(Some(player_id))),
    }
}

/// Plays a single 4-player game between 4 bots, recording decisions.
pub fn play_game_record(bot_types: [&str; 4]) -> (Vec<StepRecord>, i8) {
    let mut rng = rand::thread_rng();
    let mut state = GameState::new_4p();
    let bots: [Box<dyn LudoBot>; 4] = [
        create_bot(bot_types[0], 0),
        create_bot(bot_types[1], 1),
        create_bot(bot_types[2], 2),
        create_bot(bot_types[3], 3),
    ];

    let mut trajectory = Vec::with_capacity(250);
    let mut moves_count = 0;

    while !state.is_terminal && moves_count < 800 {
        let cp = state.current_player;
        let d = rng.gen_range(1..=6);
        state.set_dice(d);

        let (legal, count) = state.legal_moves();
        if count == 0 {
            state.pass_turn();
            continue;
        }

        // Encode state in mover's POV
        let mut frame = [-1i8; BOARD_SIZE * BOARD_SIZE * 5];
        encode_frame_4p_into(&state, Some(cp), &mut frame);

        // Compute 225 legal source-cell mask
        let mut legal_mask = [0.0f32; NUM_BOARD_CELLS];
        for &t in &legal[..count] {
            let pos = state.positions[cp as usize][t as usize];
            let (r, c) = position_to_cell_in_pov(pos, cp, cp, t as usize);
            let idx = cell_to_index(r, c);
            legal_mask[idx] = 1.0;
        }

        // Normalized 4-way standings (Me, Next, Opp, Prev)
        let standings = state.normalized_standings(cp);

        // Bot selects move
        let action = bots[cp as usize].select_move(&state);

        // Target cell of the chosen token
        let chosen_pos = state.positions[cp as usize][action as usize];
        let (ch_r, ch_c) = position_to_cell_in_pov(chosen_pos, cp, cp, action as usize);
        let target_cell = cell_to_index(ch_r, ch_c) as i64;

        trajectory.push(StepRecord {
            frame,
            legal_mask,
            target_cell,
            standings,
            mover: cp,
        });

        state = state.apply_move(action);
        moves_count += 1;
    }

    let winner = if state.is_terminal { state.winner } else { -1 };
    (trajectory, winner)
}

/// Generates a batch of winner-only transitions.
/// Returns: (frames, legal_masks, target_cells, rel_winners, standings)
pub fn generate_winner_batch_py<'py>(
    py: Python<'py>,
    min_states: usize,
    bot_pool: Vec<String>,
) -> PyResult<(
    Bound<'py, PyArray4<i8>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let mut collected_frames: Vec<[i8; BOARD_SIZE * BOARD_SIZE * 5]> = Vec::with_capacity(min_states + 200);
    let mut collected_masks: Vec<[f32; NUM_BOARD_CELLS]> = Vec::with_capacity(min_states + 200);
    let mut collected_targets: Vec<i64> = Vec::with_capacity(min_states + 200);
    let mut collected_rel_winners: Vec<i64> = Vec::with_capacity(min_states + 200);
    let mut collected_standings: Vec<[f32; 4]> = Vec::with_capacity(min_states + 200);

    let default_pool = vec!["MCTS".to_string(), "Depth2".to_string()];
    let pool = if bot_pool.is_empty() { &default_pool } else { &bot_pool };
    let mut rng = rand::thread_rng();

    while collected_frames.len() < min_states {
        // Randomly assign bots to 4 seats
        let b0 = &pool[rng.gen_range(0..pool.len())];
        let b1 = &pool[rng.gen_range(0..pool.len())];
        let b2 = &pool[rng.gen_range(0..pool.len())];
        let b3 = &pool[rng.gen_range(0..pool.len())];

        let (traj, winner) = play_game_record([b0, b1, b2, b3]);
        if winner < 0 {
            continue;
        }

        // Winner-only extraction: ONLY retain steps made by the winner!
        for step in traj {
            if step.mover as i8 == winner {
                collected_frames.push(step.frame);
                collected_masks.push(step.legal_mask);
                collected_targets.push(step.target_cell);
                // Winner relative seat from mover's POV is always 0
                collected_rel_winners.push(0);
                collected_standings.push(step.standings);
            }
        }
    }

    let n = collected_frames.len();

    // Flatten frames into a contiguous vector
    let mut flat_frames: Vec<i8> = Vec::with_capacity(n * BOARD_SIZE * BOARD_SIZE * 5);
    for f in &collected_frames {
        flat_frames.extend_from_slice(f);
    }

    let mut flat_masks: Vec<f32> = Vec::with_capacity(n * NUM_BOARD_CELLS);
    for m in &collected_masks {
        flat_masks.extend_from_slice(m);
    }

    let mut flat_standings: Vec<f32> = Vec::with_capacity(n * 4);
    for s in &collected_standings {
        flat_standings.extend_from_slice(s);
    }

    let py_frames = PyArray1::from_vec_bound(py, flat_frames)
        .reshape([n, BOARD_SIZE, BOARD_SIZE, 5])?;
    let py_masks = PyArray1::from_vec_bound(py, flat_masks)
        .reshape([n, NUM_BOARD_CELLS])?;
    let py_targets = PyArray1::from_vec_bound(py, collected_targets);
    let py_winners = PyArray1::from_vec_bound(py, collected_rel_winners);
    let py_standings = PyArray1::from_vec_bound(py, flat_standings)
        .reshape([n, 4])?;

    Ok((py_frames, py_masks, py_targets, py_winners, py_standings))
}

#[derive(Clone)]
pub struct AlphaZero2PStep {
    pub frame: [f32; V17_FRAME_SIZE],
    pub legal_mask: [f32; 4],
    pub pi: [f32; 4],
    pub action: i64,
    pub mover: u8,
}

/// Plays a single 2-player AlphaZero self-play game using TwoPlayerMCTSBot.
/// Returns all steps with their final zero-sum terminal outcome z in {+1.0, -1.0}.
pub fn play_alphazero_2p_game(
    num_simulations: usize,
    temperature_cutoff: usize,
    use_pure_terminal: bool,
    use_heuristic_prior: bool,
) -> Vec<(AlphaZero2PStep, f32)> {
    let mut rng = rand::thread_rng();
    let mut state = GameState::new_2p();
    let mut bot = TwoPlayerMCTSBot::new(None, num_simulations);
    bot.use_pure_terminal = use_pure_terminal;
    bot.use_heuristic_prior = use_heuristic_prior;
    bot.add_dirichlet = true; // AlphaZero root exploration noise

    let mut raw_steps: Vec<AlphaZero2PStep> = Vec::with_capacity(200);
    let mut move_count = 0;

    while !state.is_terminal && move_count < 600 {
        let cp = state.current_player;
        let d = rng.gen_range(1..=6);
        state.set_dice(d);

        let (legal, count) = state.legal_moves();
        if count == 0 {
            state.pass_turn();
            continue;
        }

        let mut legal_mask = [0.0f32; 4];
        for &t in &legal[..count] {
            legal_mask[t as usize] = 1.0;
        }

        let mut frame = [0.0f32; V17_FRAME_SIZE];
        encode_state_v17_into(&state, &mut frame);

        let temp = if move_count < temperature_cutoff { 1.0 } else { 0.0 };
        let (action, pi) = bot.search_with_temp(&state, temp);

        raw_steps.push(AlphaZero2PStep {
            frame,
            legal_mask,
            pi,
            action: action as i64,
            mover: cp,
        });

        state = state.apply_move(action);
        move_count += 1;
    }

    let winner = if state.is_terminal { state.winner } else { -1 };

    raw_steps
        .into_iter()
        .map(|step| {
            let target_value = if winner >= 0 {
                if step.mover as i8 == winner { 1.0f32 } else { -1.0f32 }
            } else {
                0.0f32
            };
            (step, target_value)
        })
        .collect()
}

/// Multi-threaded batch generation for AlphaZero 2-Player Self-Play.
/// Returns: (frames, legal_masks, target_pis, target_values, actions)
pub fn generate_alphazero_2p_batch_py<'py>(
    py: Python<'py>,
    min_states: usize,
    num_simulations: usize,
    temperature_cutoff: usize,
    use_heuristic_prior: bool,
) -> PyResult<(
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let mut collected: Vec<(AlphaZero2PStep, f32)> = Vec::with_capacity(min_states + 250);

    while collected.len() < min_states {
        let chunk_size = rayon::current_num_threads().max(1);
        let new_games: Vec<Vec<(AlphaZero2PStep, f32)>> = (0..chunk_size)
            .into_par_iter()
            .map(|_| play_alphazero_2p_game(num_simulations, temperature_cutoff, true, use_heuristic_prior))
            .collect();

        for game in new_games {
            collected.extend(game);
        }
    }

    let n = collected.len();

    let mut flat_frames: Vec<f32> = Vec::with_capacity(n * V17_FRAME_SIZE);
    let mut flat_masks: Vec<f32> = Vec::with_capacity(n * 4);
    let mut flat_pis: Vec<f32> = Vec::with_capacity(n * 4);
    let mut values: Vec<f32> = Vec::with_capacity(n);
    let mut actions: Vec<i64> = Vec::with_capacity(n);

    for (step, val) in &collected {
        flat_frames.extend_from_slice(&step.frame);
        flat_masks.extend_from_slice(&step.legal_mask);
        flat_pis.extend_from_slice(&step.pi);
        values.push(*val);
        actions.push(step.action);
    }

    let py_frames = PyArray1::from_vec_bound(py, flat_frames)
        .reshape([n, V17_CHANNELS, BOARD_SIZE, BOARD_SIZE])?;
    let py_masks = PyArray1::from_vec_bound(py, flat_masks)
        .reshape([n, 4])?;
    let py_pis = PyArray1::from_vec_bound(py, flat_pis)
        .reshape([n, 4])?;
    let py_values = PyArray1::from_vec_bound(py, values);
    let py_actions = PyArray1::from_vec_bound(py, actions);

    Ok((py_frames, py_masks, py_pis, py_values, py_actions))
}

/// Single-state move selection using Expecti-MCTS in native Rust for evaluation.
pub fn select_mcts_move_2p_py(
    positions: &Bound<'_, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
    num_simulations: usize,
) -> PyResult<u8> {
    let pos_readonly = positions.readonly();
    let pos_slice = pos_readonly.as_slice()?;
    if pos_slice.len() != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err("positions array must have exactly 16 elements (4x4)"));
    }

    let mut state = GameState::new_2p();
    for p in 0..4 {
        for t in 0..4 {
            state.positions[p][t] = pos_slice[p * 4 + t];
        }
    }
    state.current_player = current_player;
    state.current_dice_roll = dice_roll;
    for p in 0..4 {
        state.scores[p] = state.positions[p].iter().filter(|&&pos| pos == 99).count() as u8;
    }

    let (legal, count) = state.legal_moves();
    if count == 0 {
        return Ok(0);
    }
    if count == 1 {
        return Ok(legal[0]);
    }

    let mut bot = TwoPlayerMCTSBot::new(Some(current_player), num_simulations);
    bot.add_dirichlet = false;
    bot.use_pure_terminal = false;
    let (action, _) = bot.search_with_temp(&state, 0.0);
    Ok(action)
}

