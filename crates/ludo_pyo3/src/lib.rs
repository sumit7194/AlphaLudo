use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray4};

pub mod generator;
use generator::{generate_alphazero_2p_batch_py, generate_winner_batch_py, select_mcts_move_2p_py};

#[pyfunction]
#[pyo3(signature = (min_states, bot_pool=None))]
fn generate_winner_batch<'py>(
    py: Python<'py>,
    min_states: usize,
    bot_pool: Option<Vec<String>>,
) -> PyResult<(
    Bound<'py, PyArray4<i8>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
)> {
    generate_winner_batch_py(py, min_states, bot_pool.unwrap_or_default())
}

#[pyfunction]
#[pyo3(signature = (min_states, num_simulations=3000, temperature_cutoff=20, use_heuristic_prior=false))]
fn generate_alphazero_2p_batch<'py>(
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
    generate_alphazero_2p_batch_py(
        py,
        min_states,
        num_simulations,
        temperature_cutoff,
        use_heuristic_prior,
    )
}

#[pyfunction]
#[pyo3(signature = (positions, current_player, dice_roll, num_simulations=500))]
fn select_mcts_move_2p<'py>(
    positions: &Bound<'py, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
    num_simulations: usize,
) -> PyResult<u8> {
    select_mcts_move_2p_py(positions, current_player, dice_roll, num_simulations)
}

#[pymodule]
fn alphaludo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_winner_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_alphazero_2p_batch, m)?)?;
    m.add_function(wrap_pyfunction!(select_mcts_move_2p, m)?)?;
    Ok(())
}

