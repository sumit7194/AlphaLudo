use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray4};

pub mod generator;
use generator::{
    generate_alphazero_2p_batch_py, generate_winner_batch_py, select_bot_move_2p_py, select_mcts_move_2p_py,
};

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
#[pyo3(signature = (positions, current_player, dice_roll, num_simulations=500, consecutive_sixes=0))]
fn select_mcts_move_2p<'py>(
    positions: &Bound<'py, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
    num_simulations: usize,
    consecutive_sixes: u8,
) -> PyResult<u8> {
    select_mcts_move_2p_py(positions, current_player, dice_roll, num_simulations, consecutive_sixes)
}

#[pyfunction]
#[pyo3(signature = (bot, positions, current_player, dice_roll, consecutive_sixes=0, num_simulations=3000))]
fn select_bot_move_2p<'py>(
    bot: &str,
    positions: &Bound<'py, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
    consecutive_sixes: u8,
    num_simulations: usize,
) -> PyResult<u8> {
    select_bot_move_2p_py(bot, positions, current_player, dice_roll, consecutive_sixes, num_simulations)
}

/// Bit-exactness check helper: Rust V15 2P frame for one state.
#[pyfunction]
fn encode_v15_2p<'py>(
    py: Python<'py>,
    positions: &Bound<'py, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    use numpy::PyArrayMethods;
    let ro = positions.readonly();
    let s = ro.as_slice()?;
    let mut st = ludo_core::GameState::new_2p();
    for p in 0..4 {
        for t in 0..4 {
            st.positions[p][t] = s[p * 4 + t];
        }
    }
    st.current_player = current_player;
    st.current_dice_roll = dice_roll;
    let mut out = vec![0i8; ludo_core::V15_FRAME_SIZE];
    ludo_core::encode_v15_2p_into(&st, current_player, &mut out);
    Ok(PyArray1::from_vec_bound(py, out))
}

fn seat_from_dict(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<ludo_search::SeatConfig> {
    use ludo_search::{NetMctsConfig, SeatConfig, SeatKind};
    fn get<'py, T: pyo3::FromPyObject<'py>>(d: &Bound<'py, pyo3::types::PyDict>, k: &str, default: T) -> PyResult<T> {
        match d.get_item(k)? {
            Some(v) => v.extract(),
            None => Ok(default),
        }
    }
    let kind: String = get(d, "kind", "mcts".to_string())?;
    let def = NetMctsConfig::default();
    let kind = match kind.as_str() {
        "mcts" => SeatKind::Mcts(NetMctsConfig {
            sims: get(d, "sims", def.sims)?,
            c_puct: get(d, "c_puct", def.c_puct)?,
            fpu_reduction: get(d, "fpu_reduction", def.fpu_reduction)?,
            leaves_per_round: get(d, "leaves_per_round", def.leaves_per_round)?,
            dirichlet_eps: get(d, "dirichlet_eps", def.dirichlet_eps)?,
            reuse_tree: get(d, "reuse_tree", def.reuse_tree)?,
        }),
        "expecti" => {
            let de = ludo_search::NetExpectiConfig::default();
            SeatKind::Expecti(ludo_search::NetExpectiConfig {
                max_evals: get(d, "max_evals", de.max_evals)?,
                c_puct: get(d, "c_puct", de.c_puct)?,
                fpu_reduction: get(d, "fpu_reduction", de.fpu_reduction)?,
                expansions_per_round: get(d, "expansions_per_round", de.expansions_per_round)?,
                dirichlet_eps: get(d, "dirichlet_eps", de.dirichlet_eps)?,
                pi_temp: get(d, "pi_temp", de.pi_temp)?,
                pick_by_visits: get(d, "pick_by_visits", de.pick_by_visits)?,
            })
        }
        "greedy" => SeatKind::Greedy,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!("seat kind {other:?}: mcts | expecti | greedy"))),
    };
    Ok(SeatConfig {
        kind,
        model: get(d, "model", 0u8)?,
        record: get(d, "record", false)?,
        explore_moves: get(d, "explore_moves", 0u32)?,
    })
}

/// Batched game farm for network-guided MCTS. Python answers evaluation
/// requests with a neural network; Rust owns games, dice, trees and records.
#[pyclass]
struct NetFarm {
    inner: ludo_search::NetFarm,
}

#[pymethods]
impl NetFarm {
    #[new]
    #[pyo3(signature = (seat_a, seat_b, n_games, concurrency=128, seed=0, max_moves=600))]
    fn new(
        seat_a: &Bound<'_, pyo3::types::PyDict>,
        seat_b: &Bound<'_, pyo3::types::PyDict>,
        n_games: u32,
        concurrency: usize,
        seed: u64,
        max_moves: u32,
    ) -> PyResult<Self> {
        let mut inner = ludo_search::NetFarm::new(seat_from_dict(seat_a)?, seat_from_dict(seat_b)?, n_games, concurrency, seed);
        inner.max_moves = max_moves;
        Ok(NetFarm { inner })
    }

    /// Advance all games to their next network call; returns the batch size (0 = done).
    fn collect(&mut self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.inner.collect())
    }

    /// (x int8 (n, 675), legal_mask f32 (n, 225), model_id u8 (n,)) for the current batch.
    fn batch<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u8>>)> {
        use numpy::PyArrayMethods;
        let n = self.inner.req_model.len();
        let x = PyArray1::from_slice_bound(py, &self.inner.req_x).reshape([n, ludo_core::V15_FRAME_SIZE])?;
        let m = PyArray1::from_slice_bound(py, &self.inner.req_mask).reshape([n, ludo_core::NUM_BOARD_CELLS])?;
        let id = PyArray1::from_slice_bound(py, &self.inner.req_model);
        Ok((x, m, id))
    }

    /// Answers the current batch: policy (n, 225) probabilities, p_win (n,).
    fn submit(&mut self, policy: numpy::PyReadonlyArray2<'_, f32>, p_win: numpy::PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let pol = policy.as_slice()?;
        let v = p_win.as_slice()?;
        if v.len() != self.inner.req_model.len() || pol.len() != v.len() * ludo_core::NUM_BOARD_CELLS {
            return Err(pyo3::exceptions::PyValueError::new_err("submit: shape does not match the current batch"));
        }
        self.inner.submit(pol, v);
        Ok(())
    }

    /// Finished games: list of (game_id, a_seat, winner_seat or -1, moves).
    fn results(&self) -> Vec<(u32, u8, i8, u32)> {
        self.inner.results.iter().map(|r| (r.id, r.a_seat, r.winner, r.moves)).collect()
    }

    fn reused_visits(&self) -> u64 {
        self.inner.reused_visits
    }

    fn games_finished(&self) -> usize {
        self.inner.games_finished()
    }

    /// Drains recorded decisions (format v2 records, no header) as bytes.
    fn take_records<'py>(&mut self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let recs = std::mem::take(&mut self.inner.records);
        pyo3::types::PyBytes::new_bound(py, ludo_search::records::records_as_bytes(&recs))
    }
}

fn unified_state(
    positions: &Bound<'_, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
    active: [bool; 4],
    consecutive_sixes: [u8; 4],
) -> PyResult<ludo_core::GameState> {
    use numpy::PyArrayMethods;
    let ro = positions.readonly();
    let s = ro.as_slice()?;
    if s.len() != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err("positions must be 4x4"));
    }
    let mut st = ludo_core::GameState::new_4p();
    st.active_players = active;
    for p in 0..4 {
        for t in 0..4 {
            st.positions[p][t] = s[p * 4 + t];
        }
        st.scores[p] = st.positions[p].iter().filter(|&&x| x == 99).count() as u8;
    }
    st.current_player = current_player;
    st.current_dice_roll = dice_roll;
    st.consecutive_sixes = consecutive_sixes;
    Ok(st)
}

/// Unified (V17) frame (15*15*5 int8) and legal mask (225 f32) for one state, mover's POV.
#[pyfunction]
#[pyo3(signature = (positions, current_player, dice_roll, active, consecutive_sixes=[0, 0, 0, 0]))]
fn encode_unified<'py>(
    py: Python<'py>,
    positions: &Bound<'py, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
    active: [bool; 4],
    consecutive_sixes: [u8; 4],
) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<f32>>)> {
    let st = unified_state(positions, current_player, dice_roll, active, consecutive_sixes)?;
    let mut x = vec![0i8; ludo_core::UNI_FRAME_SIZE];
    ludo_core::encode_unified_into(&st, current_player, &mut x);
    let mut m = vec![0f32; ludo_core::NUM_BOARD_CELLS];
    ludo_core::uni_legal_mask_into(&st, &mut m);
    Ok((PyArray1::from_vec_bound(py, x), PyArray1::from_vec_bound(py, m)))
}

/// Native bot move for an arbitrary 2P/4P state ("heuristic", "expert", "expectimax",
/// "depth2", "mcts" (2P only)). `noise=False` makes the scripted bots deterministic.
#[pyfunction]
#[pyo3(signature = (bot, positions, current_player, dice_roll, active, consecutive_sixes=[0, 0, 0, 0], noise=true, num_simulations=3000))]
fn select_bot_move(
    bot: &str,
    positions: &Bound<'_, PyArray2<i8>>,
    current_player: u8,
    dice_roll: u8,
    active: [bool; 4],
    consecutive_sixes: [u8; 4],
    noise: bool,
    num_simulations: usize,
) -> PyResult<u8> {
    use ludo_search::LudoBot;
    let st = unified_state(positions, current_player, dice_roll, active, consecutive_sixes)?;
    let (legal, count) = st.legal_moves();
    if count == 0 {
        return Ok(0);
    }
    let seat = Some(current_player);
    Ok(match bot.to_ascii_lowercase().as_str() {
        "heuristic" | "expert" => {
            let kind = if bot.eq_ignore_ascii_case("expert") { ludo_search::ScriptedKind::Expert } else { ludo_search::ScriptedKind::Heuristic };
            let mut b = ludo_search::ScriptedBot::new(kind, seat);
            b.noise = noise;
            b.select_move(&st)
        }
        "expectimax" => ludo_search::ExpectimaxBot::new(seat).select_move(&st),
        "depth2" => ludo_search::Depth2ExpectimaxBot::new(seat).select_move(&st),
        "mcts" => {
            if count == 1 { legal[0] } else { ludo_search::TwoPlayerMCTSBot::new(seat, num_simulations).search_with_temp(&st, 0.0).0 }
        }
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown bot {other:?}"))),
    })
}

/// Board cell index (mover frame) of `owner`'s token at track position 0..=55 seen by `pov`.
#[pyfunction]
fn unified_track_cell(pos: i8, owner: u8, pov: u8) -> PyResult<usize> {
    if !(0..=55).contains(&pos) || owner > 3 || pov > 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("pos must be 0..=55, seats 0..3"));
    }
    Ok(ludo_core::uni_track_cell(pos, owner, pov))
}

/// Streaming SL data: MCTS-vs-MCTS self-play on background Rust threads (no files).
#[pyclass]
struct SlStream {
    inner: Option<ludo_search::sl_stream::SlStream>,
}

#[pymethods]
impl SlStream {
    /// frac4: target share of 4P positions (0 = the original 2P stream); teacher4: a
    /// bot spec dict for the 4P teacher in every seat ({"bot": "round", ...} or
    /// {"bot": "maxn", ...}); required when frac4 > 0.
    #[new]
    #[pyo3(signature = (threads, sims=3000, keep_prob=0.25, capacity=200_000, max_moves=600, seed=0,
                        frac4=0.0, teacher4=None, max_moves4=1500))]
    fn new(threads: usize, sims: usize, keep_prob: f64, capacity: usize, max_moves: usize, seed: u64,
           frac4: f64, teacher4: Option<Bound<'_, pyo3::types::PyDict>>, max_moves4: usize) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&keep_prob) || keep_prob == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("keep_prob must be in (0, 1]"));
        }
        let teacher4 = match teacher4 {
            None => None,
            Some(d) => match parse_player(&d)? {
                ludo_search::farm::PlayerSpec::Bot(k @ (ludo_search::farm::BotKind::Round(_)
                | ludo_search::farm::BotKind::MaxN(_)
                | ludo_search::farm::BotKind::Mcts(_))) => Some(k),
                _ => return Err(pyo3::exceptions::PyValueError::new_err("teacher4 must be a round / maxn / mcts bot")),
            },
        };
        if frac4 > 0.0 && teacher4.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err("frac4 > 0 needs teacher4"));
        }
        let cfg = ludo_search::sl_stream::SlStreamConfig {
            threads, sims, keep_prob, capacity, max_moves, seed, frac4, teacher4, max_moves4,
        };
        Ok(SlStream { inner: Some(ludo_search::sl_stream::SlStream::start(cfg)) })
    }

    /// Up to n records as a dict of numpy arrays: x (k,1125) i8, cells (k,4) u8,
    /// pi (k,4) f32, n_cells (k,) u8, action (k,) u8, winner_rel (k,) u8 (255 = unfinished),
    /// legal_tokens (k,) u8.
    #[pyo3(signature = (n, timeout_ms=1000))]
    fn take<'py>(&self, py: Python<'py>, n: usize, timeout_ms: u64) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use numpy::PyArrayMethods;
        let inner = self.inner.as_ref().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("stream stopped"))?;
        let recs = py.allow_threads(|| inner.take(n, timeout_ms));
        let k = recs.len();
        let mut x = Vec::with_capacity(k * ludo_core::UNI_FRAME_SIZE);
        let (mut cells, mut pi) = (Vec::with_capacity(k * 4), Vec::with_capacity(k * 4));
        let (mut nc, mut act, mut win, mut lt) = (Vec::with_capacity(k), Vec::with_capacity(k), Vec::with_capacity(k), Vec::with_capacity(k));
        let mut np_: Vec<u8> = Vec::with_capacity(k);
        for r in &recs {
            x.extend_from_slice(&r.x[..]);
            cells.extend_from_slice(&r.cells);
            pi.extend_from_slice(&r.pi);
            nc.push(r.n_cells);
            act.push(r.action);
            win.push(r.winner_rel);
            lt.push(r.legal_tokens);
            np_.push(r.n_players);
        }
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("x", PyArray1::from_vec_bound(py, x).reshape([k, ludo_core::UNI_FRAME_SIZE])?)?;
        d.set_item("cells", PyArray1::from_vec_bound(py, cells).reshape([k, 4])?)?;
        d.set_item("pi", PyArray1::from_vec_bound(py, pi).reshape([k, 4])?)?;
        d.set_item("n_cells", PyArray1::from_vec_bound(py, nc))?;
        d.set_item("action", PyArray1::from_vec_bound(py, act))?;
        d.set_item("winner_rel", PyArray1::from_vec_bound(py, win))?;
        d.set_item("legal_tokens", PyArray1::from_vec_bound(py, lt))?;
        d.set_item("n_players", PyArray1::from_vec_bound(py, np_))?;
        Ok(d)
    }

    fn stats(&self) -> (u64, u64, usize) {
        match &self.inner {
            Some(s) => (s.games(), s.positions(), s.queued()),
            None => (0, 0, 0),
        }
    }

    /// (4P games, 4P positions) generated so far.
    fn stats4(&self) -> (u64, u64) {
        match &self.inner {
            Some(s) => (s.games4(), s.positions4()),
            None => (0, 0),
        }
    }

    /// Changes the target share of 4P positions (from each worker's next game).
    fn set_frac4(&self, f: f64) -> PyResult<()> {
        let s = self.inner.as_ref().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("stream stopped"))?;
        s.set_frac4(f);
        Ok(())
    }

    fn stop(&mut self, py: Python<'_>) {
        if let Some(mut s) = self.inner.take() {
            py.allow_threads(move || s.stop());
        }
    }
}

fn parse_player(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<ludo_search::farm::PlayerSpec> {
    use ludo_search::farm::{BotKind, PlayerSpec};
    if let Some(m) = d.get_item("net")? {
        let temperature: f32 = match d.get_item("temp")? { Some(v) => v.extract()?, None => 1.0 };
        let record: bool = match d.get_item("record")? { Some(v) => v.extract()?, None => false };
        return Ok(PlayerSpec::Net { model: m.extract()?, temperature, record });
    }
    let bot: String = d
        .get_item("bot")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("player needs 'net' or 'bot'"))?
        .extract()?;
    let kind = match bot.to_ascii_lowercase().as_str() {
        "random" => BotKind::Random,
        "heuristic" => BotKind::Heuristic,
        "expert" => BotKind::Expert,
        "expectimax" => BotKind::Expectimax,
        "depth2" => BotKind::Depth2,
        "mcts" => BotKind::Mcts(match d.get_item("sims")? { Some(v) => v.extract()?, None => 3000 }),
        "maxn" => BotKind::MaxN(parse_maxn(d)?),
        "round" => BotKind::Round(parse_round(d)?),
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown bot {other:?}"))),
    };
    Ok(PlayerSpec::Bot(kind))
}

/// {"bot": "maxn", "sims": 3000, "c": 0.5, "smooth": 0.15, "depth": 10,
///  "rollout": "score" | "share", "leaf": "legacy" | "e4", "weights": [N_E4 floats],
///  "prior": {round bot spec}}   (prior omitted = the legacy 1-ply Expectimax prior)
/// Missing keys default to the legacy engine; leaf "e4" without weights uses the
/// compiled-in fitted weights.
fn parse_maxn(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<ludo_search::MaxNConfig> {
    use ludo_search::{LeafEval4, MaxNConfig, Rollout4, E4_WEIGHTS, N_E4};
    fn opt<'py, T: FromPyObject<'py>>(d: &Bound<'py, pyo3::types::PyDict>, k: &str, def: T) -> PyResult<T> {
        Ok(match d.get_item(k)? { Some(v) => v.extract()?, None => def })
    }
    let mut c = MaxNConfig::legacy(opt(d, "sims", 3000u32)?);
    c.c_puct = opt(d, "c", c.c_puct)?;
    c.prior_smooth = opt(d, "smooth", c.prior_smooth)?;
    c.rollout_depth = opt(d, "depth", c.rollout_depth)?;
    c.rollout = match opt(d, "rollout", "score".to_string())?.as_str() {
        "score" => Rollout4::ScorePosition,
        "share" => Rollout4::OwnShare,
        o => return Err(pyo3::exceptions::PyValueError::new_err(format!("rollout {o:?}: score | share"))),
    };
    if let Some(pd) = d.get_item("prior")? {
        let pd = pd.downcast::<pyo3::types::PyDict>()?;
        c.prior = ludo_search::Prior4::Round(parse_round(pd)?);
    }
    c.leaf = match opt(d, "leaf", "legacy".to_string())?.as_str() {
        "legacy" => LeafEval4::Legacy,
        "e4" => {
            let w: Vec<f32> = opt(d, "weights", E4_WEIGHTS.to_vec())?;
            if w.len() != N_E4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("weights need {N_E4} values")));
            }
            let mut a = [0f32; N_E4];
            a.copy_from_slice(&w);
            LeafEval4::E4(a)
        }
        o => return Err(pyo3::exceptions::PyValueError::new_err(format!("leaf {o:?}: legacy | e4"))),
    };
    Ok(c)
}

/// {"bot": "round", "leaf": "e4" | "score" | "lin", "weights": [...], "beta": 0.5, "rolls": 3}
fn parse_round(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<ludo_search::RoundConfig> {
    use ludo_search::{RoundConfig, RoundLeaf, E4_WEIGHTS, N_E4};
    let rolls: u8 = match d.get_item("rolls")? { Some(v) => v.extract()?, None => 3 };
    let leaf: String = match d.get_item("leaf")? { Some(v) => v.extract()?, None => "e4".into() };
    let leaf = match leaf.as_str() {
        "score" => RoundLeaf::Score,
        "lin" => {
            let w: Vec<f32> = match d.get_item("weights")? { Some(v) => v.extract()?, None => E4_WEIGHTS.to_vec() };
            if w.len() != N_E4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("weights need {N_E4} values")));
            }
            let mut a = [0f32; N_E4];
            a.copy_from_slice(&w);
            if let Some(uo) = d.get_item("opp")? {
                let u: Vec<f32> = uo.extract()?;
                if u.len() != N_E4 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("opp needs {N_E4} values")));
                }
                let mut b = [0f32; N_E4];
                b.copy_from_slice(&u);
                RoundLeaf::Lin2 { w: a, u: b }
            } else {
                let beta: f32 = match d.get_item("beta")? { Some(v) => v.extract()?, None => 0.5 };
                RoundLeaf::Lin { w: a, beta }
            }
        }
        "e4" => {
            let w: Vec<f32> = match d.get_item("weights")? { Some(v) => v.extract()?, None => E4_WEIGHTS.to_vec() };
            if w.len() != N_E4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("weights need {N_E4} values")));
            }
            let mut a = [0f32; N_E4];
            a.copy_from_slice(&w);
            RoundLeaf::E4(a)
        }
        o => return Err(pyo3::exceptions::PyValueError::new_err(format!("leaf {o:?}: e4 | score"))),
    };
    Ok(RoundConfig { leaf, max_rolls: rolls })
}

/// 4P evaluator training data: bot-vs-bot games (rayon-parallel), pre-roll states
/// kept with probability keep_prob. Returns numpy arrays: feats (N,4,F), active
/// (N,4), legacy (N,4) legacy payoff shares, winner (N,), game (N,), t (N,).
#[pyfunction]
#[pyo3(signature = (seats, games, keep_prob=0.3, seed=0, max_moves=1500))]
fn eval4_dataset<'py>(
    py: Python<'py>,
    seats: Vec<Bound<'py, pyo3::types::PyDict>>,
    games: usize,
    keep_prob: f64,
    seed: u64,
    max_moves: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use ludo_search::farm::{BotKind, PlayerSpec};
    use ludo_search::N_E4;
    use numpy::PyArrayMethods;
    let mut kinds: Vec<BotKind> = Vec::new();
    for d in &seats {
        match parse_player(d)? {
            PlayerSpec::Bot(k) => kinds.push(k),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("eval4_dataset seats must be bots")),
        }
    }
    if kinds.len() != 2 && kinds.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("need 2 or 4 seats"));
    }
    let rows = py.allow_threads(|| ludo_search::eval4_data::generate(&kinds, games, keep_prob, max_moves, seed));
    let n = rows.len();
    let mut feats = Vec::with_capacity(n * 4 * N_E4);
    let mut active = Vec::with_capacity(n * 4);
    let mut legacy = Vec::with_capacity(n * 4);
    let (mut winner, mut game, mut t) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for r in &rows {
        for p in 0..4 {
            feats.extend_from_slice(&r.feats[p]);
            active.push(r.active[p]);
            legacy.push(r.legacy[p]);
        }
        winner.push(r.winner);
        game.push(r.game);
        t.push(r.t);
    }
    let out = pyo3::types::PyDict::new_bound(py);
    out.set_item("feats", numpy::PyArray1::from_vec_bound(py, feats).reshape([n, 4, N_E4])?)?;
    out.set_item("active", numpy::PyArray1::from_vec_bound(py, active).reshape([n, 4])?)?;
    out.set_item("legacy", numpy::PyArray1::from_vec_bound(py, legacy).reshape([n, 4])?)?;
    out.set_item("winner", numpy::PyArray1::from_vec_bound(py, winner))?;
    out.set_item("game", numpy::PyArray1::from_vec_bound(py, game))?;
    out.set_item("t", numpy::PyArray1::from_vec_bound(py, t))?;
    Ok(out)
}

/// (feature names, compiled-in weights) of the 4P evaluator.
#[pyfunction]
fn eval4_info() -> (Vec<String>, Vec<f32>) {
    (ludo_search::E4_NAMES.iter().map(|s| s.to_string()).collect(), ludo_search::E4_WEIGHTS.to_vec())
}

/// Network-guided Max^n search farm (2P/4P, V17 unified encoding). roles: 2 or 4
/// dicts, each {"search": model, "sims": 200, "c": 1.0, "fpu": 0.1, "leaves": 8},
/// {"greedy": model} or a bot spec ({"bot": ...}). Role i sits on a shuffled seat.
#[pyclass]
struct SearchFarm {
    inner: ludo_search::net_maxn::SearchFarm,
}

#[pymethods]
impl SearchFarm {
    #[new]
    #[pyo3(signature = (roles, n_models=1, games=100, concurrency=64, seed=0))]
    fn new(roles: Vec<Bound<'_, pyo3::types::PyDict>>, n_models: usize, games: u64, concurrency: usize, seed: u64) -> PyResult<Self> {
        use ludo_search::net_maxn::{NetMaxNConfig, SeatSpec};
        fn opt<'py, T: FromPyObject<'py>>(d: &Bound<'py, pyo3::types::PyDict>, k: &str, def: T) -> PyResult<T> {
            Ok(match d.get_item(k)? { Some(v) => v.extract()?, None => def })
        }
        if roles.len() != 2 && roles.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("need 2 or 4 roles"));
        }
        let mut specs = Vec::new();
        for d in &roles {
            if let Some(m) = d.get_item("search")? {
                let def = NetMaxNConfig::default();
                let cfg = NetMaxNConfig {
                    sims: opt(d, "sims", def.sims)?,
                    c_puct: opt(d, "c", def.c_puct)?,
                    fpu_reduction: opt(d, "fpu", def.fpu_reduction)?,
                    leaves_per_round: opt(d, "leaves", def.leaves_per_round)?,
                };
                specs.push(SeatSpec::Search { model: m.extract()?, cfg });
            } else if let Some(m) = d.get_item("greedy")? {
                specs.push(SeatSpec::Greedy { model: m.extract()? });
            } else {
                match parse_player(d)? {
                    ludo_search::farm::PlayerSpec::Bot(k) => specs.push(SeatSpec::Bot(k)),
                    _ => return Err(pyo3::exceptions::PyValueError::new_err("role: use search / greedy / bot")),
                }
            }
        }
        Ok(SearchFarm { inner: ludo_search::net_maxn::SearchFarm::new(specs, n_models, games, concurrency, seed) })
    }

    fn collect(&mut self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.inner.collect())
    }

    fn models_pending(&self) -> Vec<u8> {
        self.inner.models_pending()
    }

    /// (x int8 (n, 1125), legal_mask f32 (n, 225)) for one model's pending requests.
    fn batch<'py>(&self, py: Python<'py>, model: u8) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray2<f32>>)> {
        use numpy::PyArrayMethods;
        let n = self.inner.request_count(model);
        let xa = PyArray1::from_slice_bound(py, self.inner.request_x(model)).reshape([n, ludo_core::UNI_FRAME_SIZE])?;
        let ma = PyArray1::from_slice_bound(py, self.inner.request_mask(model)).reshape([n, ludo_core::NUM_BOARD_CELLS])?;
        Ok((xa, ma))
    }

    /// policy (n, 225) probabilities; value (n, 4) win probabilities over RELATIVE seats.
    fn submit(&mut self, model: u8, policy: numpy::PyReadonlyArray2<'_, f32>, value: numpy::PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        self.inner.submit(model, policy.as_slice()?, value.as_slice()?).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Finished games: (game_id, role_of_seat[4], winner seat or -1, moves).
    fn take_results(&mut self) -> Vec<(u64, [u8; 4], i8, u32)> {
        std::mem::take(&mut self.inner.results).into_iter().map(|r| (r.game_id, r.role_of_seat, r.winner, r.moves)).collect()
    }

    /// (network evaluations by search seats, by greedy seats, games in flight).
    fn stats(&self) -> (u64, u64, usize) {
        (self.inner.search_evals, self.inner.greedy_evals, self.inner.in_flight())
    }
}

/// Batched 2P/4P game farm for V17 RL / evaluation. See ludo_search::farm.
#[pyclass]
struct Farm {
    inner: ludo_search::farm::GameFarm,
}

#[pymethods]
impl Farm {
    /// encoders: one entry per model id, "unified" or "v15".
    #[new]
    #[pyo3(signature = (encoders, concurrency=256, seed=0, max_moves=1200))]
    fn new(encoders: Vec<String>, concurrency: usize, seed: u64, max_moves: u32) -> PyResult<Self> {
        let enc = encoders
            .iter()
            .map(|e| match e.as_str() {
                "unified" => Ok(ludo_search::farm::Encoder::Unified),
                "v15" => Ok(ludo_search::farm::Encoder::V15),
                o => Err(pyo3::exceptions::PyValueError::new_err(format!("encoder {o:?}: unified | v15"))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut inner = ludo_search::farm::GameFarm::new(enc, concurrency, seed);
        inner.max_moves = max_moves;
        Ok(Farm { inner })
    }

    /// matchups: list of {"players": [player, ...] (2 or 4), "weight": w}; player =
    /// {"net": model_id, "temp": T (0 = greedy), "record": bool} or {"bot": name, "sims": n}.
    fn set_matchups(&mut self, matchups: Vec<Bound<'_, pyo3::types::PyDict>>) -> PyResult<()> {
        let mut out = Vec::new();
        for m in &matchups {
            let players: Vec<Bound<'_, pyo3::types::PyDict>> = m
                .get_item("players")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("matchup needs 'players'"))?
                .extract()?;
            let weight: f64 = match m.get_item("weight")? { Some(v) => v.extract()?, None => 1.0 };
            let players = players.iter().map(parse_player).collect::<PyResult<Vec<_>>>()?;
            out.push(ludo_search::farm::Matchup { players, weight });
        }
        self.inner.set_matchups(out).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn set_game_limit(&mut self, n: u64) {
        self.inner.set_game_limit(n);
    }

    fn collect(&mut self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.inner.collect())
    }

    fn models_pending(&self) -> Vec<u8> {
        self.inner.models_pending()
    }

    /// (x int8 (n, frame), legal_mask f32 (n, 225)) for one model's pending requests.
    fn batch<'py>(&self, py: Python<'py>, model: u8) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray2<f32>>)> {
        use numpy::PyArrayMethods;
        let n = self.inner.request_count(model);
        let x = self.inner.request_x(model);
        let fx = if n > 0 { x.len() / n } else { 0 };
        let xa = PyArray1::from_slice_bound(py, x).reshape([n, fx])?;
        let ma = PyArray1::from_slice_bound(py, self.inner.request_mask(model)).reshape([n, ludo_core::NUM_BOARD_CELLS])?;
        Ok((xa, ma))
    }

    fn submit(&mut self, model: u8, policy: numpy::PyReadonlyArray2<'_, f32>, value: numpy::PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        self.inner.submit(model, policy.as_slice()?, value.as_slice()?).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Drains recorded decisions of finished games as a dict of numpy arrays.
    fn take_records<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use numpy::PyArrayMethods;
        let recs = std::mem::take(&mut self.inner.records);
        let k = recs.len();
        let fx = recs.first().map(|r| r.x.len()).unwrap_or(ludo_core::UNI_FRAME_SIZE);
        let mut x = Vec::with_capacity(k * fx);
        let mut cells = Vec::with_capacity(k * 4);
        let (mut game, mut matchup, mut seat, mut step, mut nc, mut act, mut logp, mut val, mut lt, mut np_, mut rew) =
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for r in &recs {
            x.extend_from_slice(&r.x);
            cells.extend_from_slice(&r.cells);
            game.push(r.game_id);
            matchup.push(r.matchup);
            seat.push(r.seat);
            step.push(r.step);
            nc.push(r.n_cells);
            act.push(r.action);
            logp.push(r.logp);
            val.push(r.value);
            lt.push(r.legal_tokens);
            np_.push(r.n_players);
            rew.push(r.reward);
        }
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("x", PyArray1::from_vec_bound(py, x).reshape([k, fx])?)?;
        d.set_item("cells", PyArray1::from_vec_bound(py, cells).reshape([k, 4])?)?;
        d.set_item("game", PyArray1::from_vec_bound(py, game))?;
        d.set_item("matchup", PyArray1::from_vec_bound(py, matchup))?;
        d.set_item("seat", PyArray1::from_vec_bound(py, seat))?;
        d.set_item("step", PyArray1::from_vec_bound(py, step))?;
        d.set_item("n_cells", PyArray1::from_vec_bound(py, nc))?;
        d.set_item("action", PyArray1::from_vec_bound(py, act))?;
        d.set_item("logp", PyArray1::from_vec_bound(py, logp))?;
        d.set_item("value", PyArray1::from_vec_bound(py, val))?;
        d.set_item("legal_tokens", PyArray1::from_vec_bound(py, lt))?;
        d.set_item("n_players", PyArray1::from_vec_bound(py, np_))?;
        d.set_item("reward", PyArray1::from_vec_bound(py, rew))?;
        Ok(d)
    }

    /// Drains finished-game results: list of (game_id, matchup, role_of_seat[4], winner_seat, moves).
    fn take_results(&mut self) -> Vec<(u64, u16, [u8; 4], i8, u32)> {
        std::mem::take(&mut self.inner.results)
            .into_iter()
            .map(|r| (r.game_id, r.matchup, r.role_of_seat, r.winner, r.moves))
            .collect()
    }

    fn in_flight(&self) -> usize {
        self.inner.in_flight()
    }

    fn finished(&self) -> u64 {
        self.inner.finished
    }
}

#[pymodule]
fn alphaludo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_winner_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_alphazero_2p_batch, m)?)?;
    m.add_function(wrap_pyfunction!(select_mcts_move_2p, m)?)?;
    m.add_function(wrap_pyfunction!(select_bot_move_2p, m)?)?;
    m.add_function(wrap_pyfunction!(encode_v15_2p, m)?)?;
    m.add_class::<NetFarm>()?;
    m.add_function(wrap_pyfunction!(encode_unified, m)?)?;
    m.add_function(wrap_pyfunction!(unified_track_cell, m)?)?;
    m.add_function(wrap_pyfunction!(select_bot_move, m)?)?;
    m.add_class::<SlStream>()?;
    m.add_class::<Farm>()?;
    m.add_class::<SearchFarm>()?;
    m.add_function(wrap_pyfunction!(eval4_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(eval4_info, m)?)?;
    Ok(())
}

