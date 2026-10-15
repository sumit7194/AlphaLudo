//! Teacher / self-play training record, format v2 (shared by the
//! `generate_sl_teacher` binary and the network-MCTS farm).
//!
//! A shard is a 16-byte header — magic b"ALSL" | u16 version = 2 |
//! u16 record_size = 32 | u32 n_records | u32 0 — followed by n_records
//! little-endian `SLRecordV2`. Reader: td_ludo/sl_dataset_v153.py.

pub const FORMAT_VERSION: u16 = 2;
pub const MAGIC: &[u8; 4] = b"ALSL";
pub const HEADER_BYTES: usize = 16;

/// Bot kind ids stored in `mover_bot`.
pub const BOT_MCTS: u8 = 0;
pub const BOT_DEPTH2: u8 = 1;
pub const BOT_EXPECTIMAX: u8 = 2;
/// Network-guided MCTS (priors + values from a V15 GraphTransformer).
pub const BOT_NET_MCTS: u8 = 3;
/// Bare network, greedy.
pub const BOT_NET_GREEDY: u8 = 4;

/// `flags` bits.
pub const FLAG_HAS_PI: u8 = 1 << 0; // `pi` holds a search visit distribution
pub const FLAG_TEACHER_SEAT: u8 = 1 << 1; // mover is the teacher seat
pub const FLAG_FORCED: u8 = 1 << 2; // exactly one legal move
pub const FLAG_EXPLORATORY: u8 = 1 << 3; // move PLAYED was sampled, not argmax

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct SLRecordV2 {
    pub positions: [i8; 16],
    pub current_player: i8,
    pub dice_roll: i8,
    /// Label action (argmax / chosen by the policy), token index 0..3.
    pub action: i8,
    /// Final outcome from the mover's perspective: +1 win, -1 loss, 0 unfinished.
    pub target_value: i8,
    pub mover_bot: u8,
    pub legal_count: u8,
    /// Mover's consecutive-six count after this roll.
    pub consecutive_sixes: u8,
    pub flags: u8,
    /// Visit distribution by token index, quantised to 0..255.
    pub pi: [u8; 4],
    pub game_id: u32,
}
const _: () = assert!(std::mem::size_of::<SLRecordV2>() == 32);

pub fn quantise_pi(pi: &[f32; 4]) -> [u8; 4] {
    let mut q = [0u8; 4];
    for i in 0..4 {
        q[i] = (pi[i].clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    q
}

pub fn shard_header(n_records: usize) -> [u8; HEADER_BYTES] {
    let mut h = [0u8; HEADER_BYTES];
    h[0..4].copy_from_slice(MAGIC);
    h[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    h[6..8].copy_from_slice(&32u16.to_le_bytes());
    h[8..12].copy_from_slice(&(n_records as u32).to_le_bytes());
    h
}

pub fn records_as_bytes(recs: &[SLRecordV2]) -> &[u8] {
    // SAFETY: repr(C, packed), 32 bytes of plain integers.
    unsafe { std::slice::from_raw_parts(recs.as_ptr() as *const u8, recs.len() * 32) }
}
