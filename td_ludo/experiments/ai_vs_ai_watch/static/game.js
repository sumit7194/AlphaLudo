/**
 * AI-vs-AI Watch & Comment — frontend
 *
 * Auto-play loop: setInterval calls /api/play_step at the chosen speed.
 * Pause/resume just toggles the interval. Scrubbing is purely visual —
 * the underlying game keeps its current ply; clicking 'live ⇥' or
 * resuming play snaps the view back to the latest move.
 */

// ── Constants (mirror play/static/game.js for board geometry) ─────────
const PATH_COORDS_P0 = [
    [6,1],[6,2],[6,3],[6,4],[6,5],
    [5,6],[4,6],[3,6],[2,6],[1,6],[0,6],
    [0,7],[0,8],
    [1,8],[2,8],[3,8],[4,8],[5,8],
    [6,9],[6,10],[6,11],[6,12],[6,13],[6,14],
    [7,14],[8,14],
    [8,13],[8,12],[8,11],[8,10],[8,9],
    [9,8],[10,8],[11,8],[12,8],[13,8],[14,8],
    [14,7],[14,6],
    [13,6],[12,6],[11,6],[10,6],[9,6],
    [8,5],[8,4],[8,3],[8,2],[8,1],[8,0],
    [7,0],
];
const HOME_RUN_P0 = [[7,1],[7,2],[7,3],[7,4],[7,5]];
const HOME_COORD = [7,7];
const BASE_COORDS = {
    0: [[2,2],[2,3],[3,2],[3,3]],
    1: [[2,11],[2,12],[3,11],[3,12]],
    2: [[11,11],[11,12],[12,11],[12,12]],
    3: [[11,2],[11,3],[12,2],[12,3]],
};
const SAFE_INDICES = new Set([0, 8, 13, 21, 26, 34, 39, 47]);

// ── State ─────────────────────────────────────────────────────────────
let boardLayout = null;
let history = [];           // local copy of server-side history
let viewIdx = -1;           // which history record we're showing (-1 = empty)
let isLive = true;          // are we showing the latest record?
let playing = false;        // auto-play interval is active
// Default 1000ms (was 1500ms) — search analysis adds ~5ms server-side and the
// extra panel makes pauses more useful, so faster default helps you scan.
let intervalMs = 1000;
let timer = null;
let terminal = false;
let winner = -1;
let gameId = null;
let commentTargetIdx = null;
let commentPreferred = null;
let commentedIdxs = new Set();   // local hint that comment was saved (visual)

const cellTypes = {};

// ── Board geometry helpers ────────────────────────────────────────────
function rotate90cw([r, c]) { return [c, 14 - r]; }

function getBoardCoord(player, pos, tokenIndex = 0) {
    if (pos === -1) return BASE_COORDS[player][tokenIndex];
    let local;
    if (pos === 99) local = HOME_COORD;
    else if (pos > 50) {
        const idx = pos - 51;
        local = HOME_RUN_P0[idx] || HOME_COORD;
    } else local = PATH_COORDS_P0[pos];
    let coord = local;
    for (let i = 0; i < player; i++) coord = rotate90cw(coord);
    return coord;
}

function getCornerRegion(row, col) {
    if (row < 6 && col < 6) return { player: 0, name: 'green', yard: row >= 1 && row <= 4 && col >= 1 && col <= 4 };
    if (row < 6 && col > 8) return { player: 1, name: 'red', yard: row >= 1 && row <= 4 && col >= 10 && col <= 13 };
    if (row > 8 && col > 8) return { player: 2, name: 'yellow', yard: row >= 10 && row <= 13 && col >= 10 && col <= 13 };
    if (row > 8 && col < 6) return { player: 3, name: 'blue', yard: row >= 10 && row <= 13 && col >= 1 && col <= 4 };
    return null;
}

function getBoardMetrics() {
    const styles = getComputedStyle(document.documentElement);
    const readVar = (name, fb) => {
        const v = parseFloat(styles.getPropertyValue(name));
        return Number.isFinite(v) ? v : fb;
    };
    const tokenScale = readVar('--token-scale', 0.68);
    let cellSize = readVar('--cell-size', NaN);
    if (!Number.isFinite(cellSize)) {
        const sample = document.querySelector('.cell');
        cellSize = sample ? sample.getBoundingClientRect().width : 38;
    }
    let tokenSize = readVar('--token-size', NaN);
    if (!Number.isFinite(tokenSize)) tokenSize = Math.max(8, Math.round(cellSize * tokenScale));
    return { cellSize, boardGap: readVar('--board-gap', 1), boardPad: readVar('--board-pad', 3), tokenSize };
}

// ── Board build ───────────────────────────────────────────────────────
function buildBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';
    // Dice overlay sits over the central pinwheel — built once, updated on every render.
    const dice = document.createElement('div');
    dice.id = 'diceOverlay';
    dice.className = 'dice-overlay empty';
    dice.innerHTML = '<span class="dice-num">—</span>';
    board.appendChild(dice);

    const pathSet = new Set();
    boardLayout.path_squares.forEach(([r, c]) => pathSet.add(`${r},${c}`));
    const safeSet = new Set();
    boardLayout.safe_squares.forEach(([r, c]) => safeSet.add(`${r},${c}`));
    const hrSets = {};
    for (const [p, coords] of Object.entries(boardLayout.home_runs)) {
        hrSets[p] = new Set(coords.map(([r, c]) => `${r},${c}`));
    }
    const classicHomeRuns = {};
    for (const player of [0, 1, 2, 3]) {
        classicHomeRuns[player] = new Set(
            HOME_RUN_P0.map((_, idx) => getBoardCoord(player, 51 + idx).join(','))
        );
    }
    const baseSets = {};
    for (const [player, coords] of Object.entries(boardLayout.bases)) {
        coords.forEach(([r, c]) => { baseSets[`${r},${c}`] = player; });
    }
    const spawnSet = {};
    for (const [player, [r, c]] of Object.entries(boardLayout.spawn_squares)) {
        spawnSet[`${r},${c}`] = player;
    }
    const [hcr, hcc] = boardLayout.home_center;

    for (let r = 0; r < 15; r++) {
        for (let c = 0; c < 15; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = `cell-${r}-${c}`;
            const key = `${r},${c}`;
            const corner = getCornerRegion(r, c);
            if (corner) {
                cell.classList.add(`corner-${corner.name}`, `corner-player-${corner.player}`);
                if (corner.yard) cell.classList.add('yard-cell', `yard-${corner.name}`);
            }
            for (const player of [0, 1, 2, 3]) {
                if (classicHomeRuns[player].has(key)) cell.classList.add(`home-run-${player}`);
            }
            if (r === hcr && c === hcc) { cell.classList.add('home-center'); cellTypes[key] = 'home-center'; }
            else if (baseSets[key] !== undefined) { const p = baseSets[key]; cell.classList.add(`base-${p}`); cellTypes[key] = `base-${p}`; }
            else if (hrSets['0']?.has(key)) { cell.classList.add('home-run-0'); cellTypes[key] = 'home-run-0'; }
            else if (hrSets['2']?.has(key)) { cell.classList.add('home-run-2'); cellTypes[key] = 'home-run-2'; }
            else if (safeSet.has(key)) { cell.classList.add('safe'); cellTypes[key] = 'safe'; }
            else if (pathSet.has(key)) { cell.classList.add('path'); cellTypes[key] = 'path'; }
            else { cell.classList.add('empty'); cellTypes[key] = 'empty'; }
            if (spawnSet[key] !== undefined) cell.classList.add(`spawn-${spawnSet[key]}`);
            board.appendChild(cell);
        }
    }
}

function renderTokens(tokenCoords, highlightToken, highlightPlayer) {
    document.querySelectorAll('.token').forEach(t => t.remove());
    const board = document.getElementById('board');
    const { cellSize, boardGap, boardPad, tokenSize } = getBoardMetrics();
    // Track scored count per player so we can spread their finished
    // tokens around the home center (behind the dice overlay).
    const scoredCount = { 0: 0, 2: 0 };
    for (const [playerStr, tokens] of Object.entries(tokenCoords || {})) {
        const player = parseInt(playerStr);
        tokens.forEach((tok, idx) => {
            const token = document.createElement('div');
            token.className = `token player-${player}`;
            token.innerHTML = `<span class="token-label">${idx}</span>`;

            if (tok.scored) {
                // Park scored tokens at home center (7,7) with a small
                // per-player offset so they sit behind the dice overlay
                // rather than disappearing entirely.
                const cx = boardPad + 7 * (cellSize + boardGap) + (cellSize - tokenSize) / 2;
                const cy = boardPad + 7 * (cellSize + boardGap) + (cellSize - tokenSize) / 2;
                // Player 0 fans up-left, player 2 fans down-right
                const dir = player === 0 ? -1 : 1;
                const off = scoredCount[player] * 6;
                token.style.left = `${cx + dir * off}px`;
                token.style.top  = `${cy + dir * off}px`;
                token.classList.add('scored-token');
                scoredCount[player] += 1;
                board.appendChild(token);
                return;
            }

            const left = boardPad + tok.col * (cellSize + boardGap) + (cellSize - tokenSize) / 2;
            const top = boardPad + tok.row * (cellSize + boardGap) + (cellSize - tokenSize) / 2;
            token.style.left = `${left}px`;
            token.style.top = `${top}px`;
            const stackKey = `${tok.row},${tok.col}`;
            const existing = board.querySelectorAll(`.token[data-stackkey="${stackKey}"]:not(.scored-token)`);
            if (existing.length > 0) {
                const offset = existing.length * 5;
                token.style.left = `${left + offset}px`;
                token.style.top = `${top - offset}px`;
                token.style.zIndex = 10 + existing.length;
            }
            token.dataset.stackkey = stackKey;
            if (highlightPlayer === player && highlightToken === idx) {
                token.classList.add('model-pick');
            }
            board.appendChild(token);
        });
    }
}

function updateBoardOverlays(rec) {
    // Dice in board center
    const dice = document.getElementById('diceOverlay');
    if (dice) {
        dice.classList.remove('player-0', 'player-2', 'empty');
        const numEl = dice.querySelector('.dice-num');
        if (rec && rec.dice) {
            dice.classList.add(`player-${rec.player}`);
            numEl.textContent = rec.dice;
            // Trigger pop animation only when dice value actually changes.
            if (dice.dataset.lastSig !== `${rec.move_idx}:${rec.dice}`) {
                numEl.classList.remove('dice-roll-anim');
                // force reflow to restart animation
                void numEl.offsetWidth;
                numEl.classList.add('dice-roll-anim');
                dice.dataset.lastSig = `${rec.move_idx}:${rec.dice}`;
            }
        } else {
            dice.classList.add('empty');
            numEl.textContent = '—';
            dice.dataset.lastSig = '';
        }
    }
    // Corner highlight: toggle on every cell with corner-player-N
    document.querySelectorAll('.cell.turn-on').forEach(c => c.classList.remove('turn-on'));
    if (rec && (rec.player === 0 || rec.player === 2)) {
        document.querySelectorAll(`.cell.corner-player-${rec.player}`).forEach(c => {
            c.classList.add('turn-on');
        });
    }
}

// ── Init / state load ────────────────────────────────────────────────
async function init() {
    boardLayout = await (await fetch('/api/layout')).json();
    buildBoard();
    const info = await (await fetch('/api/info')).json();
    document.getElementById('meta').textContent = `model: ${info.model_version} · game: ${info.game_id}`;
    gameId = info.game_id;
    await refreshHistory();
    wireControls();
    render();
}

async function refreshHistory(since = 0) {
    const data = await (await fetch(`/api/history?since=${since}`)).json();
    if (since === 0) {
        history = data.records;
    } else {
        history = history.concat(data.records);
    }
    terminal = data.terminal;
    winner = data.winner;
    if (history.length > 0 && (isLive || viewIdx === -1)) {
        viewIdx = history.length - 1;
    }
}

// ── Auto-play loop ───────────────────────────────────────────────────
async function tick() {
    if (terminal) { stopPlaying(); return; }
    try {
        const data = await (await fetch('/api/play_step', { method: 'POST' })).json();
        if (data.record) {
            history.push(data.record);
            terminal = data.terminal;
            winner = data.winner;
            if (isLive) viewIdx = history.length - 1;
        }
        if (data.terminal) stopPlaying();
        render();
    } catch (e) {
        console.warn('play_step failed', e);
    }
}

function startPlaying() {
    if (playing || terminal) return;
    playing = true;
    // Whenever we resume, snap back to live so the user sees the latest
    // ply before the next tick fires.
    isLive = true;
    if (history.length > 0) viewIdx = history.length - 1;
    render();
    document.getElementById('playPauseBtn').textContent = '⏸ Pause';
    tick();   // immediate first step
    timer = setInterval(tick, intervalMs);
}

function stopPlaying() {
    playing = false;
    if (timer) { clearInterval(timer); timer = null; }
    document.getElementById('playPauseBtn').textContent = terminal ? '▣ Done' : '▶ Play';
}

function setSpeed(ms) {
    intervalMs = ms;
    if (playing) {
        clearInterval(timer);
        timer = setInterval(tick, intervalMs);
    }
}

// ── Scrubbing ────────────────────────────────────────────────────────
function viewAt(idx) {
    if (history.length === 0) return;
    idx = Math.max(0, Math.min(history.length - 1, idx));
    viewIdx = idx;
    isLive = (idx === history.length - 1);
    render();
}

function jumpToLive() {
    if (history.length === 0) return;
    isLive = true;
    viewIdx = history.length - 1;
    render();
}

// ── Rendering ────────────────────────────────────────────────────────
function render() {
    const scrubInfo = document.getElementById('scrubInfo');
    const scrubber = document.getElementById('scrubber');
    scrubber.max = Math.max(0, history.length - 1);
    scrubber.value = viewIdx >= 0 ? viewIdx : 0;

    if (history.length === 0) {
        scrubInfo.textContent = 'no moves yet';
        renderTokens(null);
        updateBoardOverlays(null);
        document.getElementById('moveCard').querySelector('.move-meta').textContent = 'no move yet';
        document.getElementById('movePolicy').innerHTML = '';
        document.getElementById('commentCard').style.display = 'none';
        renderTimeline();
        return;
    }

    const rec = history[viewIdx];
    const liveTag = isLive
        ? '<span class="live-tag">● LIVE</span>'
        : '<span class="past-tag">⏪ HISTORY</span>';
    scrubInfo.innerHTML = `move ${viewIdx + 1} / ${history.length} · ${liveTag}`
        + (terminal ? `<br><b>Game over — P${winner} won</b>` : '');

    // Board: render the post-move state (what you'd see right after that ply)
    renderTokens(rec.token_coords_after, rec.ai_chosen, rec.player);
    updateBoardOverlays(rec);

    renderMoveCard(rec);
    renderSearchCard(rec);
    renderCommentCard(rec);
    renderTimeline();

    // Message area
    const msg = document.getElementById('messageArea');
    if (rec.passed) {
        msg.textContent = `P${rec.player} rolled ${rec.dice} — ${rec.pass_reason === 'triple_six' ? 'triple 6, turn lost' : 'no legal moves'}.`;
    } else {
        const tag = rec.captured ? ' ⚔️ CAPTURE' : '';
        const bonus = rec.bonus ? ' (bonus turn)' : '';
        msg.textContent = `P${rec.player} rolled ${rec.dice} → token ${rec.ai_chosen}: ${rec.from_pos}→${rec.to_pos}${tag}${bonus}`;
    }
}

function renderMoveCard(rec) {
    const card = document.getElementById('moveCard');
    const meta = card.querySelector('.move-meta');
    const policyDiv = document.getElementById('movePolicy');

    const playerCls = `p${rec.player}`;
    let badges = '';
    if (rec.passed) badges += '<span class="badge passed">passed</span>';
    if (rec.forced) badges += '<span class="badge forced">forced</span>';
    if (rec.captured) badges += '<span class="badge captured">capture</span>';
    if (rec.bonus) badges += '<span class="badge bonus">bonus</span>';

    meta.innerHTML = `<span class="player-tag ${playerCls}">P${rec.player}</span>
        ply ${rec.ply} · 🎲 ${rec.dice}${badges}`;

    if (rec.passed || !rec.ai_policy) {
        policyDiv.innerHTML = '<div style="font-size: 12px; color: #888;">No model decision (turn skipped).</div>';
        return;
    }

    const legalSet = new Set(rec.legal_tokens || []);
    policyDiv.innerHTML = rec.ai_policy.map((p, i) => {
        const pct = (p * 100).toFixed(1);
        const isChosen = i === rec.ai_chosen;
        const isIllegal = !legalSet.has(i);
        const cls = [
            isChosen ? 'chosen' : '',
            isIllegal ? 'illegal' : '',
        ].filter(Boolean).join(' ');
        return `<div class="prob-row ${cls}">
            <span class="prob-label">T${i}${isChosen ? '✓' : ''}</span>
            <span class="prob-bar"><span class="prob-fill" style="width:${pct}%"></span></span>
            <span class="prob-pct">${pct}%</span>
        </div>`;
    }).join('');

    if (typeof rec.ai_value === 'number') {
        policyDiv.innerHTML += `<div style="font-size: 11px; color: #888; margin-top: 6px; font-family: 'IBM Plex Mono', monospace;">
            value (P${rec.player} POV): ${rec.ai_value.toFixed(3)}
        </div>`;
    }
}

function renderSearchCard(rec) {
    const card = document.getElementById('searchCard');
    const picks = document.getElementById('searchPicks');
    const table = document.getElementById('searchTable');
    const detail = document.getElementById('penaltyDetail');

    const sa = rec.search_analysis;
    if (!sa) {
        card.style.display = 'none';
        return;
    }
    card.style.display = '';

    const fmtNum = (v) => {
        if (Math.abs(v) < 1e-6) return `<span class="num-zero">0.000</span>`;
        const cls = v > 0 ? 'num-pos' : 'num-neg';
        const sign = v > 0 ? '+' : '';
        return `<span class="${cls}">${sign}${v.toFixed(3)}</span>`;
    };
    const posLabel = (p) => {
        if (p === -1) return 'base';
        if (p === 99) return 'HOME';
        if (p > 50) return `H${p - 50}`;
        return String(p);
    };

    // Picks summary at top
    const picksHtml = [
        ['model',  'M', sa.model_pick],
        ['reward', 'R', sa.reward_pick],
        ['blend',  `B(α=${sa.alpha})`, sa.blend_pick],
    ].map(([cls, label, t]) => {
        const sameAsModel = (t === sa.model_pick) && cls !== 'model';
        return `<span class="pick-tag ${cls} ${sameAsModel ? 'same' : ''}">${label}: T${t}</span>`;
    }).join('');
    picks.innerHTML = picksHtml;

    // Per-action rows
    const sortedActions = Object.keys(sa.actions).map(Number).sort((a, b) => a - b);
    const rows = sortedActions.map(a => {
        const info = sa.actions[a];
        const isModel = (a === sa.model_pick);
        const isReward = (a === sa.reward_pick);
        const isBlend = (a === sa.blend_pick);
        const picksOnRow = [isModel, isReward, isBlend].filter(Boolean).length;
        const cls = (picksOnRow > 1)
            ? 'is-multi'
            : (isModel ? 'is-model' : (isReward ? 'is-reward' : (isBlend ? 'is-blend' : '')));
        const marks = [
            isModel ? '<span class="mk-m">M</span>' : '',
            isReward ? '<span class="mk-r">R</span>' : '',
            isBlend ? '<span class="mk-b">B</span>' : '',
        ].filter(Boolean).join(' ');
        const captureMarker = info.captured ? ' ⚔' : '';
        return `<tr class="row-action ${cls}" data-action="${a}">
            <td class="tl">T${a}</td>
            <td>${posLabel(info.pre_pos)}<span class="move-arrow"> → </span>${posLabel(info.post_pos)}${captureMarker}</td>
            <td>${fmtNum(info.shaped_reward)}</td>
            <td>${fmtNum(info.penalty_total)}</td>
            <td>${(info.policy_prob * 100).toFixed(1)}%</td>
            <td>${fmtNum(info.score_reward)}</td>
            <td>${fmtNum(info.score_blend)}</td>
            <td class="pick-marks">${marks}</td>
        </tr>`;
    }).join('');

    table.innerHTML = `<thead><tr>
        <th>tok</th><th>move</th><th>shaped</th><th>pen</th><th>π</th>
        <th>R-score</th><th>B-score</th><th></th>
    </tr></thead><tbody>${rows}</tbody>`;

    // Penalty breakdown detail
    const detailLines = [];
    sortedActions.forEach(a => {
        const bd = sa.actions[a].penalty_breakdown;
        if (Object.keys(bd).length > 0) {
            const parts = Object.entries(bd).map(([k, v]) => `<b>${k}</b>=${v.toFixed(3)}`);
            detailLines.push(`T${a}: ${parts.join(', ')}`);
        }
    });
    detail.innerHTML = detailLines.length
        ? `<i>Triggered penalties:</i> ${detailLines.join(' · ')}`
        : '';
}

function renderCommentCard(rec) {
    const card = document.getElementById('commentCard');
    if (rec.passed || rec.ai_policy === null) {
        // Can't comment on a passed turn — no decision was made.
        card.style.display = 'none';
        commentTargetIdx = null;
        return;
    }
    card.style.display = '';
    if (commentTargetIdx !== rec.move_idx) {
        // Switched to a new record — reset the form.
        commentTargetIdx = rec.move_idx;
        commentPreferred = null;
        document.getElementById('commentText').value = '';
        document.getElementById('commentStatus').textContent = '';
        document.getElementById('commentStatus').className = 'comment-status';
    }
    document.getElementById('commentMoveIdx').textContent = rec.move_idx;

    const legalSet = new Set(rec.legal_tokens || []);
    card.querySelectorAll('.pref-btn').forEach(btn => {
        const tok = parseInt(btn.dataset.tok, 10);
        btn.classList.remove('active', 'illegal', 'chosen');
        if (!legalSet.has(tok)) {
            btn.classList.add('illegal');
            btn.disabled = true;
        } else if (tok === rec.ai_chosen) {
            btn.classList.add('chosen');
            btn.disabled = true;
            btn.title = 'AI chose this — pick a different token to mark a disagreement';
        } else {
            btn.disabled = false;
            btn.title = '';
            if (tok === commentPreferred) btn.classList.add('active');
        }
    });
}

function renderTimeline() {
    const tl = document.getElementById('timeline');
    // Render in reverse (newest first) so it doesn't push the action off-screen
    const rows = [];
    for (let i = history.length - 1; i >= 0; i--) {
        const r = history[i];
        const isCurrent = i === viewIdx;
        const isCommented = commentedIdxs.has(i);
        const playerCls = `p${r.player}`;
        const marker = r.captured ? '⚔️' : (r.bonus ? '✨' : (r.passed ? '∅' : ''));
        const choice = r.passed
            ? `pass (${r.pass_reason || ''})`
            : `T${r.ai_chosen} ${r.from_pos}→${r.to_pos}${r.forced ? ' (forced)' : ''}`;
        const commentMark = isCommented ? '💬' : '';
        rows.push(`<div class="timeline-row ${isCurrent ? 'current' : ''} ${isCommented ? 'commented' : ''}" data-idx="${i}">
            <span class="tl-idx">#${i}</span>
            <span class="tl-player ${playerCls}">P${r.player}</span>
            <span>🎲${r.dice}</span>
            <span>${choice}</span>
            <span class="tl-marker">${marker}${commentMark}</span>
        </div>`);
    }
    tl.innerHTML = rows.join('');
    tl.querySelectorAll('.timeline-row').forEach(row => {
        row.addEventListener('click', () => {
            const idx = parseInt(row.dataset.idx, 10);
            viewAt(idx);
        });
    });
}

// ── Comment submission ──────────────────────────────────────────────
async function saveComment() {
    if (commentTargetIdx === null) return;
    const text = document.getElementById('commentText').value.trim();
    const status = document.getElementById('commentStatus');
    if (!text && commentPreferred === null) {
        status.textContent = 'Add a comment or pick a preferred token.';
        status.className = 'comment-status error';
        return;
    }
    status.textContent = 'Saving…';
    status.className = 'comment-status';
    try {
        const res = await fetch('/api/comment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                move_idx: commentTargetIdx,
                comment: text,
                preferred_token: commentPreferred,
            }),
        });
        const data = await res.json();
        if (data.ok) {
            status.textContent = `Saved ✓ (move #${commentTargetIdx})`;
            status.className = 'comment-status success';
            commentedIdxs.add(commentTargetIdx);
            renderTimeline();
        } else {
            status.textContent = `Failed: ${data.error || 'unknown'}`;
            status.className = 'comment-status error';
        }
    } catch (e) {
        status.textContent = 'Failed: network error';
        status.className = 'comment-status error';
    }
}

function clearComment() {
    document.getElementById('commentText').value = '';
    commentPreferred = null;
    document.getElementById('commentStatus').textContent = '';
    document.getElementById('commentStatus').className = 'comment-status';
    if (commentTargetIdx !== null && history[commentTargetIdx]) {
        renderCommentCard(history[commentTargetIdx]);
    }
}

// ── Wiring ──────────────────────────────────────────────────────────
function wireControls() {
    document.getElementById('playPauseBtn').addEventListener('click', () => {
        if (playing) stopPlaying(); else startPlaying();
    });
    document.querySelectorAll('.watch-btn.speed').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.watch-btn.speed').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            setSpeed(parseInt(btn.dataset.speed, 10));
        });
    });
    document.getElementById('stepBackBtn').addEventListener('click', () => {
        if (history.length === 0) return;
        viewAt(viewIdx - 1);
    });
    document.getElementById('stepFwdBtn').addEventListener('click', () => {
        if (history.length === 0) return;
        viewAt(viewIdx + 1);
    });
    document.getElementById('jumpLiveBtn').addEventListener('click', jumpToLive);
    document.getElementById('scrubber').addEventListener('input', e => {
        viewAt(parseInt(e.target.value, 10));
    });
    document.getElementById('newGameBtn').addEventListener('click', async () => {
        stopPlaying();
        const data = await (await fetch('/api/new_game', { method: 'POST' })).json();
        history = [];
        viewIdx = -1;
        terminal = false;
        winner = -1;
        gameId = data.game_id;
        commentedIdxs.clear();
        commentTargetIdx = null;
        commentPreferred = null;
        document.getElementById('commentText').value = '';
        document.getElementById('meta').textContent = `model: ${data.model_version} · game: ${data.game_id}`;
        render();
    });
    document.getElementById('commentSaveBtn').addEventListener('click', saveComment);
    document.getElementById('commentClearBtn').addEventListener('click', clearComment);
    document.getElementById('commentCard').querySelectorAll('.pref-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tok = parseInt(btn.dataset.tok, 10);
            commentPreferred = (commentPreferred === tok) ? null : tok;
            if (commentTargetIdx !== null && history[commentTargetIdx]) {
                renderCommentCard(history[commentTargetIdx]);
            }
        });
    });

    // Keyboard: space=pause/resume, ←/→ scrub, l=live
    document.addEventListener('keydown', e => {
        const inField = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
        if (inField) return;
        if (e.code === 'Space') { e.preventDefault(); if (playing) stopPlaying(); else startPlaying(); }
        else if (e.code === 'ArrowLeft') { e.preventDefault(); viewAt(viewIdx - 1); }
        else if (e.code === 'ArrowRight') { e.preventDefault(); viewAt(viewIdx + 1); }
        else if (e.key === 'l' || e.key === 'L') jumpToLive();
    });
}

// ── Boot ────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', init);
window.addEventListener('resize', render);
