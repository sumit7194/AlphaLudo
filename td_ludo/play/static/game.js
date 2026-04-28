/**
 * AlphaLudo Play — Game Controller
 * Handles board rendering, game flow, API calls, and animations.
 */

// ── Constants ───────────────────────────────────────────────
const HUMAN = 0;
const AI = 2;
const CELL_SIZE = 38;
const BOARD_GAP = 1;
const BOARD_PAD = 3;
const TOKEN_SIZE = 26;

// ── State ───────────────────────────────────────────────────
let gameState = null;
let boardLayout = null;
let legalMoves = [];
let awaitingMove = false;
let isProcessing = false;

// Eval-Lens — track current game_id so we can fetch its review
// after the game ends (server returns a fresh id from /api/new_game).
let currentGameId = null;
let reviewedGameId = null;

// Eval-Lens — V12's recommended token for the current human roll.
// Set from the /api/roll_dice response, cleared on each new roll/move.
let modelPick = null;     // int 0..3, or null
let modelPolicy = null;   // [p0, p1, p2, p3], or null

function syncGameId(data) {
    if (data && typeof data.game_id === 'string' && data.game_id) {
        currentGameId = data.game_id;
    }
}

// Cell type lookup (built from layout data)
const cellTypes = {};  // "r,c" -> type string

// ── Initialization ──────────────────────────────────────────
async function init() {
    // Fetch model info and populate header chip + AI subtitle
    try {
        const infoRes = await fetch('/api/info');
        if (infoRes.ok) {
            const info = await infoRes.json();
            const subtitle = document.getElementById('aiSubtitle');
            if (subtitle && info.subtitle) subtitle.textContent = info.subtitle;
            const chip = document.getElementById('modelChip');
            if (chip) chip.textContent = info.label || (info.version || 'AI').toUpperCase();
        }
    } catch (e) { console.warn('Could not fetch model info', e); }

    // Fetch board layout
    const res = await fetch('/api/layout');
    boardLayout = await res.json();
    buildBoard();
    await newGame();
}

function buildBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';
    
    // Create path lookup
    const pathSet = new Set();
    boardLayout.path_squares.forEach(([r, c]) => pathSet.add(`${r},${c}`));
    
    const safeSet = new Set();
    boardLayout.safe_squares.forEach(([r, c]) => safeSet.add(`${r},${c}`));
    
    const hrSets = {};
    for (const [player, coords] of Object.entries(boardLayout.home_runs)) {
        hrSets[player] = new Set(coords.map(([r, c]) => `${r},${c}`));
    }
    
    const baseSets = {};
    for (const [player, coords] of Object.entries(boardLayout.bases)) {
        coords.forEach(([r, c]) => {
            baseSets[`${r},${c}`] = player;
        });
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
            
            if (r === hcr && c === hcc) {
                cell.classList.add('home-center');
                cellTypes[key] = 'home-center';
            } else if (baseSets[key] !== undefined) {
                const p = baseSets[key];
                cell.classList.add(`base-${p}`);
                cellTypes[key] = `base-${p}`;
            } else if (hrSets['0'] && hrSets['0'].has(key)) {
                cell.classList.add('home-run-0');
                cellTypes[key] = 'home-run-0';
            } else if (hrSets['2'] && hrSets['2'].has(key)) {
                cell.classList.add('home-run-2');
                cellTypes[key] = 'home-run-2';
            } else if (safeSet.has(key)) {
                cell.classList.add('safe');
                cellTypes[key] = 'safe';
            } else if (pathSet.has(key)) {
                cell.classList.add('path');
                cellTypes[key] = 'path';
            } else {
                cell.classList.add('empty');
                cellTypes[key] = 'empty';
            }
            
            // Add spawn marker
            if (spawnSet[key] !== undefined) {
                cell.classList.add(`spawn-${spawnSet[key]}`);
            }
            
            board.appendChild(cell);
        }
    }
}

// ── API Calls ───────────────────────────────────────────────
async function newGame() {
    const res = await fetch('/api/new_game', { method: 'POST' });
    gameState = await res.json();
    syncGameId(gameState);
    reviewedGameId = null;
    legalMoves = [];
    awaitingMove = false;
    isProcessing = false;
    modelPick = null;
    modelPolicy = null;

    // Clear UI
    document.getElementById('winModal').classList.remove('show');
    document.getElementById('reviewModal').classList.remove('show');
    document.getElementById('diceValue').textContent = '?';
    document.getElementById('diceHint').textContent = 'Click to roll';
    document.getElementById('dice').classList.remove('disabled');
    document.getElementById('messageArea').textContent = '';
    document.getElementById('aiProbs').textContent = '';
    document.getElementById('logEntries').innerHTML = '';
    
    renderState();
    updateTurnIndicator();
    addLog('system', 'New game started. You are Green (P0).');
    
    // If AI goes first (shouldn't happen in 2P with P0 starting)
    if (gameState.current_player === AI) {
        await doAITurn();
    }
}

async function rollDice() {
    if (isProcessing || awaitingMove) return;
    if (gameState.is_terminal) return;
    if (gameState.current_player !== HUMAN) return;
    
    isProcessing = true;
    
    // Animate dice
    const dice = document.getElementById('dice');
    dice.classList.add('rolling');
    document.getElementById('diceHint').textContent = '';
    
    // Quick number cycling animation
    let cycles = 0;
    const maxCycles = 8;
    const cycleInterval = setInterval(() => {
        document.getElementById('diceValue').textContent = Math.floor(Math.random() * 6) + 1;
        cycles++;
        if (cycles >= maxCycles) clearInterval(cycleInterval);
    }, 60);
    
    const res = await fetch('/api/roll_dice', { method: 'POST' });
    const data = await res.json();
    gameState = data;
    syncGameId(data);

    // V12's prediction for this roll — used to highlight the recommended
    // token. Cleared whenever the human commits a move.
    modelPick = (typeof data.model_pick === 'number') ? data.model_pick : null;
    modelPolicy = Array.isArray(data.model_policy) ? data.model_policy : null;
    
    // Wait for animation
    await sleep(500);
    clearInterval(cycleInterval);
    dice.classList.remove('rolling');
    
    // `rolled` is the actual roll value (preserved even when server cleared dice_roll
    // because turn passed). `dice_roll` reflects current state (0 after pass).
    const humanRoll = data.rolled || data.dice_roll || '?';
    document.getElementById('diceValue').textContent = humanRoll;

    if (data.triple_six) {
        showMessage('Triple 6! Turn lost! 💀');
        addLog('human', `Rolled 6 (3rd consecutive) — turn lost!`);
        isProcessing = false;
        renderState();
        await sleep(800);
        await doAITurn();
        return;
    }

    if (data.no_moves) {
        showMessage(`Rolled ${humanRoll} — no legal moves`);
        addLog('human', `Rolled ${humanRoll} — no moves available.`);
        isProcessing = false;
        renderState();
        await sleep(1500);  // longer pause so user sees the dice value
        await doAITurn();
        return;
    }

    legalMoves = data.legal_moves || [];
    addLog('human', `Rolled ${humanRoll}`);
    
    if (legalMoves.length === 1) {
        // Auto-play single legal move
        showMessage(`Rolled ${humanRoll} — auto-moving token ${legalMoves[0]}`);
        isProcessing = false;
        await selectToken(legalMoves[0]);
        return;
    }

    showMessage(`Rolled ${humanRoll} — click a highlighted token`);
    document.getElementById('diceHint').textContent = 'Select a token';
    dice.classList.add('disabled');
    
    awaitingMove = true;
    isProcessing = false;
    renderState();
}

async function selectToken(tokenIndex) {
    if (isProcessing) return;
    if (!awaitingMove && legalMoves.length !== 1) return;
    
    isProcessing = true;
    awaitingMove = false;
    legalMoves = [];

    // Capture which token was V12's pick before we clear the prediction,
    // so we can show "you agreed/disagreed" feedback in the log.
    const v12Pick = modelPick;
    modelPick = null;
    modelPolicy = null;

    const res = await fetch('/api/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: tokenIndex }),
    });
    const data = await res.json();
    gameState = data;
    syncGameId(data);
    
    const lastMove = data.last_move;
    if (lastMove) {
        let msg = `Moved token ${lastMove.token}: ${lastMove.from_pos}→${lastMove.to_pos}`;
        if (lastMove.captured) msg += ' ⚔️ CAPTURE!';
        if (v12Pick !== null && v12Pick !== undefined) {
            if (v12Pick === lastMove.token) {
                msg += ' · agreed with AI ✓';
            } else {
                msg += ` · AI wanted T${v12Pick} ✗`;
            }
        }
        addLog('human', msg);
    }

    renderState();

    // Check game over
    if (data.is_terminal) {
        isProcessing = false;
        await sleep(500);
        showGameOver(data.winner);
        return;
    }
    
    // Bonus turn?
    if (data.bonus_turn) {
        showMessage('Bonus turn! Roll again 🎲');
        document.getElementById('dice').classList.remove('disabled');
        document.getElementById('diceValue').textContent = '?';
        document.getElementById('diceHint').textContent = 'Click to roll';
        isProcessing = false;
        return;
    }
    
    isProcessing = false;
    
    // AI's turn
    await sleep(400);
    await doAITurn();
}

async function doAITurn() {
    if (gameState.is_terminal) return;
    if (gameState.current_player !== AI) {
        // Back to human
        updateTurnIndicator();
        document.getElementById('dice').classList.remove('disabled');
        document.getElementById('diceValue').textContent = '?';
        document.getElementById('diceHint').textContent = 'Click to roll';
        return;
    }
    
    updateTurnIndicator();
    document.getElementById('aiThinking').classList.add('show');
    document.getElementById('dice').classList.add('disabled');
    isProcessing = true;
    
    // Delay so user sees the thinking animation
    await sleep(1000);
    
    const res = await fetch('/api/ai_turn', { method: 'POST' });
    const data = await res.json();
    gameState = data;
    syncGameId(data);
    
    document.getElementById('aiThinking').classList.remove('show');
    
    // Show AI's dice roll — pause so user can see it
    // Server always sets ai_roll now (even on no_moves / triple_six paths)
    const aiRoll = data.ai_roll || data.rolled || data.dice_roll || '?';
    document.getElementById('diceValue').textContent = aiRoll;
    showMessage(`AI rolled ${aiRoll}`);
    await sleep(1500);  // Pause longer so user can see dice result before AI moves
    
    // Show AI probabilities — visual prob bars per token
    if (data.ai_probs) {
        const probsDiv = document.getElementById('aiProbs');
        const legalSet = new Set(data.legal_moves || []);
        probsDiv.innerHTML = data.ai_probs
            .map((p, i) => {
                const pct = (p * 100).toFixed(1);
                const isChosen = i === data.ai_chosen;
                const isIllegal = data.legal_moves && !legalSet.has(i);
                const cls = isChosen ? 'chosen' : (isIllegal ? 'illegal' : '');
                return `<div class="prob-row ${cls}">
                    <span class="prob-label">T${i}${isChosen ? '✓' : ''}</span>
                    <span class="prob-bar"><span class="prob-fill" style="width:${pct}%"></span></span>
                    <span class="prob-pct">${pct}%</span>
                </div>`;
            })
            .join('');
    }
    
    if (data.triple_six) {
        addLog('ai', `Rolled 6 (3rd consecutive) — turn lost!`);
        showMessage('AI rolled triple 6! 💀');
    } else if (data.no_moves) {
        addLog('ai', `Rolled ${aiRoll || '?'} — no moves.`);
        showMessage(`AI rolled ${aiRoll || '?'} — no moves`);
    } else {
        const lastMove = data.last_move;
        if (lastMove) {
            let msg = `Rolled ${aiRoll}. Token ${lastMove.token}: ${lastMove.from_pos}→${lastMove.to_pos}`;
            if (lastMove.captured) msg += ' ⚔️ CAPTURE!';
            addLog('ai', msg);
            showMessage(`AI: Token ${lastMove.token} moved ${lastMove.from_pos}→${lastMove.to_pos}${lastMove.captured ? ' ⚔️ CAPTURE!' : ''}`);
        }
    }
    
    renderState();
    await sleep(1500);  // Pause so user can see what AI did
    isProcessing = false;
    
    // Check game over
    if (data.is_terminal) {
        await sleep(500);
        showGameOver(data.winner);
        return;
    }
    
    // AI bonus turn? Keep going
    if (data.bonus_turn || gameState.current_player === AI) {
        showMessage(`AI gets a bonus turn!`);
        await sleep(1000);
        await doAITurn();
        return;
    }
    
    // Back to human — short pause before switching
    await sleep(500);
    updateTurnIndicator();
    document.getElementById('dice').classList.remove('disabled');
    document.getElementById('diceValue').textContent = '?';
    document.getElementById('diceHint').textContent = 'Click to roll';
}

// ── Rendering ───────────────────────────────────────────────
function renderState() {
    if (!gameState) return;
    
    // Remove old tokens
    document.querySelectorAll('.token').forEach(t => t.remove());
    
    const board = document.getElementById('board');
    const coords = gameState.token_coords;
    
    for (const [playerStr, tokens] of Object.entries(coords)) {
        const player = parseInt(playerStr);
        tokens.forEach((tok, idx) => {
            if (tok.scored) return; // Don't render scored tokens on board
            
            const token = document.createElement('div');
            token.className = `token player-${player}`;
            token.textContent = idx;
            token.dataset.player = player;
            token.dataset.token = idx;
            
            // Position based on grid coordinates
            const left = BOARD_PAD + tok.col * (CELL_SIZE + BOARD_GAP) + (CELL_SIZE - TOKEN_SIZE) / 2;
            const top = BOARD_PAD + tok.row * (CELL_SIZE + BOARD_GAP) + (CELL_SIZE - TOKEN_SIZE) / 2;
            token.style.left = `${left}px`;
            token.style.top = `${top}px`;
            
            // Handle stacking (offset slightly if multiple tokens on same cell)
            const stackKey = `${tok.row},${tok.col}`;
            const existing = board.querySelectorAll(`.token[data-stackkey="${stackKey}"]`);
            if (existing.length > 0) {
                const offset = existing.length * 5;
                token.style.left = `${left + offset}px`;
                token.style.top = `${top - offset}px`;
                token.style.zIndex = 10 + existing.length;
            }
            token.dataset.stackkey = stackKey;
            
            // Legal move highlighting
            if (player === HUMAN && awaitingMove && legalMoves.includes(idx)) {
                token.classList.add('legal-move');
                token.onclick = () => selectToken(idx);
            }

            // V12's predicted token — shown only while the human is choosing
            if (player === HUMAN && awaitingMove && idx === modelPick) {
                token.classList.add('model-pick');
                token.title = 'AI recommends this token';
            }

            board.appendChild(token);
        });
    }
    
    // Update scores
    updateScores();
    updateTurnIndicator();
    updateStatusPills();
    updateWinChance();
}

function updateWinChance() {
    if (!gameState) return;
    const fill = document.getElementById('winChanceFill');
    const aiTxt = document.getElementById('winChanceAi');
    const humanTxt = document.getElementById('winChanceHuman');
    if (!fill || !aiTxt || !humanTxt) return;

    const aiWin = gameState.ai_win_chance;
    if (aiWin === null || aiWin === undefined) {
        fill.style.width = '50%';
        aiTxt.textContent = '—';
        humanTxt.textContent = '—';
        return;
    }
    const aiPct = Math.round(aiWin * 100);
    const humanPct = 100 - aiPct;
    fill.style.width = aiPct + '%';
    aiTxt.textContent = aiPct + '% AI';
    humanTxt.textContent = humanPct + '% You';
}

function updateScores() {
    if (!gameState) return;
    
    const humanScore = gameState.scores['0'] || 0;
    const aiScore = gameState.scores['2'] || 0;
    
    for (let i = 0; i < 4; i++) {
        const hEl = document.getElementById(`human-scored-${i}`);
        const aEl = document.getElementById(`ai-scored-${i}`);
        if (hEl) hEl.classList.toggle('active', i < humanScore);
        if (aEl) aEl.classList.toggle('active', i < aiScore);
    }
    
    document.getElementById('human-score-label').textContent = `${humanScore} / 4`;
    document.getElementById('ai-score-label').textContent = `${aiScore} / 4`;
}

function updateStatusPills() {
    if (!gameState) return;
    const humanPill = document.getElementById('humanStatusPill');
    const aiPill = document.getElementById('aiStatusPill');
    const humanText = document.getElementById('humanStatusText');
    const aiText = document.getElementById('aiStatusText');
    if (!humanPill || !aiPill) return;

    humanPill.classList.remove('active');
    aiPill.classList.remove('active');

    if (gameState.is_terminal) {
        const winner = gameState.winner;
        humanText.textContent = winner === HUMAN ? 'Won 🏆' : 'Lost';
        aiText.textContent = winner === AI_PLAYER ? 'Won 🏆' : 'Lost';
        return;
    }

    if (gameState.current_player === HUMAN) {
        humanPill.classList.add('active');
        humanText.textContent = 'Your Turn';
        aiText.textContent = 'Waiting…';
    } else {
        aiPill.classList.add('active');
        aiText.textContent = 'Thinking…';
        humanText.textContent = 'Waiting…';
    }
}

function updateTurnIndicator() {
    if (!gameState) return;
    const el = document.getElementById('turnIndicator');
    const text = document.getElementById('turnText');
    
    el.classList.remove('human-turn', 'ai-turn');
    
    if (gameState.is_terminal) {
        text.textContent = 'Game Over';
        return;
    }
    
    if (gameState.current_player === HUMAN) {
        el.classList.add('human-turn');
        text.textContent = "Your Turn";
    } else {
        el.classList.add('ai-turn');
        text.textContent = "AI Thinking...";
    }
}

function showMessage(msg) {
    const el = document.getElementById('messageArea');
    el.textContent = msg;
    el.classList.remove('flash');
    void el.offsetWidth; // Force reflow
    el.classList.add('flash');
}

function addLog(type, message) {
    const entries = document.getElementById('logEntries');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}-log`;
    entry.textContent = message;
    entries.prepend(entry);
    
    // Keep max 50 entries
    while (entries.children.length > 50) {
        entries.removeChild(entries.lastChild);
    }
}

function showGameOver(winner) {
    // Pin the just-finished game id for the Review button. `currentGameId`
    // will get overwritten the moment newGame() fires.
    reviewedGameId = currentGameId;

    const modal = document.getElementById('winModal');
    const icon = document.getElementById('modalIcon');
    const title = document.getElementById('modalTitle');
    const sub = document.getElementById('modalSubtext');
    
    if (winner === HUMAN) {
        icon.textContent = '🏆';
        title.textContent = 'You Win!';
        sub.textContent = 'Congratulations! You beat AlphaLudo AI!';
    } else {
        icon.textContent = '🤖';
        title.textContent = 'AI Wins';
        sub.textContent = 'AlphaLudo outplayed you this time.';
    }
    
    modal.classList.add('show');
    addLog('system', winner === HUMAN ? '🏆 YOU WIN!' : '🤖 AI WINS!');
}

// ── Eval Lens — Level 2: Review modal ──────────────────────
async function openReviewFromWin() {
    // Server may have already started the next game's id by the time the
    // user clicks; reviewedGameId pins the game we're reviewing. If we
    // didn't capture one before reset, fall back to /latest.
    const gameId = reviewedGameId || currentGameId;
    document.getElementById('winModal').classList.remove('show');
    await fetchAndRenderReview(gameId);
}

async function fetchAndRenderReview(gameId) {
    const list = document.getElementById('reviewList');
    list.innerHTML = '<p class="review-loading">Loading…</p>';
    document.getElementById('reviewModal').classList.add('show');

    let url;
    if (gameId) {
        url = `/api/review_decisions/${encodeURIComponent(gameId)}?n=5`;
    } else {
        url = '/api/review_decisions/latest?n=5';
    }
    let payload;
    try {
        const res = await fetch(url);
        payload = await res.json();
    } catch (e) {
        list.innerHTML = '<p class="review-error">Could not load decisions.</p>';
        return;
    }

    const decisions = payload.decisions || [];
    if (decisions.length === 0) {
        list.innerHTML = '<p class="review-empty">No decisions logged for this game.</p>';
        return;
    }

    // Pin the game_id we're labeling against
    const labelGameId = payload.game_id;
    list.innerHTML = '';
    decisions.forEach(d => {
        list.appendChild(renderReviewCard(labelGameId, d));
    });
}

function renderReviewCard(gameId, d) {
    const card = document.createElement('div');
    card.className = 'review-card';
    card.dataset.decisionId = d.decision_id;

    const policy = d.v12_policy || [];
    const argmax = d.v12_argmax;
    const human = d.human_token;
    const winPct = d.v12_win_prob != null ? Math.round(d.v12_win_prob * 100) : null;

    const tokenLine = (player, label) => {
        const positions = (d.positions && d.positions[String(player)]) || [];
        const pretty = positions.map(p => {
            if (p === -1) return 'base';
            if (p === 99) return 'HOME';
            if (p > 50) return `H${p - 50}`;
            return String(p);
        }).join(', ');
        return `<div class="review-tokens"><span class="review-tokens-label">${label}</span><code>[${pretty}]</code></div>`;
    };

    const probRow = (i) => {
        const pct = ((policy[i] || 0) * 100).toFixed(1);
        const isAi = i === argmax;
        const isHuman = i === human;
        const isLegal = (d.legal_tokens || []).includes(i);
        const cls = [
            isAi ? 'ai-pick' : '',
            isHuman ? 'human-pick' : '',
            !isLegal ? 'illegal' : '',
        ].filter(Boolean).join(' ');
        const tag = [
            isAi ? '<span class="tag ai">AI</span>' : '',
            isHuman ? '<span class="tag you">You</span>' : '',
        ].join('');
        return `<div class="review-prob ${cls}">
            <span class="review-prob-label">T${i}</span>
            <span class="review-prob-bar"><span class="review-prob-fill" style="width:${pct}%"></span></span>
            <span class="review-prob-pct">${pct}%</span>
            <span class="review-prob-tags">${tag}</span>
        </div>`;
    };

    const probsHtml = [0, 1, 2, 3].map(probRow).join('');

    card.innerHTML = `
        <div class="review-header">
            <span class="review-move">Move #${d.move_count ?? '—'}</span>
            <span class="review-dice">🎲 ${d.dice}</span>
            ${winPct != null ? `<span class="review-win">AI win: ${winPct}%</span>` : ''}
            <span class="review-agree ${d.agree ? 'agree' : 'disagree'}">${d.agree ? 'agreed' : 'disagreed'}</span>
        </div>
        ${tokenLine(0, 'You')}
        ${tokenLine(2, 'AI')}
        <div class="review-probs">${probsHtml}</div>
        <div class="review-actions">
            <button class="rate-btn" data-label="v12_right">AI was right</button>
            <button class="rate-btn" data-label="human_right">I was right</button>
            <button class="rate-btn" data-label="either">Either is fine</button>
            <button class="rate-btn" data-label="both_bad">Both bad</button>
        </div>
        <div class="review-rated-tag" hidden>✓ Rated</div>
    `;

    card.querySelectorAll('.rate-btn').forEach(btn => {
        btn.addEventListener('click', () => submitRating(gameId, d.decision_id, btn.dataset.label, card));
    });

    return card;
}

async function submitRating(gameId, decisionId, label, cardEl) {
    if (!gameId) return;
    cardEl.querySelectorAll('.rate-btn').forEach(b => b.disabled = true);
    try {
        const res = await fetch('/api/submit_rating', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_id: gameId, decision_id: decisionId, label }),
        });
        const data = await res.json();
        if (data.ok) {
            cardEl.classList.add('review-card-labeled');
            cardEl.dataset.label = label;
            const tag = cardEl.querySelector('.review-rated-tag');
            if (tag) {
                tag.hidden = false;
                tag.textContent = `✓ Rated: ${labelText(label)}`;
            }
        } else {
            // Re-enable on failure
            cardEl.querySelectorAll('.rate-btn').forEach(b => b.disabled = false);
            console.warn('Rating failed:', data.error);
        }
    } catch (e) {
        cardEl.querySelectorAll('.rate-btn').forEach(b => b.disabled = false);
        console.warn('Rating error:', e);
    }
}

function labelText(label) {
    return ({
        v12_right: 'AI was right',
        human_right: 'I was right',
        either: 'Either is fine',
        both_bad: 'Both bad',
    })[label] || label;
}

function closeReview() {
    document.getElementById('reviewModal').classList.remove('show');
}

function finishReview() {
    document.getElementById('reviewModal').classList.remove('show');
    newGame();
}

// ── Utilities ───────────────────────────────────────────────
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Boot ────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', init);
