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
    legalMoves = [];
    awaitingMove = false;
    isProcessing = false;
    
    // Clear UI
    document.getElementById('winModal').classList.remove('show');
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
    
    // Wait for animation
    await sleep(500);
    clearInterval(cycleInterval);
    dice.classList.remove('rolling');
    
    document.getElementById('diceValue').textContent = data.dice_roll || '?';
    
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
        showMessage(`Rolled ${data.dice_roll} — no legal moves`);
        addLog('human', `Rolled ${data.dice_roll} — no moves available.`);
        isProcessing = false;
        renderState();
        await sleep(800);
        await doAITurn();
        return;
    }
    
    legalMoves = data.legal_moves || [];
    addLog('human', `Rolled ${data.dice_roll}`);
    
    if (legalMoves.length === 1) {
        // Auto-play single legal move
        showMessage(`Rolled ${data.dice_roll} — auto-moving token ${legalMoves[0]}`);
        isProcessing = false;
        await selectToken(legalMoves[0]);
        return;
    }
    
    showMessage(`Rolled ${data.dice_roll} — click a highlighted token`);
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
    
    const res = await fetch('/api/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: tokenIndex }),
    });
    const data = await res.json();
    gameState = data;
    
    const lastMove = data.last_move;
    if (lastMove) {
        let msg = `Moved token ${lastMove.token}: ${lastMove.from_pos}→${lastMove.to_pos}`;
        if (lastMove.captured) msg += ' ⚔️ CAPTURE!';
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
    
    document.getElementById('aiThinking').classList.remove('show');
    
    // Show AI's dice roll — pause so user can see it
    const aiRoll = data.ai_roll || data.dice_roll;
    if (aiRoll) {
        document.getElementById('diceValue').textContent = aiRoll;
        showMessage(`AI rolled ${aiRoll}`);
    }
    await sleep(1200);  // Pause to show dice result
    
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

// ── Utilities ───────────────────────────────────────────────
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Boot ────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', init);
