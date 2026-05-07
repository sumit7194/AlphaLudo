// Disagreement review UI

let LAYOUT = null;
let ALL_ROWS = [];
let FILTERED = [];
let IDX = 0;
let DRAFT_COMMENT = '';
let CURRENT_MARK = null;

const $ = id => document.getElementById(id);

async function fetchJSON(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return await r.json();
}

async function init() {
  LAYOUT = await fetchJSON('/api/layout');
  const data = await fetchJSON('/api/disagreements');
  ALL_ROWS = data.rows;
  $('total').textContent = data.count;

  buildBoardCells();
  applyFilters();
  attachEvents();
  renderCurrent();
}

function buildBoardCells() {
  const board = $('board');
  board.innerHTML = '';
  const pathSet = new Set(LAYOUT.path_squares.map(([r, c]) => `${r},${c}`));
  const safeSet = new Set(LAYOUT.safe_squares.map(([r, c]) => `${r},${c}`));
  const hr0 = new Set(LAYOUT.home_runs['0'].map(([r, c]) => `${r},${c}`));
  const hr2 = new Set(LAYOUT.home_runs['2'].map(([r, c]) => `${r},${c}`));
  const base0 = new Set(LAYOUT.bases['0'].map(([r, c]) => `${r},${c}`));
  const base2 = new Set(LAYOUT.bases['2'].map(([r, c]) => `${r},${c}`));
  const sp0 = `${LAYOUT.spawn_squares['0'][0]},${LAYOUT.spawn_squares['0'][1]}`;
  const sp2 = `${LAYOUT.spawn_squares['2'][0]},${LAYOUT.spawn_squares['2'][1]}`;
  const home = `${LAYOUT.home_center[0]},${LAYOUT.home_center[1]}`;

  for (let r = 0; r < 15; r++) {
    for (let c = 0; c < 15; c++) {
      const key = `${r},${c}`;
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.r = r; cell.dataset.c = c;
      if (pathSet.has(key)) cell.classList.add('path');
      if (safeSet.has(key)) cell.classList.add('safe');
      if (hr0.has(key))     cell.classList.add('home-run-0');
      if (hr2.has(key))     cell.classList.add('home-run-2');
      if (key === home)     cell.classList.add('home-center');
      if (base0.has(key))   cell.classList.add('base-0');
      if (base2.has(key))   cell.classList.add('base-2');
      if (key === sp0)      cell.classList.add('spawn-0');
      if (key === sp2)      cell.classList.add('spawn-2');
      cell.style.gridColumn = c + 1;
      cell.style.gridRow = r + 1;
      board.appendChild(cell);
    }
  }
}

function clearTokens() {
  document.querySelectorAll('.token, .token-stack').forEach(t => t.remove());
}

function placeTokens(row) {
  clearTokens();
  const cp = row.current_player;
  const opp = (cp + 2) % 4;
  const humanPick = row.human_token;
  const aiPick = row.v12_argmax;

  const place = (player, posList, klass) => {
    // Group tokens by cell coord to render stacks
    const byCell = new Map();
    posList.forEach((entry, t) => {
      const [r, c] = entry.coord;
      const k = `${r},${c}`;
      if (!byCell.has(k)) byCell.set(k, []);
      byCell.get(k).push({ tok: t, pos: entry.pos });
    });
    for (const [k, items] of byCell.entries()) {
      const [r, c] = k.split(',').map(Number);
      const cell = document.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
      if (!cell) continue;
      // Place a single representative token; show count badge if >1
      const tok = document.createElement('div');
      tok.className = `token ${klass}`;
      const ts = items.map(i => `T${i.tok}`).join(',');
      tok.textContent = items.length === 1 ? `T${items[0].tok}` : `${items.length}`;
      tok.title = `${ts} @ pos ${items[0].pos}`;
      // Highlight if human's pick or ai's pick is in this stack (player == cp only)
      if (player === cp) {
        const tokIds = items.map(i => i.tok);
        if (tokIds.includes(humanPick)) tok.classList.add('picked-h');
        if (tokIds.includes(aiPick))    tok.classList.add('picked-a');
      }
      cell.appendChild(tok);
      if (items.length > 1) {
        const stack = document.createElement('span');
        stack.className = 'token-stack';
        stack.textContent = `×${items.length}`;
        cell.appendChild(stack);
      }
    }
  };

  // current player's tokens (always render as "human" color = green for cp)
  // since all our games are P0=human=green, we render cp=0 as human, cp=2 as ai class color
  const cpKlass = cp === 0 ? 'h' : 'a';
  const oppKlass = cp === 0 ? 'a' : 'h';
  place(cp, row._coords[String(cp)], cpKlass);
  place(opp, row._coords[String(opp)], oppKlass);
}

function renderProbs(row) {
  const div = $('probs');
  div.innerHTML = '';
  const policy = row.v12_policy || [0, 0, 0, 0];
  const legal = new Set(row.legal_tokens || []);
  const humanPick = row.human_token;
  const aiPick = row.v12_argmax;
  const cp = row.current_player;
  const positions = row._coords[String(cp)] || [];
  for (let t = 0; t < 4; t++) {
    const p = policy[t] || 0;
    const pct = (p * 100).toFixed(1);
    const r = document.createElement('div');
    r.className = 'prob-row';
    if (!legal.has(t)) r.classList.add('illegal');
    if (t === humanPick && t === aiPick) r.classList.add('picked-both');
    else if (t === humanPick) r.classList.add('picked-h');
    else if (t === aiPick) r.classList.add('picked-a');
    const pos = positions[t]?.pos;
    const posLabel = pos === -1 ? 'B' : pos === 99 ? 'H' : String(pos);
    r.innerHTML = `
      <span class="lab">T${t}</span>
      <span class="bar"><span class="fill" style="width:${pct}%"></span></span>
      <span class="pct">${pct}% <small>@${posLabel}</small></span>
    `;
    div.appendChild(r);
  }
}

function fmtCoords(coords) {
  return coords.map(c => c.pos === -1 ? 'B' : c.pos === 99 ? 'H' : String(c.pos)).join(',');
}

function closeAnnot() {
  const card = $('annotCard');
  card.classList.remove('open');
  $('btnAnnotToggle').classList.remove('is-open');
}

function renderCurrent() {
  if (FILTERED.length === 0) {
    $('idx').textContent = '0';
    $('meta').textContent = 'no rows match the current filter';
    clearTokens();
    return;
  }
  IDX = Math.max(0, Math.min(IDX, FILTERED.length - 1));
  const row = FILTERED[IDX];
  $('idx').textContent = String(IDX + 1);

  const ann = row._annotation || {};
  const annLabel = ann.mark ? ` · marked: ${ann.mark}` : '';
  $('meta').textContent = `interest_score=${(row.interest_score ?? 0).toFixed(3)} · KL=${(row.kl_v12_to_human ?? 0).toFixed(3)}${annLabel}`;

  $('kvGame').textContent = (row.game_id || '').slice(-12);
  $('kvMove').textContent = row.move_count ?? '–';
  $('kvDice').textContent = row.dice ?? '–';
  $('kvCp').textContent = `P${row.current_player}`;
  $('kvWin').textContent = (row.v12_win_prob ?? 0).toFixed(3);
  $('kvMoves').textContent = (row.v12_moves_remaining ?? 0).toFixed(1);
  $('kvLegal').textContent = (row.legal_tokens || []).map(t => `T${t}`).join(', ');
  $('kvHumanPick').textContent = `T${row.human_token}`;
  $('kvAiPick').textContent = `T${row.v12_argmax}`;
  $('kvKL').textContent = (row.kl_v12_to_human ?? 0).toFixed(3);

  placeTokens(row);
  renderProbs(row);

  // Annotation panel
  $('comment').value = ann.comment || '';
  CURRENT_MARK = ann.mark || null;
  document.querySelectorAll('.mark').forEach(b => b.classList.remove('active'));
  if (CURRENT_MARK) {
    const map = { important: 'btnImportant', normal: 'btnNormal', dismiss: 'btnDismiss' };
    $(map[CURRENT_MARK])?.classList.add('active');
  }
  $('annotMeta').textContent = ann.ts ? `last saved: ${ann.ts}` : 'no annotation yet';

  // Color the 💬 toggle gold if this row already has any annotation, so the
  // user can see at a glance whether to open it.
  const hasAnnot = !!ann.mark || (ann.comment && ann.comment.length > 0);
  $('btnAnnotToggle').classList.toggle('has-annot', hasAnnot);
  $('btnAnnotToggle').textContent = hasAnnot ? '💬•' : '💬';
}

function applyFilters() {
  const hideD = $('hideDismissed').checked;
  const fm = $('filterMark').value;
  const sortBy = $('sortBy').value;
  FILTERED = ALL_ROWS.filter(r => {
    const mark = r._annotation?.mark || null;
    if (hideD && mark === 'dismiss' && fm !== 'dismiss') return false;
    if (fm === 'all') return true;
    if (fm === 'unmarked') return mark === null;
    return mark === fm;
  });
  if (sortBy === 'interest') FILTERED.sort((a, b) => (b.interest_score || 0) - (a.interest_score || 0));
  else if (sortBy === 'kl')  FILTERED.sort((a, b) => (b.kl_v12_to_human || 0) - (a.kl_v12_to_human || 0));
  else if (sortBy === 'game_newest') {
    // Game IDs encode timestamp (e.g. g_20260502_201821_xxxxxx) so string
    // compare DESC = newest first. Within game: move_count ascending (replay order).
    FILTERED.sort((a, b) =>
      (b.game_id || '').localeCompare(a.game_id || '') ||
      (a.move_count || 0) - (b.move_count || 0));
  }
  else if (sortBy === 'game_oldest') {
    FILTERED.sort((a, b) =>
      (a.game_id || '').localeCompare(b.game_id || '') ||
      (a.move_count || 0) - (b.move_count || 0));
  }
  $('total').textContent = FILTERED.length;
  $('filterInfo').textContent = `(of ${ALL_ROWS.length} total)`;
  IDX = 0;
}

async function saveAnnotation(mark) {
  const row = FILTERED[IDX];
  if (!row) return;
  const comment = $('comment').value;
  const body = {
    game_id: row.game_id,
    decision_id: row.decision_id,
    mark, comment,
  };
  try {
    const result = await fetchJSON('/api/annotate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    // Update local copy
    row._annotation = { mark, comment, ts: result.saved.ts || new Date().toISOString() };
    renderCurrent();
  } catch (e) {
    alert('Save failed: ' + e.message);
  }
}

function attachEvents() {
  $('btnPrev').onclick = () => { if (IDX > 0) IDX--; renderCurrent(); closeAnnot(); };
  $('btnNext').onclick = () => { if (IDX < FILTERED.length - 1) IDX++; renderCurrent(); closeAnnot(); };
  $('btnImportant').onclick = () => saveAnnotation('important');
  $('btnNormal').onclick    = () => saveAnnotation('normal');
  $('btnDismiss').onclick   = () => saveAnnotation('dismiss');
  $('btnAnnotToggle').onclick = () => {
    const card = $('annotCard');
    const isOpen = card.classList.toggle('open');
    $('btnAnnotToggle').classList.toggle('is-open', isOpen);
    if (isOpen) $('comment').focus();
  };
  $('hideDismissed').onchange = () => { applyFilters(); renderCurrent(); };
  $('filterMark').onchange    = () => { applyFilters(); renderCurrent(); };
  $('sortBy').onchange        = () => { applyFilters(); renderCurrent(); };
  // Comment autosave: debounced per-keystroke (1s after last keypress)
  // PLUS save-on-blur as a backup for navigation.
  let commentTimer = null;
  const debouncedSave = () => {
    if (commentTimer) clearTimeout(commentTimer);
    commentTimer = setTimeout(() => {
      const row = FILTERED[IDX];
      if (!row) return;
      const newC = $('comment').value;
      if (newC !== (row._annotation?.comment || '')) {
        saveAnnotation(row._annotation?.mark || null);
      }
    }, 1000);
  };
  $('comment').addEventListener('input', debouncedSave);
  $('comment').addEventListener('blur', () => {
    if (commentTimer) clearTimeout(commentTimer);
    const row = FILTERED[IDX];
    if (!row) return;
    const newC = $('comment').value;
    if (newC !== (row._annotation?.comment || '')) {
      saveAnnotation(row._annotation?.mark || null);
    }
  });
  // Keyboard nav
  document.addEventListener('keydown', (e) => {
    if (document.activeElement?.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft')  { $('btnPrev').click(); }
    if (e.key === 'ArrowRight') { $('btnNext').click(); }
    if (e.key === 'i')          { $('btnImportant').click(); }
    if (e.key === 'n')          { $('btnNormal').click(); }
    if (e.key === 'd')          { $('btnDismiss').click(); }
  });
}

window.addEventListener('DOMContentLoaded', init);
