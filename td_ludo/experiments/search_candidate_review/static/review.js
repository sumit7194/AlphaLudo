// Search candidate review UI

let LAYOUT = null;
let ALL_ROWS = [];
let FILTERED = [];
let IDX = 0;
let CURRENT_MARK = null;
let CURRENT_DEPTH = null;

const $ = id => document.getElementById(id);

async function fetchJSON(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return await r.json();
}

async function init() {
  LAYOUT = await fetchJSON('/api/layout');
  const data = await fetchJSON('/api/candidates');
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
  const aiPick = row.action_chosen;
  // model's argmax: highest prob token
  const policy = [row.policy_t0, row.policy_t1, row.policy_t2, row.policy_t3];
  const argmax = policy.indexOf(Math.max(...policy));

  const place = (player, posList, klass, isActive) => {
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
      const tok = document.createElement('div');
      tok.className = `token ${klass}`;
      tok.textContent = items.length === 1 ? `T${items[0].tok}` : `${items.length}`;
      tok.title = items.map(i => `T${i.tok}@${i.pos}`).join(', ');
      if (isActive) {
        const tokIds = items.map(i => i.tok);
        if (tokIds.includes(argmax)) tok.classList.add('picked-h');
        if (tokIds.includes(aiPick) && aiPick !== argmax) tok.classList.add('picked-a');
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

  const cpKlass = cp === 0 ? 'h' : 'a';
  const oppKlass = cp === 0 ? 'a' : 'h';
  place(cp, row._coords[String(cp)], cpKlass, true);
  place(opp, row._coords[String(opp)], oppKlass, false);
}

function renderProbs(row) {
  const div = $('probs');
  div.innerHTML = '';
  const policy = [row.policy_t0, row.policy_t1, row.policy_t2, row.policy_t3];
  const legal = new Set();
  for (let i = 0; i < row.legal_mask.length; i++) {
    if (row.legal_mask[i] === '1') legal.add(i);
  }
  const argmax = policy.indexOf(Math.max(...policy));
  const cp = row.current_player;
  const positions = row._coords[String(cp)] || [];
  for (let t = 0; t < 4; t++) {
    const p = policy[t] || 0;
    const pct = (p * 100).toFixed(1);
    const r = document.createElement('div');
    r.className = 'prob-row';
    if (!legal.has(t)) r.classList.add('illegal');
    if (t === argmax)  r.classList.add('picked-h');
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
  const annLabel = ann.mark ? ` · marked: ${ann.mark}${ann.depth ? ` (depth ${ann.depth})` : ''}` : '';
  $('meta').textContent = `${row.phase} · ${row.uncertainty} · max_prob ${(row._max_prob ?? 0).toFixed(3)}${annLabel}`;

  // Bucket row
  const tac = [];
  if (row.capture_available) tac.push('<span class="tac-flag cap">CAPTURE</span>');
  if (row.danger_present)    tac.push('<span class="tac-flag dng">DANGER</span>');
  $('bucketRow').innerHTML = `
    <span class="bucket-tag">${row._bucket || 'unbucketed'}</span>
    ${tac.join('')}
    <span class="meta" style="margin-left:6px;">phase: <b>${row.phase}</b> · uncertainty: <b>${row.uncertainty}</b> · win-pos: <b>${row.win_bucket}</b></span>
  `;

  $('kvGame').textContent = String(row.game_id ?? '–');
  $('kvMove').textContent = row.move_idx ?? '–';
  $('kvDice').textContent = row.dice ?? '–';
  $('kvCp').textContent = `P${row.current_player}`;
  $('kvWin').textContent = (row.win_prob ?? 0).toFixed(3);
  $('kvMoves').textContent = (row.moves_remaining ?? 0).toFixed(1);
  const legal = [];
  for (let i = 0; i < row.legal_mask.length; i++) if (row.legal_mask[i] === '1') legal.push(`T${i}`);
  $('kvLegal').textContent = legal.join(', ');
  $('kvWinner').textContent = row.winner == null ? '?' : (row.winner === row.current_player ? 'cp won ✓' : (row.winner === -1 ? 'timeout' : 'cp lost ✗'));

  const policy = [row.policy_t0, row.policy_t1, row.policy_t2, row.policy_t3];
  const argmax = policy.indexOf(Math.max(...policy));
  $('kvActed').textContent = `T${argmax} (sampled T${row.action_chosen})`;
  $('kvMaxProb').textContent = (Math.max(...policy)).toFixed(3);

  placeTokens(row);
  renderProbs(row);

  // Annotation panel
  $('comment').value = ann.comment || '';
  CURRENT_MARK = ann.mark || null;
  CURRENT_DEPTH = ann.depth || null;
  document.querySelectorAll('.mark').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
  if (CURRENT_MARK) {
    const map = { critical: 'btnCritical', helpful: 'btnHelpful', unnecessary: 'btnUnnecessary', dismiss: 'btnDismiss' };
    $(map[CURRENT_MARK])?.classList.add('active');
  }
  if (CURRENT_DEPTH) {
    document.querySelector(`.depth-btn[data-depth="${CURRENT_DEPTH}"]`)?.classList.add('active');
  }
  $('annotMeta').textContent = ann.ts ? `last saved: ${ann.ts}` : 'no annotation yet';
  const hasAnnot = !!ann.mark || (ann.comment && ann.comment.length > 0);
  $('btnAnnotToggle').classList.toggle('has-annot', hasAnnot);
  $('btnAnnotToggle').textContent = hasAnnot ? '💬•' : '💬';
}

function applyFilters() {
  const hideD = $('hideDismissed').checked;
  const fm = $('filterMark').value;
  const fp = $('filterPhase').value;
  const ft = $('filterTac').value;
  FILTERED = ALL_ROWS.filter(r => {
    const mark = r._annotation?.mark || null;
    if (hideD && mark === 'dismiss' && fm !== 'dismiss') return false;
    if (fm === 'unmarked' && mark !== null) return false;
    if (fm !== 'all' && fm !== 'unmarked' && mark !== fm) return false;
    if (fp !== 'all' && r.phase !== fp) return false;
    if (ft === 'capture' && !(r.capture_available && !r.danger_present)) return false;
    if (ft === 'danger'  && !(r.danger_present  && !r.capture_available)) return false;
    if (ft === 'both'    && !(r.capture_available && r.danger_present)) return false;
    if (ft === 'none'    && (r.capture_available || r.danger_present)) return false;
    return true;
  });
  $('total').textContent = FILTERED.length;
  $('filterInfo').textContent = `(of ${ALL_ROWS.length})`;
  IDX = 0;
}

async function saveAnnotation(mark, depth) {
  const row = FILTERED[IDX];
  if (!row) return;
  const comment = $('comment').value;
  const body = {
    id: row.id,
    mark: mark === undefined ? CURRENT_MARK : mark,
    depth: depth === undefined ? CURRENT_DEPTH : depth,
    comment,
  };
  try {
    const result = await fetchJSON('/api/annotate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    row._annotation = { mark: body.mark, depth: body.depth, comment, ts: result.saved.ts };
    renderCurrent();
  } catch (e) {
    alert('Save failed: ' + e.message);
  }
}

function attachEvents() {
  $('btnPrev').onclick = () => { if (IDX > 0) IDX--; renderCurrent(); closeAnnot(); };
  $('btnNext').onclick = () => { if (IDX < FILTERED.length - 1) IDX++; renderCurrent(); closeAnnot(); };
  $('btnCritical').onclick    = () => saveAnnotation('critical', undefined);
  $('btnHelpful').onclick     = () => saveAnnotation('helpful', undefined);
  $('btnUnnecessary').onclick = () => saveAnnotation('unnecessary', undefined);
  $('btnDismiss').onclick     = () => saveAnnotation('dismiss', undefined);
  document.querySelectorAll('.depth-btn').forEach(btn => {
    btn.onclick = () => saveAnnotation(undefined, parseInt(btn.dataset.depth, 10));
  });
  $('btnAnnotToggle').onclick = () => {
    const card = $('annotCard');
    const isOpen = card.classList.toggle('open');
    $('btnAnnotToggle').classList.toggle('is-open', isOpen);
    if (isOpen) $('comment').focus();
  };
  $('hideDismissed').onchange = () => { applyFilters(); renderCurrent(); };
  $('filterMark').onchange    = () => { applyFilters(); renderCurrent(); };
  $('filterPhase').onchange   = () => { applyFilters(); renderCurrent(); };
  $('filterTac').onchange     = () => { applyFilters(); renderCurrent(); };

  // Comment autosave (debounce 1s) + on-blur backup
  let commentTimer = null;
  const debouncedSave = () => {
    if (commentTimer) clearTimeout(commentTimer);
    commentTimer = setTimeout(() => {
      const row = FILTERED[IDX];
      if (!row) return;
      if ($('comment').value !== (row._annotation?.comment || '')) saveAnnotation();
    }, 1000);
  };
  $('comment').addEventListener('input', debouncedSave);
  $('comment').addEventListener('blur', () => {
    if (commentTimer) clearTimeout(commentTimer);
    const row = FILTERED[IDX];
    if (!row) return;
    if ($('comment').value !== (row._annotation?.comment || '')) saveAnnotation();
  });

  document.addEventListener('keydown', (e) => {
    if (document.activeElement?.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft')  { $('btnPrev').click(); }
    if (e.key === 'ArrowRight') { $('btnNext').click(); }
    if (e.key === 'c') { $('btnCritical').click(); }
    if (e.key === 'h') { $('btnHelpful').click(); }
    if (e.key === 'u') { $('btnUnnecessary').click(); }
    if (e.key === 'd') { $('btnDismiss').click(); }
    if (e.key === '1' || e.key === '2' || e.key === '3') {
      saveAnnotation(undefined, parseInt(e.key, 10));
    }
  });
}

window.addEventListener('DOMContentLoaded', init);
