# Engineering Principles & Working Mantra for AlphaLudo

## The Core Mantra
> **Never retreat to an inferior alternative at the first sign of a technical hurdle. Always investigate deeply, formulate first-principles solutions, try relentlessly, and NEVER abandon a superior path without failing multiple times across multiple approaches.**

---

## Non-Negotiable Standards of Rigor

1. **No Corner-Cutting:**
   - Do not substitute a weaker, simpler heuristic when the ambitious, mathematically superior approach (e.g., Multi-Agent MCTS, high-depth lookahead search) is what the project demands.
   - Strive for state-of-the-art results, not merely "convenient" results.

2. **First-Principles Problem Solving:**
   - If existing legacy code makes restrictive assumptions (e.g., 2-player zero-sum minimax in a 4-player game), do **not** abandon the algorithm. Redesign it properly from first principles (e.g., $\text{Max}^n$ multi-agent tree search with 4-dimensional payoff vectors).

3. **Performance & Systems Architecture:**
   - Heavy compute (game simulation, rollouts, multi-ply search trees, state encoders) belongs in native compiled code (Rust) with zero heap allocations and L1 cache residency.
   - Python is strictly a high-level ML coordinator; never let interpreted loops bottleneck data generation.

4. **Detached Background Execution:**
   - Long training and rollout runs must be launchable as fully detached background daemons (`nohup` / disown), with output logged to disk, so the user can close the IDE / Antigravity to free up Mac RAM without killing the training process.
   - Every training process must be cleanly interruptable (`touch stop`) and seamlessly resumable from metadata without data corruption.
