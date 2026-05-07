# V12.2 Gameplay Observations — Live Play Notes

Running log of model weaknesses spotted while playing against V12.2 (pre-search,
post-encoder-fix) on the play server. Each entry: situation, AI's choice,
expected better choice, hypothesis on why the model failed.

These observations inform future training experiments — they are **NOT**
ground-truth verdicts (the model may be making decisions on positional
considerations the user is missing). But high-confidence-disagreement moves
are worth investigating, especially when the model leaves a more advanced
own token in danger.

---

## Format

For each observation, capture:

- **Game ID** (from server log) and approximate **move #**
- **Dice roll** and **whose turn**
- **Board summary** — relevant positions (own + opp tokens that matter)
- **AI's pick** with policy distribution if available
- **What I'd have done** and rough confidence (e.g. "70-30 chase / unlock")
- **Hypothesis** — danger blindness, capture obsession, T0/T1 spawn bias, etc.
- **Severity** — was it actually losing, or marginal?

---

## Observations

### Obs 1 — chase with T1 instead of running endangered T0 (2026-05-02 ~20:04)

- **Situation:** AI rolled 4. Multiple legal options.
- **AI's pick:** T1, with **high confidence** (~higher than other tokens).
- **What was lost:** T0 (more advanced) was already in danger and would have
  been a higher-priority move (either run forward or to safety).
- **Why this matters:** AI prioritized aggression with a less-advanced token
  while ignoring an endangered own token that was further along the track.
  Standard heuristic in Ludo: protect your most-advanced token first because
  losing it costs the most progress.
- **Probable cause:**
  1. **Danger blindness defect** (mech-interp confirmed: V12.2's
     `leading_token_in_danger` linear probe hits 95.4% accuracy = baseline.
     Probe can't beat majority class.) Model literally doesn't see the danger.
  2. **Capture-map dominance** (Ch 22 KL=1.67 in capture states): when
     a capture is available, the model attends almost exclusively to that
     channel and may ignore defensive considerations.
- **Severity:** medium-high — losing T0 from an advanced position usually
  costs ~30+ board squares of progress.
- **Suggested fix path:** add explicit auxiliary loss on
  `leading_token_in_danger` during RL (one of the V13.1 proposals).

### Obs 2 — human's choice: unlock T1 vs run advanced T0 (2026-05-02 ~20:05)

- **Situation:** Human rolled 6. Choice was to spawn T1 from base, OR run
  T0 (more advanced) which was also in danger.
- **Human's call:** judged ~30-70 / 40-60 toward unlocking T1 over running.
  Reasoning: getting T1 out gives optionality for future turns, while
  running T0 forward only delays the danger by one dice-roll window.
- **AI didn't decide here** (it was the human's turn) — but worth recording
  to compare V12.2's predicted policy distribution for this state once we
  have the "AI predicts your turn" panel showing.
- **What to check:** does V12.2's policy on this state weight T1 (spawn) at
  60%+ like the user did, or does it pick T0 (run away)? Disagreement
  reveals the model's defensive heuristic in action.

---

## Defects to watch for systematically

Based on mech-interp + earlier eval-lens results, expect to see these patterns:

| Defect | What to look for |
|---|---|
| **Danger blindness** | Endangered own tokens (esp. leading) ignored; AI prefers offense |
| **Capture obsession** | When Ch 22 fires (capture available), AI almost always takes it even if it leaves own tokens exposed |
| **T0/T1 spawn bias** | V12.2 prefers T0 to spawn first; T2/T3 spawn rarely chosen as opener |
| **Two-roll lookahead missing** | Doesn't pre-position to capture next turn (Ch 25 is dead in both global and conditional) |
| **No streak awareness** | Plays same token N times in a row even when other tokens are stuck (idle channels Ch 28-31 are dead) |
| **End-game closure** | Once 3 tokens are home, may not optimally close out the 4th |
| **Defensive blockades** | Doesn't create or break blockades strategically |

---

## How to capture future observations

1. **Use the "AI's prediction for YOUR turn" panel** (added to play server
   under human player card). It shows V12.2's full policy distribution for
   any decision YOU make.
2. **Note disagreements** — when V12.2's top pick differs from yours,
   consider why. If you can't justify your pick over V12.2's, V12.2 might
   be right and you're learning. If you can justify yours, write it up.
3. **Game logs** are saved at `play/decision_logs/game_<id>.log` — every
   AI decision includes its policy probabilities for reference.
4. **Decision log JSONL** at `play/decision_logs/decisions_<id>.jsonl`
   captures every human decision with V12.2's would-have-chosen + KL.
   Sort by `interest_score` (= max-policy × KL-to-human) to surface the
   highest-disagreement moves.

---

## TODO

- [ ] Add Obs 3+ as more games are played
- [ ] After enough observations (~10-20), categorize by defect type and
      build a "test set" of named scenarios
- [ ] Use the test set to evaluate V13.1 (or any future model) — does it
      fix specific defects or trade them for new ones?
