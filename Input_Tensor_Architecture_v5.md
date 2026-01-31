# AlphaLudo Input Tensor Architecture (v5)

## Overview

We have redesigned the Neural Network input from a simple 12-channel representation to a **rich 18-channel tensor**. This change aims to solve specific "blind spots" in the previous model where it struggled to understand stacking mechanics, race dynamics, and dice strategies.

### Shape: `(18, 15, 15)`
The input is a stack of 18 spatial planes of size 15x15 (the Ludo board size).

---

## Detailed Channel Breakdown

### Group 1: Piece Positions & Saturation (Channels 0-3)

| Channel | Feature | Logic | Justification |
| :--- | :--- | :--- | :--- |
| **0** | **Me** | `Count / 4.0` | **Stacking Awareness**: Previous binary inputs (0 or 1) couldn't distinguish between 1 piece and a stack of 4. Dividing by 4.0 gives a "density" value (0.25, 0.5, 0.75, 1.0), allowing the model to prioritize moving stacks or breaking them. |
| **1** | **Next Opp** | `Count / 4.0` | **Threat Assessment**: Knowing the density of the next opponent helps predict their defensive capability. |
| **2** | **Teammate** | `Count / 4.0` | **Cooperation**: In 2v2, knowing teammate stack density helps in forming blockades. |
| **3** | **Prev Opp** | `Count / 4.0` | **Chaser Awareness**: The previous opponent is the "hunter." High density behind you signals danger. |

### Group 2: Board Geometry (Channel 4)

| Channel | Feature | Logic | Justification |
| :--- | :--- | :--- | :--- |
| **4** | **Safe Zones** | Binary Mask (1.0 at stars) | **Spatial Context**: While the model *can* learn safe zones over time, providing them explicitly acts as a "coordinate system" anchor, speeding up training and ensuring it never "forgets" where safety lies. |

### Group 3: Global Context (Channels 5-7)

> **Note**: These are scalar values (single numbers) broadcasted to fill the entire 15x15 plane. This allows a Convolutional Network (which looks at local windows) to "see" global game state everywhere.

| Channel | Feature | Logic | Justification |
| :--- | :--- | :--- | :--- |
| **5** | **Score Diff** | `(MyScore - MaxOppScore) / 50.0` | **Winning/Losing State**: If positive, the model knows to play defensively (minimize risk). If negative, it knows to take risks (aggressive captures). Normalizing by 50 keeps gradients stable. |
| **6** | **My Locked** | `LockedCount / 4.0` | **Urgency**: How many pieces are stuck at home? High value = High urgency to roll a 6. |
| **7** | **Opp Locked** | `TotalLocked / 12.0` | **Opponent Vulnerability**: If opponents are locked, the board is safer to traverse. |

### Group 4: Relative Progress (Channels 8-10)

> **Why Relative?** Absolute progress (0 to 52) is less useful than knowing "Am I ahead of them?".

| Channel | Feature | Logic | Justification |
| :--- | :--- | :--- | :--- |
| **8** | **vs Next** | `(MyProg - NextProg) / 52.0` | **The Race**: Direct measure of who is winning the footrace to the finish. |
| **9** | **vs Teammate** | `(MyProg - TeamProg) / 52.0` | **Balancing**: Helps the model decide whether to help a lagging teammate or push a leading piece. |
| **10** | **vs Prev** | `(MyProg - PrevProg) / 52.0` | **Escape**: Distance from the chaser. |

### Group 5: Temporal History (Channel 11)

| Channel | Feature | Logic | Justification |
| :--- | :--- | :--- | :--- |
| **11** | **Ghost Trail** | `1.0` (Current), `0.5` (T-1), `0.25` (T-2) | **Motion Perception**: Standard ConvNets are static frame-by-frame. Adding a "fading trail" of the last 2 moves gives the model a sense of **momentum** and immediate history (e.g., "This piece just captured" or "This piece is running away") without the cost of a Recurrent Neural Network (LSTM). |

### Group 6: Dice Roll (Channels 12-17)

> **Why One-Hot?** We represent the dice roll as 6 separate binary channels rather than a single channel with value `Roll/6.0`.

| Channel | Feature | Logic | Justification |
| :--- | :--- | :--- | :--- |
| **12** | **Roll=1** | All 1.0 if Roll=1, else 0 | **Non-Linear Value**: In Ludo, a 6 is not just "6x better" than a 1. A 6 has special properties (exit home, roll again). A 1 is crucial for small steps into safety. |
| **13** | **Roll=2** | All 1.0 if Roll=2, else 0 | **Distinct Strategies**: One-hot encoding allows the network to learn completely distinct filters/strategies for each specific dice number, rather than trying to interpolate a continuous value. |
| **14** | **Roll=3** | All 1.0 if Roll=3, else 0 | |
| **15** | **Roll=4** | All 1.0 if Roll=4, else 0 | |
| **16** | **Roll=5** | All 1.0 if Roll=5, else 0 | |
| **17** | **Roll=6** | All 1.0 if Roll=6, else 0 | **The Power Roll**: Activates specific "opening" neurons in the network. |

---

## Summary of Improvements

1.  **Fixed "Stack Blindness"**: By using density (0.25-1.0) instead of binary (0/1), the model can now see stacks.
2.  **Added "Race Intuition"**: Relative progress channels explicitly tell the model if it is winning or losing the race, rather than forcing it to calculate distances from raw coordinates.
3.  **Removed "Magnitude Fallacy"**: One-hot dice prevents the model from assuming linear relationships between dice rolls.
4.  **Added "Short-Term Memory"**: Ghost trails provide cheap temporal context.
