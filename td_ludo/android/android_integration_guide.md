# AlphaLudo Android Integration Guide

This document provides the necessary information for the Android development team to integrate the PyTorch Mobile AI (`alphaludo_mobile.ptl`) into a native Android Ludo application.

## 1. PyTorch Mobile Setup

The model is exported as a PyTorch Mobile TorchScript Lite file (`.ptl`).

1. Add the PyTorch dependencies to `app/build.gradle`:
   ```gradle
   implementation 'org.pytorch:pytorch_android_lite:2.1.0'
   ```
2. Place `alphaludo_mobile.ptl` into your Android project's `app/src/main/assets/` folder.
3. Load the module in Kotlin using `LiteModuleLoader.load()`.

---

## 2. Neural Network Input Architecture (11-Channel Tensor)

The AI strictly expects a `Float32` tensor of shape **`[1, 11, 15, 15]`**.
- `1`: Batch size
- `11`: Feature channels (layers) representing the state of the game
- `15`: The physical width/height of the Ludo board

Before making a move, your Android game logic must generate a 15x15 grid of floats corresponding to the exact current game state, organized into 11 layers:

| Channel | Description | Value Logic |
|---|---|---|
| **Ch 0-3** | **AI Tokens (Identity)** | One channel for *each* of the AI's 4 tokens. Place a `1.0` exactly at the `(row, col)` where token $i$ is. Everything else `0.0`. |
| **Ch 4** | **Opponent Density** | Represents all opponent tokens. Add `0.25` for *every* opponent token on a tile. (e.g., 2 opponent tokens on one square = `0.50` there). |
| **Ch 5** | **Safe Zones** | Static map. Place a `1.0` on the 8 central stars and the 4 starting bases. (This never changes during the game). |
| **Ch 6** | **AI Home Path** | Static map. Place a `1.0` on the 6 coloured tiles leading directly into the AI's center "Home". |
| **Ch 7** | **Opponent Home Path**| Static map. Place a `1.0` on the 6 coloured tiles leading directly into the opponent's "Home". |
| **Ch 8** | **Score Differential** | **Broadcast** channel. Fill the entire 15x15 array with: `(AI_Tokens_Home - Opponent_Tokens_Home) / 4.0` |
| **Ch 9** | **AI Locked Bases** | **Broadcast** channel. Fill the entire 15x15 array with: `(Num AI tokens still in starting base) / 4.0` |
| **Ch 10** | **Opp Locked Bases** | **Broadcast** channel. Fill the entire 15x15 array with: `(Num Opponent tokens still in starting base) / 4.0` |

### Coordinate Mapping Warning:
The AI is trained from **Player 0's perspective**. 
When feeding the board state to the AI, **you must rotate the board** so that the AI thinks it is Player 0 (starting at the bottom-left base and traveling clockwise).

---

## 3. Network Output & Move Selection

The `module.forward()` call returns a Tuple. Extract the first element (the Policy logits):
```kotlin
val policyTensor = outputTuple[0].toTensor()
val policyScores = policyTensor.dataAsFloatArray // Size 4 Array
```

**CRITICAL RULE:** The AI model **does not know what the dice roll is**. The output is simply four scores, one for each token: `[score0, score1, score2, score3]`.

**Your Android logic must:**
1. Check which of the 4 AI tokens legally *can* move given the current dice roll.
2. From the subset of *legal tokens*, find the token with the highest `policyScore`.
3. Move that token.

---

## 4. Specific AlphaLudo Game Rules

The AI was trained against a very specific set of Ludo rules (written in C++). Your Android app must enforce these exact rules identically, otherwise the AI will make invalid assumptions and perform poorly.

### Board Configuration
* **Track Length:** 52 normal squares + 5 home run squares = 57 steps to Home.
* **Safe Stops:** 8 squares total. (The 4 starting bases, plus the 4 stars 8 steps ahead of each starting base).

### Movement Rules
* **Rolling a 6:** A 6 is required to spawn a token out of the base.
* **Bonus Turns:** A player gets to roll again IF:
   1. They roll a 6.
   2. They capture an opponent's token.
   3. They route a token successfully into Home.
* **Max Consecutive Sixes:** If a player rolls three 6s in a row, the turn is immediately forfeited (the third 6 is voided, and play passes to the next player).
* **Blockades (Optional):** In the current AI training environment, stack limit / blockades (2 tokens forming a wall) are disabled. Tokens can freely bypass stacks of opponents.

### Capturing
* If Token A ends its move on exactly the same square as opponent Token B, Token B is captured and sent back to its base.
* **Exception:** Captures **cannot** occur on Safe Stops (the 8 stars/starting bases). If Token A lands on a safe stop occupied by opponent Token B, they coexist on that square.
