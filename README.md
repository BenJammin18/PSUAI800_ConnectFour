# PSUAI800_ConnectFour
## Overview

PSUAI800_ConnectFour is a two-player Connect Four style environment designed for AI experimentation. One player is a human/query player and the other is a programmatic agent. After each human move the program dynamically selects the best strategy for the current board state and plays the next move.

## Goals

- Dynamically choose the agent strategy that maximizes win probability against the current opponent at each state.
- Support multiple AI techniques and evaluation modes (Human vs AI, AI vs AI tournaments).
- Provide configurable board size and gravity behavior.

## Gameplay & Rules

- Default board: 8 × 8 (configurable n × m).
- Gravity mode (default): players choose a column; tokens fall to the lowest available cell in that column.
- Zero-gravity mode: players may place a token in any empty cell.
- Win/lose/draw determined by standard connect-four rules adapted to board dimensions.
- Rewards: +1 win, –1 loss, 0 draw.

## State & Action Space

- State: current board configuration. State space grows exponentially with board size and when zero-gravity is enabled.
- Actions:
    - Gravity mode: select a column.
    - Zero-gravity mode: select any empty cell.
- Transition model: deterministic placement; gravity applies if enabled.

## AI Agents

- Minimax with alpha–beta pruning and domain heuristics.
- Monte Carlo Tree Search (MCTS) with UCT rollouts.
- Optional Q-learning agent using state features and self-play for training.

Agent selection: after each human/query move, the system evaluates available strategies (using estimated win probability or evaluation scores) and selects the policy with the highest expected success for the new state.

## Modes

- Human vs AI: human plays against the dynamically switching agent.
- AI vs AI: run tournaments or head-to-head comparisons between strategies.
- Training: self-play loops for Q-learning or policy tuning.

## Configuration

- Board size: customizable n × m (default 8 × 8).
- Gravity: enabled/disabled.
- Agent pool: enable/disable specific strategies (Minimax, MCTS, Q-learning).
- Evaluation budget: rollout count, search depth, time limits for each agent.

## Running & Evaluation (example)

1. Configure board size and gravity in config file or CLI flags.
2. Choose mode: human-vs-ai or ai-vs-ai.
3. Start a match; after each human move the system selects an agent and computes the response.
4. For tournaments, log results and compute win/loss/draw statistics.

(Implementation-specific commands and examples should be added to the repository's usage docs or scripts.)

## Development & Contribution

- Structure code to separate game logic, agent implementations, and orchestration/selection logic.
- Add new strategies to the agent pool by implementing the agent interface and adding evaluation hooks.
- Tests: unit tests for game rules and integration tests for agent behavior.

## License
## License

This project is licensed under the GNU General Public License version 3 (GPL-3.0-or-later).

- SPDX: GPL-3.0-or-later
- Copyright (c) YEAR Your Name
- This software is free: you can redistribute it and/or modify it under the terms of the GNU GPL as published by the Free Software Foundation.
- This program is distributed WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the full license for details.

Full text: https://www.gnu.org/licenses/gpl-3.0.en.html

(Replace YEAR and Your Name as appropriate.)



