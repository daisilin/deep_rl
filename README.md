# Four-in-a-Row: Comparing Machine and Human Learning

This repository contains the implementation and analysis code for the research article ["Comparing Machine and Human Learning in a Planning Task of Intermediate Complexity"](https://escholarship.org/uc/item/8wm748d8), which studies how AI and humans learn to play the game of Four-in-a-Row.

## Overview

The project implements multiple approaches to playing Four-in-a-Row:
- AlphaZero-style deep reinforcement learning agents
- Best-First Tree Search (BFTS) variants
- Human-playable interface for data collection and comparison

### Key Features

- **Neural Network Architectures**
  - Standard AlphaZero-style policy and value networks
  - Policy-Value correlation tracking networks
  - Residual network variants with color tracking

- **Search Algorithms**
  - Monte Carlo Tree Search (MCTS)
  - Best-First Tree Search (BFTS)
  - Neural network guided tree search

- **Analysis Tools**
  - Game replay and visualization
  - Neural network activity analysis
  - Performance metrics computation
  - Human vs. AI comparison tools

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd four-in-a-row

# Install dependencies
pip install -r requirements.txt
```

## Key Components

### Neural Networks (`neural_net.py`)
- Base neural network implementation
- Policy and value head architectures
- Training and evaluation functions

### Game Logic (`beck_game.py`)
- Four-in-a-Row game rules
- Board state management
- Move validation and game end detection

### Search Algorithms (`bfts.py`, `mcts.py`)
- Tree search implementations
- Move selection strategies
- Position evaluation

### Player Interfaces
- Human player with GUI
- AI player implementations
- Random and greedy baseline players

## Usage

### Training an AI Agent

```python
from beck_game import BeckGame
from neural_net import NNetWrapper

# Initialize game and network
game = BeckGame(m=4, n=9, k=4)  # 4x9 board, 4-in-a-row to win
nnet = NNetWrapper(game)

# Train network
nnet.train(examples)  # examples: list of (board, policy, value) tuples
```

### Playing Against AI

```python
from players import HumanBeckPlayer, NNPlayer

# Initialize players
human_player = HumanBeckPlayer(game)
ai_player = NNPlayer(game, nnet)

# Play game
current_player = human_player
board = game.getInitBoard()

while not game.getGameEnded(board, 1):
    action = current_player.play(board)
    board, _ = game.getNextState(board, 1, action)
    current_player = ai_player if current_player == human_player else human_player
```

## Research Context

This repository supports research comparing how machines and humans learn to play Four-in-a-Row, a game of intermediate complexity between tic-tac-toe and chess. Key findings include:

- Analysis of different learning trajectories between AI and human players
- Comparison of search strategies 

For detailed methodology and results, please refer to the [research article](https://escholarship.org/uc/item/8wm748d8).

