# Connect Four RL

A Connect Four game with a GUI, plus an AI opponent trained via self-play reinforcement learning against an alpha-beta pruned minimax search.

The AI is a policy/value neural network (`ConnectFourNet`) that learns to play by competing against a classical minimax engine of increasing search depth. Once trained, the network plays instantly (no search at inference time), while still having learned tactics like blocking and threat creation from its minimax opponent.

## How It Works

- **`board.py`** — Core game logic: a 6x7 `Board` class that handles moves, win detection (horizontal, vertical, and both diagonals), and draw detection via the `MoveResult` enum.
- **`minimax.py`** — A depth-limited minimax search with alpha-beta pruning and center-first move ordering, used both as a standalone opponent and as the training adversary for the neural network.
- **`network.py`** — Defines `ConnectFourNet`, the policy/value network that outputs move logits and a position-value estimate from an encoded board state.
- **`train.py`** — The training loop. The network plays repeated games against minimax opponents of increasing depth (a curriculum: depth 1 → 5 → 10), using epsilon-greedy exploration and a shaped reward signal (win/draw/block/threat) combined with an actor-critic-style policy and value loss. Trained weights are saved to `c4_net.pth`.
- **`main.py`** — A Tkinter GUI for playing Connect Four against the trained network. Choose to go first (Red) or second (Yellow), and the AI responds using the saved model.

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- Tkinter (included with most standard Python installations)

Install dependencies:

```bash
pip install torch
```

## Usage

### Play against the trained AI

A trained model checkpoint (`c4_net.pth`) is required in the project root. Launch the GUI with:

```bash
python main.py
```

Choose whether to play as Red (first) or Yellow (second), then click a column to drop a piece. The AI responds automatically.

### Train a new model

To train the network from scratch via self-play against minimax:

```bash
python train.py
```

This runs 170,000 episodes by default, with the minimax opponent's search depth increasing over the course of training (curriculum learning) and the exploration rate (epsilon) annealing over time. Progress, including estimated time remaining, is printed periodically. The resulting weights are saved to `c4_net.pth` in the project root, ready to be loaded by `main.py`.

Training duration and episode count can be adjusted by editing the `train()` call in `train.py`.

## Project Structure

```
connect-four-rl/
├── board.py       # Game state, move validation, win/draw detection
├── minimax.py      # Alpha-beta pruned minimax search
├── network.py      # Policy/value neural network definition
├── train.py         # Self-play training loop
└── main.py           # Tkinter GUI for human vs. AI play
```
