import copy
import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from board import Board, MoveResult
from minimax import best_move, get_legal_moves
from network import ConnectFourNet


def encode_board(board, current_player):
    """Encode board as float tensor from current player's perspective."""
    flat = []
    for row in board.get_board():
        for cell in row:
            if cell is None:
                flat.append(0.0)
            elif cell == current_player:
                flat.append(1.0)
            else:
                flat.append(-1.0)
    return torch.tensor(flat, dtype=torch.float32)


def was_blocking_move(board, col, opponent):
    """Check if the move just played blocked the opponent from winning."""
    test_board = copy.deepcopy(board)

    # Undo the move by setting the top-most piece in the column back to None
    for r in range(5, -1, -1):
        if test_board.get_board()[r][col] is not None:
            test_board.get_board()[r][col] = None
            break

    # Check if the opponent could have won there
    result = test_board.make_move(opponent, col)
    return result == MoveResult.WIN


def get_reward(result, board, col, player):
    """Shaped reward: win, draw, block, or neutral."""
    if result == MoveResult.WIN:
        return 1.0
    if result == MoveResult.DRAW:
        return 0.3

    opponent = "Y" if player == "R" else "R"
    if was_blocking_move(board, col, opponent):
        return 0.5
    return 0.0


def play_game(net, epsilon=0.1):
    """
    Play one game where the Neural Network plays against the Minimax algorithm.
    Returns a list of (state_tensor, action, reward) tuples for the NN + winner.
    """
    board = Board()
    history = []
    current = "R"

    # Randomly assign the neural network to go first or second
    nn_player = random.choice(["R", "Y"])

    while True:
        legal_moves = get_legal_moves(board)

        if current == nn_player:
            state = encode_board(board, current)

            # Epsilon-greedy action selection
            if torch.rand(1).item() < epsilon:
                action = random.choice(legal_moves)
            else:
                logits, _ = net(state)
                # Network outputs 42 values, but we only have 7 valid columns.
                # Mask out all invalid moves (everything beyond index 6, plus full columns)
                mask = torch.full((42,), float("-inf"))
                mask[legal_moves] = 0.0

                probs = torch.softmax(logits + mask, dim=0)
                action = torch.multinomial(probs, 1).item()

            result = board.make_move(current, action)
            reward = get_reward(result, board, action, current)
            history.append((state, action, reward))

        else:
            if torch.rand(1).item() < 0.05:
                action = random.choice(legal_moves)
            else:
                # Minimax turn
                action = best_move(board, current)
            if action is None:
                action = random.choice(legal_moves)

            result = board.make_move(current, action)

        # Check for game end
        if result == MoveResult.WIN:
            return history, current, nn_player
        elif result == MoveResult.DRAW:
            return history, None, nn_player

        # Switch turns
        current = "Y" if current == "R" else "R"


def train_step(net, optimizer, history, winner, nn_player):
    """Update the network based on the NN's experience in one game."""
    if not history:
        return

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, requires_grad=True)

    for state, action, shaped_reward in history:
        # Final outcome reward from the neural network's perspective
        if winner is None:
            outcome = 0.3  # draw
        elif winner == nn_player:
            outcome = 1.0
        else:
            outcome = -1.0

        # Blend shaped reward with final outcome
        reward = 0.6 * outcome + 0.4 * shaped_reward

        logits, value = net(state)
        probs = torch.softmax(logits, dim=0)
        log_prob = torch.log(probs[action] + 1e-8)

        advantage = reward - value.squeeze().detach()

        policy_loss = -log_prob * advantage
        value_loss = F.mse_loss(
            value.squeeze(), torch.tensor(reward, dtype=torch.float32)
        )

        total_loss = total_loss + policy_loss + 0.5 * value_loss

    total_loss.backward()
    optimizer.step()


def train(episodes=170_000):
    net = ConnectFourNet()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    # Note: Training against Minimax is much slower than self-play.
    # Adjust 'episodes' down if you want a faster, less robust training cycle.
    for ep in range(episodes):
        epsilon = max(0.1, 1.0 - ep / (episodes * 0.9))

        history, winner, nn_player = play_game(net, epsilon=epsilon)
        train_step(net, optimizer, history, winner, nn_player)

        if ep % 50 == 0:
            print(
                f"{math.floor((ep / episodes) * 100)}%: {ep}/{episodes}, epsilon={epsilon:.3f} | Winner: {'NN' if winner == nn_player else ('Draw' if winner is None else 'Minimax')}"
            )

    torch.save(net.state_dict(), "c4_net.pth")
    print("Training complete. Model saved to c4_net.pth")
    return net


if __name__ == "__main__":
    train()
