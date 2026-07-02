import copy
import math
import random
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from board import Board, MoveResult
from minimax import best_move, get_legal_moves
from network import ConnectFourNet


def encode_board(board, current_player):
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

    for r in range(5, -1, -1):
        if test_board.get_board()[r][col] is not None:
            test_board.get_board()[r][col] = None
            break

    result = test_board.make_move(opponent, col)
    return result == MoveResult.WIN


def creates_threat(board, col, player):
    """Check if the move just played created a 3-in-a-row with an open end."""
    b = board.get_board()
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]

    # Find the row the piece landed in
    row = None
    for r in range(5, -1, -1):
        if b[r][col] == player:
            row = r
            break
    if row is None:
        return False

    for dx, dy in directions:
        count = 1
        open_ends = 0

        for sign in (1, -1):
            i = 1
            while True:
                nx, ny = col + dx * sign * i, row + dy * sign * i
                if 0 <= nx < 7 and 0 <= ny < 6:
                    if b[ny][nx] == player:
                        count += 1
                        i += 1
                    elif b[ny][nx] is None:
                        open_ends += 1
                        break
                    else:
                        break
                else:
                    break

        if count == 3 and open_ends >= 1:
            return True

    return False


def get_reward(result, board, col, player):
    """Shaped reward: win, draw, block, threat, or neutral."""
    if result == MoveResult.WIN:
        return 1.0
    if result == MoveResult.DRAW:
        return 0.3

    opponent = "Y" if player == "R" else "R"
    if was_blocking_move(board, col, opponent):
        return 0.5

    if creates_threat(board, col, player):
        return 0.2

    return 0.0


def play_game(net, epsilon=0.1, minimax_depth=7):
    """Play one game where the Neural Network plays against the Minimax algorithm.

    Args:
        net (ConnectFourNet): the network being trained
        epsilon (float): exploration rate for epsilon-greedy action selection
        minimax_depth (int): how many moves ahead minimax searches

    Returns:
        history (list): (state_tensor, action, reward) tuples for the NN's moves
        winner (str | None): "R", "Y", or None for draw
        nn_player (str): which colour the NN was playing
    """
    board = Board()
    history = []
    current = "R"
    nn_player = random.choice(["R", "Y"])

    while True:
        legal_moves = get_legal_moves(board)

        if current == nn_player:
            state = encode_board(board, current)

            if torch.rand(1).item() < epsilon:
                action = random.choice(legal_moves)
            else:
                logits, _ = net(state)
                logits = logits.squeeze(0)

                # Clamp to prevent extreme values destabilising softmax
                logits = torch.clamp(logits, -10, 10)

                mask = torch.full((7,), float("-inf"))
                mask[legal_moves] = 0.0
                masked_logits = logits + mask

                probs = torch.softmax(masked_logits, dim=0)

                # Fallback if probabilities are still invalid
                if (
                    torch.isnan(probs).any()
                    or torch.isinf(probs).any()
                    or (probs < 0).any()
                ):
                    action = random.choice(legal_moves)
                else:
                    action = torch.multinomial(probs, 1).item()

            result = board.make_move(current, action)
            reward = get_reward(result, board, action, current)
            history.append((state, action, reward))

        else:
            if torch.rand(1).item() < 0.05:
                action = random.choice(legal_moves)
            else:
                action = best_move(board, current, max_depth=minimax_depth)
            if action is None:
                action = random.choice(legal_moves)

            result = board.make_move(current, action)

        if result == MoveResult.WIN:
            return history, current, nn_player
        elif result == MoveResult.DRAW:
            return history, None, nn_player

        current = "Y" if current == "R" else "R"


def train_step(net, optimizer, history, winner, nn_player):
    """Update the network based on the NN's experience in one game."""
    if not history:
        return

    optimizer.zero_grad()
    losses = []

    for state, action, shaped_reward in history:
        if winner is None:
            outcome = 0.3
        elif winner == nn_player:
            outcome = 1.0
        else:
            outcome = -1.0

        reward = 0.6 * outcome + 0.4 * shaped_reward

        logits, value = net(state)
        # logits shape: (1, 7) — squeeze to (7,) for indexing
        logits = logits.squeeze(0)
        value = value.squeeze()

        probs = torch.softmax(logits, dim=0)
        log_prob = torch.log(probs[action] + 1e-8)

        advantage = reward - value.detach()

        policy_loss = -log_prob * advantage
        value_loss = F.mse_loss(value, torch.tensor(reward, dtype=torch.float32))

        losses.append(policy_loss + 0.5 * value_loss)

    total_loss = torch.stack(losses).sum()
    total_loss.backward()
    optimizer.step()


def train(episodes=170_000):
    net = ConnectFourNet()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.95)

    last_time = time.perf_counter()

    for ep in range(episodes):
        epsilon = max(0.05, 1.0 - ep / (episodes * 0.9))

        # Curriculum: start with shallow minimax, ramp up as training progresses
        progress = ep / episodes
        if progress < 0.2:
            minimax_depth = 1
        elif progress < 0.5:
            minimax_depth = 5
        else:
            minimax_depth = 10

        history, winner, nn_player = play_game(
            net, epsilon=epsilon, minimax_depth=minimax_depth
        )
        train_step(net, optimizer, history, winner, nn_player)
        scheduler.step()

        if ep % 50 == 0 and ep != 0:
            dt = (time.perf_counter() - last_time) / 50
            last_time = time.perf_counter()
            eta = dt * (episodes - ep)
            hours, remainder = divmod(int(eta), 3600)
            minutes, seconds = divmod(remainder, 60)
            winner_str = (
                "NN"
                if winner == nn_player
                else ("Draw" if winner is None else "Minimax")
            )
            print(
                f"{math.floor((ep / episodes) * 100)}%: {ep}/{episodes}, "
                f"epsilon={epsilon:.3f}, depth={minimax_depth} | "
                f"Winner: {winner_str} | ETA: {hours:02}h {minutes:02}m {seconds:02}s"
            )

    torch.save(net.state_dict(), "c4_net.pth")
    print("Training complete. Model saved to c4_net.pth")
    return net


if __name__ == "__main__":
    train()
