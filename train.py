import copy
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from network import ConnectFourNet
from board import Board, MoveResult


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board, current_player):
    """
    Encode the board as a (3, 6, 7) float tensor from current_player's POV.
      channel 0: current player's pieces
      channel 1: opponent's pieces
      channel 2: all-ones bias plane
    """
    opponent = "r" if current_player == "y" else "y"
    grid = board.get_board()
    cur_plane  = [[1.0 if grid[r][c] == current_player else 0.0 for c in range(7)] for r in range(6)]
    opp_plane  = [[1.0 if grid[r][c] == opponent      else 0.0 for c in range(7)] for r in range(6)]
    bias_plane = [[1.0] * 7 for _ in range(6)]
    t = torch.tensor([cur_plane, opp_plane, bias_plane], dtype=torch.float32)
    return t  # (3, 6, 7)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

GAMMA = 0.97   # per-step discount applied from the end of the game

def get_shaped_reward(result, board, col, row, player):
    """
    Immediate shaped reward for a single move.
    Final outcome blending happens in train_step via discounted returns.
    """
    opponent = "r" if player == "y" else "y"

    if result == MoveResult.WIN:
        return 1.0

    if result == MoveResult.DRAW:
        return 0.05          # low — we want wins, not draws

    # Penalise gifting the opponent an immediate win
    if board.opponent_wins_next(opponent):
        return -0.5

    shaped = 0.0

    # Reward creating a 3-in-a-row threat
    threats_after  = board.count_threats(player, length=3)
    if threats_after > 0:
        shaped += 0.25 * min(threats_after, 2)   # cap at 2 threats

    # Reward forks (2+ winning threats simultaneously)
    winning_threats = board.count_threats(player, length=3)
    if winning_threats >= 2:
        shaped += 0.3

    # Small centre-column bonus (columns 2-4) — bootstraps early learning
    if col in (2, 3, 4):
        shaped += 0.05

    return shaped


def compute_discounted_returns(rewards, gamma=GAMMA):
    """Convert a list of per-step rewards into discounted returns."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, col, player, shaped_reward, outcome):
        self.buffer.append((state, col, player, shaped_reward, outcome))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------

class OpponentPool:
    """Keeps a pool of past network snapshots for diverse self-play."""
    def __init__(self, max_size=10):
        self.pool = []
        self.max_size = max_size

    def add(self, net):
        snapshot = copy.deepcopy(net)
        snapshot.eval()
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
        self.pool.append(snapshot)

    def sample(self):
        if not self.pool:
            return None
        return random.choice(self.pool)

    def __len__(self):
        return len(self.pool)


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def pick_action(net, board, current_player, epsilon, device="cpu"):
    legal_cols = board.get_valid_cols()
    state = encode_board(board, current_player).to(device)

    with torch.no_grad():
        logits, value = net(state)

    mask = torch.full((7,), float('-inf'), device=device)
    for c in legal_cols:
        mask[c] = 0.0
    probs = torch.softmax(logits + mask, dim=0)

    if torch.rand(1).item() < epsilon:
        col = random.choice(legal_cols)
    else:
        col = torch.multinomial(probs, 1).item()

    return state.cpu(), col, value.item()


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def play_game(net, opponent_net, epsilon, device="cpu"):
    """
    Play one game between net (yellow) and opponent_net (red).
    Returns a list of (state, col, player, shaped_reward) and the winner.
    """
    board = Board()
    history = []
    current = "y"
    nets = {"y": net, "r": opponent_net}

    while True:
        state, col, _ = pick_action(nets[current], board, current, epsilon, device)
        row = board.row_of_col(col)
        result = board.make_move(current, col)
        shaped = get_shaped_reward(result, board, col, row, current)
        history.append((state, col, current, shaped))

        if result == MoveResult.WIN:
            return history, current
        if result == MoveResult.DRAW:
            return history, None

        current = "r" if current == "y" else "y"


# ---------------------------------------------------------------------------
# Training step (from replay buffer)
# ---------------------------------------------------------------------------

def train_step(net, optimizer, batch, device="cpu"):
    net.train()
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for state, col, player, shaped_reward, outcome in batch:
        # Blend discounted shaped reward with final outcome
        reward = 0.6 * outcome + 0.4 * shaped_reward
        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)

        state_t = state.to(device)
        logits, value = net(state_t)

        probs   = torch.softmax(logits, dim=0)
        log_prob = torch.log(probs[col] + 1e-8)
        advantage = reward_t - value.squeeze().detach()

        policy_loss = -log_prob * advantage
        value_loss  = F.mse_loss(value.squeeze(), reward_t)

        total_loss = total_loss + policy_loss + 0.5 * value_loss

    total_loss = total_loss / max(len(batch), 1)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(episodes=500_000, batch_size=256, device="cpu"):
    net      = ConnectFourNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100_000, gamma=0.5)

    replay   = ReplayBuffer(capacity=100_000)
    pool     = OpponentPool(max_size=10)
    pool.add(net)   # seed the pool with the initial (random) network

    for ep in range(episodes):
        epsilon = max(0.05, 1.0 - ep / (episodes * 0.8))

        # Occasionally play against a past snapshot for diversity
        opponent = pool.sample() if len(pool) > 1 and random.random() < 0.4 else net

        history, winner = play_game(net, opponent, epsilon, device)

        # Compute discounted returns for the learning player ("y")
        shaped_rewards = [sr for _, _, p, sr in history if p == "y"]
        returns        = compute_discounted_returns(shaped_rewards)

        ret_idx = 0
        for state, col, player, shaped_reward in history:
            if player != "y":
                continue
            outcome = (1.0 if winner == "y"
                       else -1.0 if winner == "r"
                       else 0.05)
            # Use discounted return instead of raw shaped reward
            replay.push(state, col, player, returns[ret_idx], outcome)
            ret_idx += 1

        # Start training once the buffer has enough samples
        if len(replay) >= batch_size:
            batch = replay.sample(batch_size)
            train_step(net, optimizer, batch, device)
            scheduler.step()

        # Snapshot the network into the opponent pool periodically
        if ep > 0 and ep % 10_000 == 0:
            pool.add(net)

        if ep % 1 == 0:
            print(f"Episode {ep:>7}/{episodes}  "
                  f"epsilon={epsilon:.3f}  "
                  f"buffer={len(replay):>6}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(net.state_dict(), "connect_four_net.pth")
    print("Training complete. Model saved to connect_four_net.pth")
    return net


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    train(device=device)
