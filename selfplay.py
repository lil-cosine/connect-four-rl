# selfplay.py

from board import Board, MoveResult
from copy import deepcopy
import torch

WIN_REWARD      = 1.0
LOSS_REWARD     = -1.0
DRAW_REWARD     = 0.0
INVALID_REWARD  = -1.0

# tiny shaping rewards only
BLOCK_REWARD    = 0.05
MISS_WIN_PENALTY = -0.05


def would_win(board, player, col):
    b = deepcopy(board)
    result = b.make_move(player, col)
    return result == MoveResult.WIN


def collect_episode(agent):
    board = Board()
    player = 0

    # separate trajectories per player
    player_transitions = {
        0: [],
        1: []
    }

    while True:

        log_probs, probs, value = agent.policy_and_value(board, player)

        # sanitize probabilities
        probs = torch.clamp(probs, min=1e-8)

        if torch.any(torch.isnan(probs)):
            probs = torch.ones_like(probs)

            valid = board.get_valid_cols()
            mask = torch.zeros(7, device=probs.device)
            mask[valid] = 1.0

            probs = probs * mask

        probs = probs / probs.sum()

        action = torch.multinomial(probs, 1).item()

        log_prob = log_probs[action]

        entropy = -(probs * torch.log(probs + 1e-8)).sum()

        opponent = 1 - player

        valid_cols = board.get_valid_cols()

        blocking_cols = [
            c for c in valid_cols
            if would_win(board, opponent, c)
        ]

        winning_cols = [
            c for c in valid_cols
            if would_win(board, player, c)
        ]

        # default reward
        reward = 0.0

        # tiny shaping rewards
        if blocking_cols and action in blocking_cols:
            reward += BLOCK_REWARD

        if winning_cols and action not in winning_cols:
            reward += MISS_WIN_PENALTY

        result = board.make_move(player, action)

        player_transitions[player].append(
            (log_prob, value, reward, entropy)
        )

        # terminal states
        if result == MoveResult.WIN:

            # reward winner
            lp, v, r, e = player_transitions[player][-1]
            player_transitions[player][-1] = (
                lp,
                v,
                r + WIN_REWARD,
                e
            )

            # punish loser
            loser = 1 - player

            if len(player_transitions[loser]) > 0:
                lp, v, r, e = player_transitions[loser][-1]

                player_transitions[loser][-1] = (
                    lp,
                    v,
                    r + LOSS_REWARD,
                    e
                )

            break

        elif result == MoveResult.DRAW:

            lp, v, r, e = player_transitions[player][-1]

            player_transitions[player][-1] = (
                lp,
                v,
                r + DRAW_REWARD,
                e
            )

            break

        elif result in (
            MoveResult.COL_FULL,
            MoveResult.INVALID_COL
        ):

            lp, v, r, e = player_transitions[player][-1]

            player_transitions[player][-1] = (
                lp,
                v,
                r + INVALID_REWARD,
                e
            )

            break

        player = 1 - player

    # flatten trajectories into one list
    transitions = []

    for p in [0, 1]:
        transitions.extend(player_transitions[p])

    return transitions
