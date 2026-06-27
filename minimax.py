import copy

from board import MoveResult

MAX_DEPTH = 10

MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]


def best_move(board, current_player):
    """Finds the best move of the current board

    Args:
        board (Board): the current board state
        current_player (char): the player making the move

    Returns:
        best (int): the column index of the best move

    """
    legal_moves = get_legal_moves(board)
    best = None
    alpha = float("-inf")
    beta = float("inf")
    opponent = "Y" if current_player == "R" else "R"

    if current_player == "R":
        best_score = float("-inf")
    else:
        best_score = float("inf")

    for move in legal_moves:
        new_board = copy.deepcopy(board)
        new_board.make_move(current_player, move)
        score = minimax_score(new_board, opponent, alpha=alpha, beta=beta)

        if current_player == "R" and score > best_score:
            best_score = score
            best = move
            alpha = max(alpha, best_score)
        elif current_player == "Y" and score < best_score:
            best_score = score
            best = move
            beta = min(beta, best_score)
    return best


def minimax_score(
    board, current_player, depth=0, alpha=float("-inf"), beta=float("inf")
):
    """Builds the minimax tree of the current move with alpha-beta pruning

    Args:
        board (Board): the current state of the board
        current_player (char): the player making the move
        depth (int): current depth in the search tree
        alpha (float): best score the maximiser can guarantee
        beta (float): best score the minimiser can guarantee

    Returns:
        score (int): the heuristic value of the board position

    """
    if depth >= MAX_DEPTH:
        return 0

    depth += 1

    if board.get_winner() == "R":
        return 10 - depth
    elif board.get_winner() == "Y":
        return depth - 10
    elif board.has_draw():
        return 0

    legal_moves = get_legal_moves(board)
    opponent = "Y" if current_player == "R" else "R"

    if current_player == "R":  # Maximiser
        val = float("-inf")
        for move in legal_moves:
            new_board = copy.deepcopy(board)
            new_board.make_move(current_player, move)
            val = max(val, minimax_score(new_board, opponent, depth + 1, alpha, beta))
            alpha = max(alpha, val)
            if alpha >= beta:
                break  # Beta cutoff
        return val

    else:  # Minimiser
        val = float("inf")
        for move in legal_moves:
            new_board = copy.deepcopy(board)
            new_board.make_move(current_player, move)
            val = min(val, minimax_score(new_board, opponent, depth + 1, alpha, beta))
            beta = min(beta, val)
            if alpha >= beta:
                break  # Alpha cutoff
        return val


def get_legal_moves(board):
    """Gets the list of legal moves for the current board, ordered centre-first
    to maximise alpha-beta pruning efficiency.

    Args:
        board (Board): the current state of the board

    Returns:
        legal_moves (list[int]): column indices of all legal moves

    """
    return [
        col for col in MOVE_ORDER if board.is_legal_move(col) == MoveResult.CONTINUE
    ]
