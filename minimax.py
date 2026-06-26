import copy

from board import MoveResult

MAX_DEPTH = 7


def best_move(board, current_player):
    """Finds the best move of the current board

    Args:
        board (Board): the current board state
        current_player (char): the player making the move

    Returns:
        best ([int,int]): the [x,y] location of the best move

    """
    legal_moves = get_legal_moves(board)
    best = None
    if current_player == "R":
        best_score = float("-inf")
    else:
        best_score = float("inf")

    for move in legal_moves:
        new_board = copy.deepcopy(board)
        new_board.make_move(current_player, move)
        if current_player == "R":
            opponent = "Y"
        else:
            opponent = "R"
        score = minimax_score(new_board, opponent)

        if current_player == "R" and score > best_score:
            best_score = score
            best = move
        elif current_player == "Y" and score < best_score:
            best_score = score
            best = move

    return best


def minimax_score(board, current_player, depth=0):
    """Builds the minimax tree of the current move

    Args:
        board (Board): the current state of the board
        player (char): the player making the move

    Returns:
        score (int): the total score of the move

    """
    if depth >= MAX_DEPTH:
        return 0

    depth += 1

    if board.get_winner() == "R":
        return 10
    elif board.get_winner() == "Y":
        return -10
    elif board.has_draw():
        return 0

    legal_moves = get_legal_moves(board)

    scores = []
    for move in legal_moves:
        new_board = copy.deepcopy(board)
        new_board.make_move(current_player, move)

        if current_player == "R":
            opponent = "Y"
        else:
            opponent = "R"

        scores.append(minimax_score(new_board, opponent, depth + 1))

    if current_player == "R":
        return max(scores)
    else:
        return min(scores)


def get_legal_moves(board):
    """Gets the list of legal moves for the current board

    Args:
        board (Board): the current state of the board

    Returns:
        legal_moves (array([int, int]): the array containing the [x,y]
                                        location of all legal moves

    """
    legal_moves = []
    for i in range(7):
        if board.is_legal_move(i) == MoveResult.CONTINUE:
            legal_moves.append(i)
    return legal_moves
