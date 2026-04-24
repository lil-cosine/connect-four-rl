from board import Board, MoveResult

def play_game(model_a, model_b, max_moves=42):
    board = Board()
    models = {0: model_a, 1: model_b}
    player = 0

    for _ in range(max_moves):
        col = models[player].choose_col(board, player)
        result = board.make_move(player, col)

        if result == MoveResult.WIN:
            return 1 if player == 0 else -1
        elif result == MoveResult.DRAW:
            return 0
        elif result in (MoveResult.INVALID_COL, MoveResult.COL_FULL):
            return -1 if player == 0 else 1

        player = (player + 1) % 2

    return 0


def evaluate(model, opponents, games_per_opponent=10):
    results = []

    for opponent in opponents:
        for i in range(games_per_opponent):
            if i % 2 == 0:
                result = play_game(model, opponent)
            else:
                result = -play_game(opponent, model)
            results.append(result)

    return sum(results) / len(results)
