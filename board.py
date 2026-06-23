from enum import Enum


class MoveResult(Enum):
    INVALID_COL = "invalid_col"
    INVALID_PLAYER = "invalid_player"
    COL_FULL = "col_full"
    CONTINUE = "continue"
    WIN = "win"
    DRAW = "draw"


class Board:
    def __init__(self):
        self.board = [[None] * 7 for _ in range(6)]
        self.winner = None

    def get_board(self):
        return self.board

    def print_board(self):
        for i in range(6):
            print(self.board[5 - i])

    def make_move(self, player, col):
        if col < 0 or col > 6:
            return MoveResult.INVALID_COL

        if player.upper() not in ["R", "Y"]:
            return MoveResult.INVALID_PLAYER

        for i in range(6):
            if self.board[i][col] is None:
                self.board[i][col] = player
                if self.has_win(player, col, i):
                    return MoveResult.WIN
                if self.has_draw():
                    return MoveResult.DRAW

                return MoveResult.CONTINUE
        return MoveResult.COL_FULL

    def is_legal_move(self, col):
        if col < 0 or col > 6:
            return MoveResult.INVALID_COL

        if self.board[5][col] is not None:
            return MoveResult.INVALID_COL

        return MoveResult.CONTINUE

    def has_win(self, player, x, y):
        directions = [
            (1, 0),
            (0, 1),
            (1, 1),
            (-1, 1),
        ]

        for dx, dy in directions:
            count = 1

            i = 1
            while (
                0 <= x + (dx * i) < 7
                and 0 <= y + (dy * i) < 6
                and self.board[y + (dy * i)][x + (dx * i)] == player
            ):
                count += 1
                i += 1

            i = 1
            while (
                0 <= x - (dx * i) < 7
                and 0 <= y - (dy * i) < 6
                and self.board[y - (dy * i)][x - (dx * i)] == player
            ):
                count += 1
                i += 1
            if count >= 4:
                self.winner = player
                return True

        return False

    def reset(self):
        self.board = [[None] * 7 for _ in range(6)]
        self.winner = None

    def get_winner(self):
        return self.winner

    def has_draw(self):
        return all(self.board[5][c] is not None for c in range(7))
