from enum import Enum

class MoveResult(Enum):
    INVALID_COL = "invalid_col"
    INVALID_PLAYER = "invalid_player"
    COL_FULL = "col_full"
    CONTINUE = "continue"
    WIN = "win"
    DRAW = "draw"

class Board():
    def __init__(self):
        self.board = [[None] * 7 for _ in range(6)]
        self.winner = None

    def get_board(self):
        return self.board

    def print_board(self):
        for i in range(6):
            print(self.board[5 - i])

    def make_move(self, player, col):
        if player not in ("y", "r"):
            return MoveResult.INVALID_PLAYER

        legal = self.is_legal_move(col)
        if legal != MoveResult.CONTINUE:
            return legal

        for i in range(6):
            if self.board[i][col] is None:
                self.board[i][col] = player
                if self.has_win(player, col, i):
                    return MoveResult.WIN
                if self.has_draw():
                    return MoveResult.DRAW
                return MoveResult.CONTINUE

    def is_legal_move(self, col):
        if col < 0 or col > 6:
            return MoveResult.INVALID_COL
        if self.board[5][col] is not None:
            return MoveResult.COL_FULL
        return MoveResult.CONTINUE

    def has_win(self, player, x, y):
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        for dx, dy in directions:
            count = 1
            i = 1
            while (0 <= x + dx * i < 7 and 0 <= y + dy * i < 6
                   and self.board[y + dy * i][x + dx * i] == player):
                count += 1
                i += 1
            i = 1
            while (0 <= x - dx * i < 7 and 0 <= y - dy * i < 6
                   and self.board[y - dy * i][x - dx * i] == player):
                count += 1
                i += 1
            if count >= 4:
                self.winner = player
                return True
        return False

    def reset(self):
        self.board = [[None] * 7 for _ in range(6)]
        self.winner = None

    def get_valid_cols(self):
        return [c for c in range(7) if self.board[5][c] is None]

    def get_winner(self):
        return self.winner

    def has_draw(self):
        return all(self.board[5][c] is not None for c in range(7))

    def row_of_col(self, col):
        for j in range(6):
            if self.board[j][col] is None:
                return j
        return None

    def count_threats(self, player, length=3):
        """Count open-ended sequences of `length` for the given player."""
        count = 0
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        for y in range(6):
            for x in range(7):
                if self.board[y][x] != player:
                    continue
                for dx, dy in directions:
                    seq = 1
                    i = 1
                    while (0 <= x + dx * i < 7 and 0 <= y + dy * i < 6
                           and self.board[y + dy * i][x + dx * i] == player):
                        seq += 1
                        i += 1
                    open_end = (0 <= x + dx * i < 7 and 0 <= y + dy * i < 6
                                and self.board[y + dy * i][x + dx * i] is None)
                    if seq >= length and open_end:
                        count += 1
        return count

    def opponent_wins_next(self, opponent):
        """Return True if opponent has any immediate winning move."""
        for col in self.get_valid_cols():
            row = self.row_of_col(col)
            self.board[row][col] = opponent
            win = self.has_win(opponent, col, row)
            self.board[row][col] = None
            if win:
                self.winner = None
                return True
        return False
