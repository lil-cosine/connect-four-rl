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
        self.board = [[""] * 7 for _ in range(6)]
        self.players = {0: "y", 1: "r"}

    def get_board(self):
        return self.board

    def print_board(self):
        for i in range(6):
            print(self.board[5-i])

    def make_move(self, player, col):
        if col < 0 or col > 6:
            return MoveResult.INVALID_COL

        if player not in (0,1):
            return MoveResult.INVALID_PLAYER
        
        for i in range(6):
            if self.board[i][col] == "":
                self.board[i][col] = self.players[player]
                if self.has_win(player, col, i):
                    return MoveResult.WIN
                if self.if_draw():
                    return MoveResult.DRAW
                
                return MoveResult.CONTINUE
        return MoveResult.COL_FULL

    def has_win(self, player, x, y):
        directions = [
            (1,0),
            (0, 1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1)
        ]
    
        for dx, dy in directions:
            count = 1

            i = 1
            while 0 <= x + (dx * i) < 7 and 0 <= y + (dy * i) < 6 and self.board[y + (dy * i)][x + (dx * i)] == self.players[player]:
                count += 1
                i += 1

            i = 1
            while 0 <= x - (dx * i) < 7 and 0 <= y - (dy * i) < 6 and self.board[y - (dy * i)][x - (dx * i)] == self.players[player]:
                count += 1
                i += 1
            if count >= 4:
                return 1
    
        return 0

    def reset(self):
        self.board = [[""] * 7 for _ in range(6)]
    
    def get_valid_cols(self):
        return [c for c in range(7) if self.board[5][c] == ""]

    def if_draw(self):
        return all(self.board[5][c] != "" for c in range(7))

    def get_state(self, player):
        state = []
        me = self.players[player]
        opp = self.players[1 - player]
        for row in self.board:
            for cell in row:
                if cell == me: state.append(1.0)
                elif cell == opp: state.append(-1.0)
                else: state.append(0.0)
        return state
