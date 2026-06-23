import tkinter as tk

from board import Board, MoveResult
from minimax import best_move


class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect 4")
        self.board = Board()
        self.buttons = [[None] * 7 for _ in range(6)]
        self.human = None
        self.ai = None
        self.player = "R"
        self.show_start_screen()

    def show_start_screen(self):
        self.start_frame = tk.Frame(self.root)
        self.start_frame.grid(row=0, column=0)
        tk.Label(self.start_frame, text="Play as:", font=("Arial", 16)).pack(pady=10)
        tk.Button(
            self.start_frame,
            text="Red (Go First)",
            font=("Arial", 14),
            width=15,
            command=lambda: self.start_game(human="R"),
        ).pack(pady=5)
        tk.Button(
            self.start_frame,
            text="Yellow (Go Second)",
            font=("Arial", 14),
            width=15,
            command=lambda: self.start_game(human="Y"),
        ).pack(pady=5)

    def start_game(self, human):
        self.human = human
        if human == "R":
            self.ai = "Y"
        else:
            self.ai = "R"

        self.start_frame.destroy()
        self.build_board()
        if self.player != self.human:
            self.status.config(text="AI is thinking...")
            self.root.after(300, self.ai_move)
        else:
            self.status.config(text=f"Your turn ({self.human})")

    def build_board(self):
        self.board_frame = tk.Frame(self.root)
        self.board_frame.grid(row=0, column=0)
        for j in range(6):
            for i in range(7):
                btn = tk.Button(
                    self.board_frame,
                    bg="white",
                    font=("Arial", 20),
                    width=4,
                    height=2,
                    command=lambda col=i: self.human_move(col),
                )
                btn.grid(row=j, column=i)
                self.buttons[j][i] = btn
        self.status = tk.Label(self.root, font=("Arial", 14))
        self.status.grid(row=1, column=0, pady=10)

    def update_ui(self):
        state = self.board.get_board()
        for r in range(6):
            for c in range(7):
                val = state[r][c]
                ui_row = 5 - r
                if val == "R":
                    self.buttons[ui_row][c].config(bg="red")
                elif val == "Y":
                    self.buttons[ui_row][c].config(bg="yellow")
                else:
                    self.buttons[ui_row][c].config(bg="white")

    def human_move(self, col):
        if self.player != self.human:
            return
        result = self.board.make_move(self.human, col)
        if result in [MoveResult.INVALID_COL, MoveResult.COL_FULL]:
            return
        self.update_ui()
        if self.check_end(result):
            return
        self.player = self.ai
        self.status.config(text="AI is thinking...")
        self.root.update()
        self.root.after(10, self.ai_move)

    def ai_move(self):
        move = best_move(self.board, self.ai)
        result = self.board.make_move(self.ai, move)
        self.update_ui()
        if self.check_end(result):
            return
        self.player = self.human
        self.status.config(text=f"Your turn ({self.human})")

    def check_end(self, result):
        if result == MoveResult.WIN:
            winner = "Red" if self.board.get_winner() == "R" else "Yellow"
            self.status.config(text=f"Player {winner} Wins!")
            self.disable_board()
            self.show_reset_button()
            return True
        elif result == MoveResult.DRAW:
            self.status.config(text="It's a Draw!")
            self.disable_board()
            self.show_reset_button()
            return True
        return False

    def show_reset_button(self):
        tk.Button(
            self.root, text="Play Again", font=("Arial", 14), command=self.reset
        ).grid(row=2, column=0, pady=10)

    def reset(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.board.reset()
        self.buttons = [[None] * 7 for _ in range(6)]
        self.player = "R"
        self.human = None
        self.ai = None
        self.show_start_screen()

    def disable_board(self):
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = Connect4GUI(root)
    root.mainloop()
