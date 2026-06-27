import tkinter as tk

import torch

from board import Board, MoveResult
from minimax import get_legal_moves
from network import ConnectFourNet
from train import encode_board

ROWS = 6
COLS = 7
CELL_SIZE = 80
RADIUS = 30


def nn_best_move(net, board, player):
    """Pick the best move using the trained neural network.

    Args:
        net (ConnectFourNet): the trained network
        board (Board): the current board state
        player (char): the AI's player character

    Returns:
        int: column index of the chosen move

    """
    legal_moves = get_legal_moves(board)
    if not legal_moves:
        return None
    state = encode_board(board, player)
    with torch.no_grad():
        logits, _ = net(state)
        mask = torch.full((42,), float("-inf"))
        mask[legal_moves] = 0.0
        probs = torch.softmax(logits + mask, dim=0)
        return torch.argmax(probs).item()


class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect 4")
        self.board = Board()
        self.human = None
        self.ai = None
        self.player = "R"

        self.net = ConnectFourNet()
        self.net.load_state_dict(torch.load("c4_net.pth"))
        self.net.eval()

        self.show_start_screen()

    def show_start_screen(self):
        self.start_frame = tk.Frame(self.root)
        self.start_frame.grid(row=0, column=0)
        tk.Label(self.start_frame, text="Play as:", font=("Arial", 16)).pack(pady=10)
        tk.Button(
            self.start_frame,
            text="Red (Go First)",
            font=("Arial", 14),
            width=14,
            command=lambda: self.start_game(human="R"),
        ).pack(pady=5)
        tk.Button(
            self.start_frame,
            text="Yellow (Go Second)",
            font=("Arial", 14),
            width=14,
            command=lambda: self.start_game(human="Y"),
        ).pack(pady=5)

    def start_game(self, human):
        self.human = human
        self.ai = "Y" if human == "R" else "R"

        self.start_frame.destroy()
        self.build_board()
        if self.player != self.human:
            self.status.config(text="AI is thinking...")
            self.root.after(300, self.ai_move)
        else:
            self.status.config(
                text=f"Your turn ({'Red' if self.human == 'R' else 'Yellow'})"
            )

    def build_board(self):
        self.canvas = tk.Canvas(
            self.root, width=COLS * CELL_SIZE, height=ROWS * CELL_SIZE, bg="#1565C0"
        )
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_hover)

        self.status = tk.Label(self.root, font=("Arial", 14))
        self.status.grid(row=1, column=0, pady=6)

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        board_state = self.board.get_board()
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL_SIZE + CELL_SIZE // 2
                y = (ROWS - 1 - r) * CELL_SIZE + CELL_SIZE // 2
                cell = board_state[r][c]
                if cell == "R":
                    color = "#E53935"
                elif cell == "Y":
                    color = "#FDD835"
                else:
                    color = "#FAFAFA"
                self.canvas.create_oval(
                    x - RADIUS,
                    y - RADIUS,
                    x + RADIUS,
                    y + RADIUS,
                    fill=color,
                    outline="#1565C0",
                    width=3,
                )

    def on_hover(self, event):
        if self.player != self.human:
            return
        col = event.x // CELL_SIZE
        self.draw_board()
        if 0 <= col < COLS:
            x = col * CELL_SIZE + CELL_SIZE // 2
            color = "#E53935" if self.human == "R" else "#FDD835"
            self.canvas.create_oval(
                x - RADIUS,
                4,
                x + RADIUS,
                4 + RADIUS * 2,
                fill=color,
                outline="#1565C0",
                width=3,
            )

    def on_click(self, event):
        if self.player != self.human:
            return
        col = event.x // CELL_SIZE
        self.human_move(col)

    def human_move(self, col):
        result = self.board.make_move(self.human, col)
        if result in (MoveResult.INVALID_COL, MoveResult.COL_FULL):
            return
        self.draw_board()
        if self.check_end(result):
            return
        self.player = self.ai
        self.status.config(text="AI is thinking...")
        self.root.after(300, self.ai_move)

    def ai_move(self):
        move = nn_best_move(self.net, self.board, self.ai)
        if move is None:
            return
        result = self.board.make_move(self.ai, move)
        self.draw_board()
        if self.check_end(result):
            return
        self.player = self.human
        self.status.config(
            text=f"Your turn ({'Red' if self.human == 'R' else 'Yellow'})"
        )

    def check_end(self, result):
        if result == MoveResult.WIN:
            winner = self.board.get_winner()
            self.status.config(text=f"{'Red' if winner == 'R' else 'Yellow'} Wins!")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Motion>")
            self.show_reset_button()
            return True
        elif result == MoveResult.DRAW:
            self.status.config(text="It's a Draw!")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Motion>")
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
        self.board = Board()
        self.player = "R"
        self.human = None
        self.ai = None
        self.show_start_screen()


root = tk.Tk()
Connect4GUI(root)
root.mainloop()
