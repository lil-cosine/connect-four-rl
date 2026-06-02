import tkinter as tk
from board import Board, MoveResult
import torch
from network import ConnectFourNet
from train import encode_board

YELLOW_BG = "#f5c518"
RED_BG    = "#e03030"
EMPTY_BG  = "#1a1a2e"
BOARD_BG  = "#0f3460"


def best_move(net, board, player, device="cpu"):
    """Pick the best column for the AI using the trained network."""
    legal_cols = board.get_valid_cols()
    state = encode_board(board, player).to(device)
    with torch.no_grad():
        logits, _ = net(state)
    mask = torch.full((7,), float('-inf'), device=device)
    for c in legal_cols:
        mask[c] = 0.0
    probs = torch.softmax(logits + mask, dim=0)
    return torch.argmax(probs).item()


class ConnectFourGUI:
    CELL   = 80    # px per cell
    RADIUS = 32    # piece radius

    def __init__(self, root):
        self.root   = root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.root.title("Connect Four")
        self.root.resizable(False, False)

        self.net = ConnectFourNet().to(self.device)
        try:
            self.net.load_state_dict(
                torch.load("connect_four_net.pth",
                           map_location=self.device,
                           weights_only=True)
            )
            self.net.eval()
            print("Loaded trained model.")
        except FileNotFoundError:
            print("No trained model found — AI plays randomly. Run train.py first.")

        self.board          = Board()
        self.human          = None
        self.ai             = None
        self.current_player = "y"
        self.game_over      = False

        self.show_start_screen()

    # ------------------------------------------------------------------
    # Start screen
    # ------------------------------------------------------------------

    def show_start_screen(self):
        self.start_frame = tk.Frame(self.root, bg="#1a1a2e", padx=40, pady=30)
        self.start_frame.pack()

        tk.Label(self.start_frame, text="Connect Four",
                 font=("Arial", 24, "bold"), fg="white", bg="#1a1a2e").pack(pady=(0, 6))
        tk.Label(self.start_frame, text="Choose your colour:",
                 font=("Arial", 14), fg="#aaaaaa", bg="#1a1a2e").pack(pady=(0, 16))

        btn_style = dict(font=("Arial", 14, "bold"), width=18,
                         relief="flat", cursor="hand2")
        tk.Button(self.start_frame, text="🟡  Yellow — Go First",
                  bg=YELLOW_BG, fg="#222",
                  command=lambda: self.start_game("y"),
                  **btn_style).pack(pady=6)
        tk.Button(self.start_frame, text="🔴  Red — Go Second",
                  bg=RED_BG, fg="white",
                  command=lambda: self.start_game("r"),
                  **btn_style).pack(pady=6)

    # ------------------------------------------------------------------
    # Game setup
    # ------------------------------------------------------------------

    def start_game(self, human):
        self.human          = human
        self.ai             = "r" if human == "y" else "y"
        self.current_player = "y"
        self.game_over      = False

        self.start_frame.destroy()
        self.build_board_ui()

        if self.current_player != self.human:
            self.status.config(text="AI is thinking…")
            self.root.after(400, self.ai_move)
        else:
            self.status.config(text="Your turn")

    # ------------------------------------------------------------------
    # Board canvas
    # ------------------------------------------------------------------

    def build_board_ui(self):
        C, R = self.CELL, self.RADIUS
        W = 7 * C
        H = 6 * C

        # Column-click buttons (invisible, sit above the canvas)
        btn_frame = tk.Frame(self.root, bg=BOARD_BG)
        btn_frame.pack()
        self.col_buttons = []
        for col in range(7):
            btn = tk.Button(btn_frame, text="▼", font=("Arial", 12),
                            width=4, bg=BOARD_BG, fg="white",
                            relief="flat", cursor="hand2",
                            command=lambda c=col: self.human_move(c))
            btn.grid(row=0, column=col, padx=2, pady=2)
            self.col_buttons.append(btn)

        # Canvas for the board
        self.canvas = tk.Canvas(self.root, width=W, height=H,
                                bg=BOARD_BG, highlightthickness=0)
        self.canvas.pack()
        self.circles = [[None] * 7 for _ in range(6)]

        for row in range(6):
            for col in range(7):
                x0 = col * C + (C - 2 * R) // 2
                y0 = (5 - row) * C + (C - 2 * R) // 2
                self.circles[row][col] = self.canvas.create_oval(
                    x0, y0, x0 + 2 * R, y0 + 2 * R,
                    fill=EMPTY_BG, outline=BOARD_BG, width=3
                )

        # Status bar
        self.status = tk.Label(self.root, text="", font=("Arial", 14),
                               bg="#1a1a2e", fg="white", pady=8)
        self.status.pack(fill="x")

    def _set_cell(self, row, col, player):
        color = YELLOW_BG if player == "y" else RED_BG
        self.canvas.itemconfig(self.circles[row][col], fill=color)

    # ------------------------------------------------------------------
    # Move handling
    # ------------------------------------------------------------------

    def human_move(self, col):
        if self.game_over or self.current_player != self.human:
            return

        row = self.board.row_of_col(col)
        if row is None:
            return

        result = self.board.make_move(self.human, col)
        if result in (MoveResult.INVALID_COL, MoveResult.COL_FULL,
                      MoveResult.INVALID_PLAYER):
            return

        self._set_cell(row, col, self.human)
        if self.check_end(result):
            return

        self.current_player = self.ai
        self.status.config(text="AI is thinking…")
        self._disable_col_buttons()
        self.root.after(350, self.ai_move)

    def ai_move(self):
        col    = best_move(self.net, self.board, self.ai, self.device)
        row    = self.board.row_of_col(col)
        result = self.board.make_move(self.ai, col)
        self._set_cell(row, col, self.ai)

        if self.check_end(result):
            return

        self.current_player = self.human
        self._enable_col_buttons()
        self.status.config(text="Your turn")

    # ------------------------------------------------------------------
    # End-of-game
    # ------------------------------------------------------------------

    def check_end(self, result):
        if result == MoveResult.WIN:
            winner = self.board.get_winner()
            label  = "You win! 🎉" if winner == self.human else "AI wins!"
            color  = YELLOW_BG if winner == "y" else RED_BG
            self.status.config(text=label, fg=color)
            self._end_game()
            return True
        if result == MoveResult.DRAW:
            self.status.config(text="It's a draw!", fg="white")
            self._end_game()
            return True
        return False

    def _end_game(self):
        self.game_over = True
        self._disable_col_buttons()
        tk.Button(self.root, text="Play Again", font=("Arial", 13, "bold"),
                  bg="#0f3460", fg="white", relief="flat", cursor="hand2",
                  command=self.reset).pack(pady=10)

    def reset(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.board          = Board()
        self.current_player = "y"
        self.human          = None
        self.ai             = None
        self.game_over      = False
        self.show_start_screen()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _disable_col_buttons(self):
        for btn in self.col_buttons:
            btn.config(state=tk.DISABLED)

    def _enable_col_buttons(self):
        for btn in self.col_buttons:
            btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    ConnectFourGUI(root)
    root.mainloop()
