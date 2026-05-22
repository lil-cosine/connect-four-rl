import torch
import torch.nn.functional as F
from board import Board, MoveResult
from network import ActorCriticNet
from agent import greedy_agent
import os
import sys

def pick_checkpoint():
    checkpoints = sorted([f for f in os.listdir("./checkpoints") if f.startswith("ckpt_") and f.endswith(".pt")])
    if not checkpoints:
        print("No checkpoints found in current directory.")
        sys.exit(1)

    print("\nAvailable checkpoints:")
    for i, ckpt in enumerate(checkpoints):
        print(f"  [{i}] {ckpt}")

    while True:
        try:
            choice = int(input("\nPick a checkpoint number: "))
            if 0 <= choice < len(checkpoints):
                return 'checkpoints/' + checkpoints[choice]
            print(f"Enter a number between 0 and {len(checkpoints)-1}")
        except ValueError:
            print("Enter a valid number.")

def load_agent(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ActorCriticNet()
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device)
    net.eval()
    return greedy_agent(net, device)

def print_board(board):
    grid = board.get_board()
    print("\n  1 2 3 4 5 6 7")
    print(" +" + "-+-".join(["-"]*7) + "+")
    for row in reversed(grid):
        display = []
        for cell in row:
            if cell == "y":   display.append("🟡")
            elif cell == "r": display.append("🔴")
            else:             display.append("⚫")
        print(" |" + "|".join(display) + "|")
    print(" +" + "-+-".join(["-"]*7) + "+")
    print("  1 2 3 4 5 6 7\n")

def get_human_move(board):
    valid = board.get_valid_cols()
    while True:
        try:
            col = int(input("Your move (1-7): ")) - 1
            if col in valid:
                return col
            elif 0 <= col <= 6:
                print("That column is full, pick another.")
            else:
                print("Enter a number between 1 and 7.")
        except ValueError:
            print("Enter a valid number.")

def play(human_player, ai_agent):
    board = Board()
    ai_player = 1 - human_player
    current = 0

    symbols = {human_player: "🟡 (you)", ai_player: "🔴 (AI)"}
    print(f"\nYou are 🟡  |  AI is 🔴")
    print(f"{'You go first!' if human_player == 0 else 'AI goes first!'}")
    print_board(board)

    while True:
        if current == human_player:
            col = get_human_move(board)
        else:
            print("AI is thinking...")
            col = ai_agent.choose_col(board, ai_player)
            print(f"AI plays column {col + 1}")

        result = board.make_move(current, col)
        print_board(board)

        if result == MoveResult.WIN:
            if current == human_player:
                print("🎉 You win!")
            else:
                print("🤖 AI wins!")
            return
        elif result == MoveResult.DRAW:
            print("It's a draw!")
            return
        elif result in (MoveResult.COL_FULL, MoveResult.INVALID_COL):
            print("Invalid move — something went wrong.")
            return

        current = 1 - current

def main():
    print("=== Connect Four vs AI ===")
    ckpt = pick_checkpoint()
    print(f"\nLoading {ckpt}...")
    agent = load_agent(ckpt)

    while True:
        while True:
            order = input("\nDo you want to go first? (y/n): ").strip().lower()
            if order in ("y", "n"):
                break
            print("Enter y or n.")

        human_player = 0 if order == "y" else 1
        play(human_player, agent)

        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main()
