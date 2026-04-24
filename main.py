from board import Board, MoveResult

b = Board()


result = MoveResult.CONTINUE
player = 0
while result != MoveResult.WIN:
    print(f"Player {player+1} Column:")
    col = input()
    result = b.make_move(player, int(col)-1)
    b.print_board()
    if result == MoveResult.CONTINUE:
        player = (player + 1) % 2
print(f'Player {player+1} Wins!')
