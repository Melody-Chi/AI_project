from search.move import random_move
from .utils import render_board

# 给定一个coordirate，board：dict(tuple,tuple), 第一个tuple是coordinate，第二个是token
# 改的应该是coordinate，不变的应该是token
# coord应该提取第一个tuple

def update_board(board: dict[tuple, tuple], coord: tuple, move = None) -> dict[tuple, tuple]:
    """
    Update the chess board with a new move.

    Args:
    - board: A dictionary containing tuples of coordinates as keys and tuples of color and power as values.
    - coord: A tuple containing the coordinates of the piece to move.
    - move: A tuple containing the new coordinates to move the piece to.

    Returns:
    - A new board dictionary with the updated move.
    """
    # Create a copy of the original board
    new_board = board.copy()
    print("board = ", board)
    # board[coord] =  ('r', 2)

    # If move is None, remove the piece from the board
    if move is None:
        del new_board[coord]
    else:
        # Get the color and power of the piece at the current position
        color, power = board[coord]
        # Remove the piece from the current position
        del new_board[coord]
        # Add the piece to the new position
        new_board[random_move(board, move)] = (color, power)
        print("new board =", new_board)

    

    return new_board
