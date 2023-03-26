from search.move import random_move
from .utils import render_board
from search.update import update_board

def spread(board, coord, direction):
    """
    Spread an existing token stack of power k which is currently controlled by the red player.
    A SPREAD action is defined by a board coordinate, (r,q), the cell where this stack currently resides,
    and a hex direction (rd,qd), which is one of the following “hex neighbour” offsets: (0,1), (−1,1), (−1,0),
    (0,−1), (1,−1), or, (1,0).
    """

    # Check if the input coordinate is valid and the red player controls the cell
    if coord not in board or board[coord][0] != 'R':
        return board

    k = board[coord][1] # Get the power of the current stack
    if k == 1:
        # If the power of the stack is 1, the spread action is equivalent to a move in the given direction
        new_coord = random_move(coord, direction)
        board[new_coord] = board.pop(coord)
        return board

    # Define the offset for the chosen direction
    rd, qd = direction

    # Remove the current stack from the cell
    board[coord] = ('R', 0)

    # Spread the tokens to the adjacent cells
    new_coords = []
    for i in range(k):
        # Compute the coordinates of the adjacent cell to update
        r = coord[0] + rd * (i+1)
        q = coord[1] + qd * (i+1)

        # Wrap around if the coordinate is out of bounds
        if r < 0:
            r = 6
            q = (q + 4) % 7
        elif r > 6:
            r = 0
            q = (q + 3) % 7
        elif q < 0:
            q = 6
            r = (r + 3) % 7
        elif q > 6:
            q = 0
            r = (r + 4) % 7

        # Increment the power of the adjacent cell by 1
        if (r, q) in board:
            color, power = board[(r, q)]
            if color == 'R':
                # If the red player controls the adjacent cell, add the token to the stack
                new_power = min(power + 1, 6)
                board[(r, q)] = ('R', new_power)
            elif color == 'B':
                # If the blue player controls the adjacent cell, take control of the stack
                new_power = min(power + 1, 6)
                board[(r, q)] = ('R', new_power)
        else:
            # If the adjacent cell is empty, add a new token to it
            board[(r, q)] = ('R', 1)
        new_coords.append((r, q))

    # Check if any stacks have been removed due to exceeding the maximum power
    for c in new_coords:
        if board[c][1] == 0:
            board.pop(c)

    return board