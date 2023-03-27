# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion

from search.move import random_move
from .utils import render_board
from search.update import update_board
from search.spread import spread


def search(input: dict[tuple, tuple]) -> list[tuple]:
    """
    This is the entry point for your submission. The input is a dictionary
    of board cell states, where the keys are tuples of (r, q) coordinates, and
    the values are tuples of (p, k) cell states. The output should be a list of 
    actions, where each action is a tuple of (r, q, dr, dq) coordinates.

    See the specification document for more details.
    """

    # The render_board function is useful for debugging -- it will print out a 
    # board state in a human-readable format. Try changing the ansi argument 
    # to True to see a colour-coded version (if your terminal supports it).
    
    # print(random_move(input, direction='up'))

    # board = render_board(input, ansi=True)


    

    # print(render_board(input, ansi=True))
#                          (6,0)    
#                     (5,0)    (6,1)    
#                 (4,0)    (5,1)    (6,2)    
#             (3,0)    (4,1)    (5,2)    (6,3)    
#         (2,0)    (3,1)    (4,2)    (5,3)    (6,4)    
#      b2     (2,1)     b1     (4,3)    (5,4)    (6,5)    
# (0,0)     b1     (2,2)    (3,3)    (4,4)    (5,5)    (6,6)    
#     (0,1)    (1,2)    (2,3)    (3,4)    (4,5)     r2     
#         (0,2)     b3     (2,4)    (3,5)    (4,6)    
#             (0,3)    (1,4)    (2,5)    (3,6)    
#                 (0,4)    (1,5)    (2,6)    
#                     (0,5)    (1,6)    
#                         (0,6)    

    # print(input)
    # new_board = update_board(input, (5,6), 'up')
    # print(render_board(new_board, ansi=True))

    print(input)
    spread_board = spread(input, (5,6), 'up')
    print(spread_board)
    print(render_board(spread_board, ansi=True))


    # Here we're returning "hardcoded" actions for the given test.csv file.
    # Of course, you'll need to replace this with an actual solution...
    return [
        (5, 6, -1, 1),
        (3, 1, 0, 1),
        (3, 2, -1, 1),
        (1, 4, 0, -1),
        (1, 3, 0, -1)
    ]
