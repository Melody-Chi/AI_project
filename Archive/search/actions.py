import random
from copy import deepcopy


def spread_a_unit(coord, dir):
    """
    A method that inputs the current coordinates of a token on a board,
    and the direction in which the token should be moved,
    then returns the amended coordinate tuple.
    """

    # Maps each direction string to a tuple representing the change in coordinates
    dir_book = {
        "N": (1, -1),
        "S": (-1, 1),
        "NW": (0, -1),
        "SE": (0, 1),
        "NE": (1, 0),
        "SW": (-1, 0),
    }

    # Move token 1 unit to the provided direction
    new_coord = list(map(lambda i, j: i + j, coord, dir_book[dir]))
    # update coord of a token if it is out of the boundrary
    amend_new_coord = tuple([0 if i == 7 else 6 if i == -1 else i for i in new_coord])

    return amend_new_coord


def update_board(board, coord, dir):
    """
    This function updates the game board with a new move by a player's token.
    It takes three parameters: the game board, the current position of the token,
    and the direction in which the player wants to move the token.
    """
    next_coord = spread_a_unit(coord, dir)
    current_board = board

    # Check if there is already a token in the next position.
    # And calculate the power of the result of the collision of the two tokens.
    if next_coord in current_board:
        next_color, next_power = current_board[next_coord]
        result_power = next_power + 1
        # If the resulting power is 7, the token at the next position is removed from the board.
        if result_power == 7:
            current_board.pop(next_coord)
        else:
            current_board[next_coord] = ("r", result_power)
    else:
        current_board[next_coord] = ("r", 1)

    return current_board


def spread(board, coord, dir):
    """
    Method for implementing the spread behavior of a token in the game
    """
    color, power = board[coord]
    current_coord = coord
    current_board = board.copy()

    # Iterate through the number of times specified by the power value,
    # updated with the new position returned by the spread_a_unit function.
    for i in range(power):
        current_board = update_board(current_board, current_coord, dir)
        current_coord = spread_a_unit(current_coord, dir)

    # Remove the original token from the board.
    current_board.pop(coord)

    # Return the spread board.
    return current_board


def random_spread(board, amount):
    """
    Takes a dictionary board and an integer amount as input and
    returns a modified board after spreading a random red token in a random direction.
    """
    dir_book = ["N", "S", "NE", "NW", "SE", "SW"]
    init_board = board.copy()

    # Repeating the random spread process `amount` times
    for k in range(amount):
        red_token = []

        # Collecting all the red tokens present in the board
        for token in init_board:
            if "r" in init_board[token]:
                red_token.append(token)

        rand_red_token = random.choice(red_token)
        rand_dir = random.choice(dir_book)

        # Spreading the token in the selected direction and updating the board
        init_board = spread(init_board, rand_red_token, rand_dir)

    # Returning the updated board
    return init_board
