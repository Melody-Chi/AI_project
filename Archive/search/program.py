# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion

from .utils import render_board
from .actions import *

import heapq


def manhattan(node1: tuple, node2: tuple) -> int:
    """
    Extract the row (r) and column (q) values from both nodes
    """
    r1, q1 = node1
    r2, q2 = node2

    # Calculate the Manhattan distance between the two nodes
    # by taking the absolute difference of their row and column values
    # and adding them.
    return abs(r1 - r2) + abs(q1 - q2)


def h(red_nodes: list, blue_nodes: list):
    """
    Return the total Manhattan distance as the heuristic value
    """

    # Initialize a variable to keep track of the total Manhattan distance
    total = 0
    total = len(blue_nodes)

    
    # for red_node in red_nodes:
    #     for blue_node in blue_nodes:
    #         total += manhattan(red_node, blue_node)

    return total


class Board:
    """
    Represents a state in the game which are used in A* search.

    Separating the tokens on the board into red and blue tokens,
    and checking if the game has ended.
    """

    def __init__(self, board, parent_board=None, parent_node=None, pnode_move=None):
        """
        Initialize the Board object with the given board state, parent board state,
        parent node, and the move that was made to reach this state
        """
        self.board = board
        self.parent_board = parent_board
        self.parent_node = parent_node
        self.pnode_move = pnode_move

        self.g = 0
        self.h = 0
        self.f = 0

        self.red_tokens = []
        self.blue_tokens = []

    def __eq__(self, other):
        return (self.f == other.f) and (self.h == other.h)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.f < other.f) or ((self.f == other.f) and (self.h < other.h))

    def __gt__(self, other):
        return (self.f > other.f) or ((self.f == other.f) and (self.h > other.h))

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other)

    def seperate_tokens(self):
        """
        Separate the tokens on the board into red tokens and blue tokens
        """
        board = self.board
        self.red_tokens = []
        self.blue_tokens = []

        for token in self.board:
            if "r" in board[token]:
                self.red_tokens.append(token)
            elif "b" in board[token]:
                self.blue_tokens.append(token)

    def is_end(self):
        """
        Check if the game has ended by checking if there are no more blue tokens on the board
        """
        return len(self.blue_tokens) == 0


def compare(item1, item2):
    return item1[0] - item2[0]


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
    # print(render_board(input, ansi=True))

    # Here we're returning "hardcoded" actions for the given test.csv file.
    # Of course, you'll need to replace this with an actual solution...
    # return [(5, 6, -1, 1), (3, 1, 0, 1), (3, 2, -1, 1), (1, 4, 0, -1), (1, 3, 0, -1)]

    # Returns a list of tuples as a path from the given start to the given end in the given maze

    #  A dictionary mapping each direction
    dir_book = ["N", "S", "NE", "NW", "SE", "SW"]

    # A dictionary mapping each direction's cooridnate
    dir_book2 = {
        "N": (1, -1),
        "S": (-1, 1),
        "NW": (0, -1),
        "SE": (0, 1),
        "NE": (1, 0),
        "SW": (-1, 0),
    }

    # Create start board class
    start_board = Board(input)

    start_board.seperate_tokens()
    start_board.h = len(start_board.blue_tokens)

    start_board.f = start_board.g + start_board.h

    # Initialize both open and closed list
    heap = []
    closed_list = []

    # Add the start node
    heapq.heappush(heap, (start_board.f, start_board))

    # Loop until you find the end
    while len(heap) > 0:
        # Get the current node
        current_board = heapq.heappop(heap)[1]

        # add to closed list
        closed_list.append(current_board)

        # Found the goal, selects the node with the lowest f value from the open_list
        # and removes it from that list. Then adds that node to the closed_list.
        if current_board.is_end():
            path = []

            while current_board.parent_node is not None:
                path.append(
                    tuple(
                        list(current_board.parent_node)
                        + list(dir_book2[current_board.pnode_move])
                    )
                )
                current_board = current_board.parent_board
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for red_token in current_board.red_tokens:
            for move_dir in dir_book:  # Adjacent squares
                # print(f"{red_token} {move_dir}")
                child_board_status = spread(current_board.board, red_token, move_dir)
                # Create new node
                new_board = Board(
                    child_board_status, current_board, red_token, move_dir
                )

                # Append
                children.append(new_board)

        # Loop through children
        for child in children:
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_board.g + 1
            child.seperate_tokens()
            child.h = len(child.blue_tokens)
            child.f = child.g + child.h

            # # Child is already in the open list
            # for open_board in open_list:
            #     if child == open_board and child.g > open_board.g:
            #         continue

            # Add the child to the open list
            heapq.heappush(heap, (child.f, child))
