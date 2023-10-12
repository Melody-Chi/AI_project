# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    Board, PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir, IllegalActionException
from typing import List, Optional
from copy import deepcopy
from referee.game.constants import *
import hashlib
import time
from typing import Union

# This is the entry point for your game playing agent. Currently the agent
# simply spawns a token at the centre of the board if playing as RED, and
# spreads a token at the centre of the board if playing as BLUE. This is
# intended to serve as an example of how to use the referee API -- obviously
# this is not a valid strategy for actually playing the game!

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent.
        """
        self._color = color
        self.board = Board()  # Create a new board instance
        self.board.render()

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as red")
            case PlayerColor.BLUE:
                print("Testing: I am playing as blue")

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        # match self._color:
        #     case PlayerColor.RED:
        #         return SpawnAction(HexPos(3, 3))
        #     case PlayerColor.BLUE:
        #         # This is going to be invalid... BLUE never spawned!
        #         return SpreadAction(HexPos(3, 3), HexDir.Up)
        adaptive_search_depth = adaptive_depth(self.board, 3, 5)
        time_limit = 2
        maximizing_player = self._color == PlayerColor.RED        
        best_action = find_best_action(self._color, self.board, adaptive_search_depth, time_limit, maximizing_player)
        return best_action

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        try:
            self.board.apply_action(action)
        except IllegalActionException as e:
            print(f"Error: Illegal action '{action}' from player {color}: {e}")

        # Your previous code for printing actions
        match action:
            case SpawnAction(cell):
                print(f"Testing: {color} SPAWN at {cell}")
            case SpreadAction(cell, direction):
                print(f"Testing: {color} SPREAD from {cell}, {direction}")


    
    
# 提取所有棋子颜色的坐标 & 空坐标，并存在list里面
def extract_positions(board, color: Optional[PlayerColor] = None) -> List[HexPos]:
        positions = []

        for position, cell_state in board._state.items():
            if color is None and cell_state.player is None:  # Extract empty positions
                positions.append(position)
            
            elif cell_state.player == color:  # Extract colored positions
                positions.append(position)
                # print(f"Adding colored position: {position}")

        return positions

    
# 给每个棋子都设定6个方向的spread
def generate_spread_actions(color, board) -> List[SpreadAction]:
    colored_positions = extract_positions(board, color)
    spread_actions = []

    for position in colored_positions:
        for direction in HexDir:
            spread_action = SpreadAction(cell=position, direction=direction)
            spread_actions.append(spread_action)

    return spread_actions

# SPAWN actions
def generate_spawn_actions(color, board) -> List[SpawnAction]:
    spawn_actions = []
    empty_list = extract_positions(board)

    for position in empty_list:
        if board._within_bounds(position):
            spawn_action = SpawnAction(cell=position)
            spawn_actions.append(spawn_action)

    return spawn_actions


def get_legal_actions(color, board) -> List[Union[SpawnAction, SpreadAction]]:
    spawn_actions = generate_spawn_actions(color, board)
    spread_actions = generate_spread_actions(color, board)
    all_actions = spawn_actions + spread_actions

    red_power = board._color_power(PlayerColor.RED)
    blue_power = board._color_power(PlayerColor.BLUE)

    if(red_power+blue_power <= 49):
        return all_actions
    

def is_terminal(game_state):
    return game_state.game_over

def hex_distance(a: HexPos, b: HexPos) -> int:
    return (abs(a.r - b.r) + abs(a.q - b.q) + abs(a.r + a.q - b.r - b.q)) // 2

def evaluate_spawn_position(position: HexPos, red_positions: List[HexPos], blue_positions: List[HexPos]) -> float:
    """
    计算所有可能spawn位置的分数来确定一个最佳spawn的位置。
    分数是根据与离spawn位置下棋的最近的红棋和蓝棋的距离确定的
    其中与最近的蓝棋的距离分数占比较大 就是优先避开蓝棋。
    """
    nearest_red_distance = 0
    nearest_blue_distance = 0
    for red_pos in red_positions:
        if(red_pos):
            nearest_red_distance = min(nearest_red_distance, hex_distance(position, red_pos))
    for blue_pos in blue_positions:
        if(blue_pos):
            nearest_blue_distance = min(nearest_blue_distance, hex_distance(position, blue_pos))

    # Adjust the weights to prioritize different aspects of the evaluation
    red_weight = 1
    blue_weight = 3  # Increase the blue_weight to prioritize avoiding blue pieces

    return red_weight * nearest_red_distance - blue_weight * nearest_blue_distance


def evaluate_state_and_complexity(game_state: Board):
    """
    算当前游戏状态的score及complexity
    score 用来给每个state进行打分。根据红蓝双方棋手的力量和棋子数量的差异 以及spawn的位置的分数的最大值来计算
    complexity用来决定整个游戏的复杂性 判断搜索程度是深还是浅 计算是双方棋手的总力量和棋子数量之和。
    """
    red_positions = extract_positions(game_state, PlayerColor.RED)
    blue_positions = extract_positions(game_state, PlayerColor.BLUE)
    empty_positions = extract_positions(game_state)
    
    # Calculate spawn position evaluations
    spawn_evaluations = [evaluate_spawn_position(pos, red_positions, blue_positions) for pos in empty_positions]
    # 只考虑最佳的spawn action，不考虑和了
    best_spawn_evaluation = max(spawn_evaluations) if spawn_evaluations else 0

    # Calculate the total power of each player
    red_power = game_state._color_power(PlayerColor.RED)
    blue_power = game_state._color_power(PlayerColor.BLUE)

    # Calculate the difference in power between players
    power_difference = red_power - blue_power

    # Calculate the number of pieces per player
    # 计算每个棋子的数量
    red_pieces = len(red_positions)
    blue_pieces = len(blue_positions)

    # Calculate the difference between players' pieces
    # 计算棋子之间的数量差
    piece_difference = red_pieces - blue_pieces

    # Assign weights to the factors and calculate the final score
    power_weight = 0.5
    piece_weight = 1.0
    spawn_weight = 0.2
    # 这里就用score来给每个state进行打分。根据红蓝双方棋手的力量和棋子数量的差异 以及所有可能spawn的位置的分数之和计算的
    score = (
        power_weight * power_difference
        + piece_weight * piece_difference
        + spawn_weight * best_spawn_evaluation
    )

    # Calculate complexity
    complexity = (red_power + blue_power) * power_weight + (red_pieces + blue_pieces) * piece_weight

    return score, complexity


# Define a function that adjusts the depth based on the complexity
def adaptive_depth(game_state, min_depth, max_depth):
    score, complexity = evaluate_state_and_complexity(game_state)

    low_complexity_threshold = 10
    high_complexity_threshold = 20

    if complexity <= low_complexity_threshold:
        return max_depth
    elif complexity >= high_complexity_threshold:
        return min_depth
    else:
        # Linear interpolation between min_depth and max_depth
        depth_range = max_depth - min_depth
        complexity_range = high_complexity_threshold - low_complexity_threshold
        depth = max_depth - ((complexity - low_complexity_threshold) / complexity_range) * depth_range
        return int(depth)
    

def score_action(action, game_state, maximizing_player):
    new_state = deepcopy(game_state)
    new_state.apply_action(action)
    score, complexity = evaluate_state_and_complexity(game_state)

    # Invert the score if it's Blue's turn (minimizing player)
    if not maximizing_player:
        score = -score

    return score

def sorted_legal_actions(color, game_state, maximizing_player):
    legal_actions = get_legal_actions(color, game_state)
    return sorted(legal_actions, key=lambda action: score_action(action, game_state, maximizing_player), reverse=True)

# Transposition table是一个缓存，它存储了游戏树中先前评估的位置的结果，允许搜索算法重复使用这些结果并节省时间。为了实现换位表，你可以使用Python字典来存储不同游戏状态的评估分数。
# create a hash function to represent the game state as a unique key.
# 通过使用换位表，带有α-β修剪的最小化搜索将重新使用以前探索过的游戏状态的评估分数，从而降低时间和空间的复杂性。这种优化的有效性取决于游戏树中遇到的转置的数量。
def hash_game_state(game_state):
    # Convert the game_state._state into a sorted list of tuples
    board_list = sorted([(cell, state.player, state.power) for cell, state in game_state._state.items()])

    # Convert the sorted list of tuples into a string
    board_string = str(board_list)

    # Create a hash from the board_string
    return hashlib.md5(board_string.encode('utf-8')).hexdigest()

def minimax_alpha_beta(game_state, depth, alpha, beta, maximizing_player, transposition_table, start_time, time_limit):
    # 检查是否超过了时间限制 如果是直接return None
    if time.time() - start_time > time_limit:
        return None

    # 如果深度为0或游戏状态已经结束，评估游戏状态返回score
    if depth == 0 or is_terminal(game_state):
        score, complexity = evaluate_state_and_complexity(game_state)
        return score

    # 创建一个唯一的identifier，和transposition_table进行检查
    # 如果相同的游戏状态在以前被评估过，它的score直接从table中搜出来 可以节省计算时间
    game_state_hash = hash_game_state(game_state)
    if game_state_hash in transposition_table:
        return transposition_table[game_state_hash]

    legal_actions = sorted_legal_actions(game_state._turn_color, game_state, maximizing_player)

    # 如果轮到max玩家，浏览每一个合法的行动，应用于游戏状态, 更新的游戏状态然后减少深度再call itself。
    # a值更新为探索后获得的最大分数。如果b值小于或等于a值，break，这样之后的分支直接剪掉 
    if maximizing_player:
        max_eval = float('-inf')
        for action in legal_actions:
            game_state.apply_action(action)
            eval = minimax_alpha_beta(game_state, depth - 1, alpha, beta, False, transposition_table, start_time, time_limit)
            game_state.undo_action()
            if eval is None:
                return None
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        # 记录下来分数存在transposition table中，可以被调用
        transposition_table[game_state_hash] = max_eval
        return max_eval
    
    # 如果是min玩家，一样的方法，但它要的是最小的score
    else:
        min_eval = float('inf')
        for action in legal_actions:
            game_state.apply_action(action)
            eval = minimax_alpha_beta(game_state, depth - 1, alpha, beta, True, transposition_table, start_time, time_limit)
            game_state.undo_action()
            if eval is None:
                return None
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[game_state_hash] = min_eval
        return min_eval


def find_best_action(color, board, max_depth, time_limit, maximizing_player):
    """
    去加深迭代 然后为当前棋手找到最佳行动。
    """
    best_eval = float('-inf') if maximizing_player else float('inf')
    best_action = None
    start_time = time.time()

    transposition_table = {}

    # loop了从1到` max_depth'的每个深度
    # 从浅的深度开始搜索，对于每个深度，得到legal action
    # 将每个action都应用在棋盘的一个copy，然后用minimax去评估结果 如果score优于目前发现的最佳分数，就更新best action
    # temp_best_eval = best_eval
    # temp_best_action = best_action
        
    legal_actions = get_legal_actions(color, board)
    # print(f"Legal actions for depth {depth}: {legal_actions}")

    for action in legal_actions:
        new_state = deepcopy(board)
        try:
            new_state.apply_action(action)
            if new_state._total_power <= 49:
                eval = minimax_alpha_beta(new_state, max_depth, float('-inf'), float('inf'), not maximizing_player, transposition_table, start_time, time_limit)
                    
                if eval is None:  # Time limit exceeded
                    # print("Time limit exceeded in find_best_action")
                    break

                if maximizing_player:
                    if eval > temp_best_eval:
                        # print(f"New best action (maximizing): {action}, eval={eval}")
                        temp_best_eval = eval
                        temp_best_action = action
                else:
                    if eval < temp_best_eval:
                        # print(f"New best action (minimizing): {action}, eval={eval}")
                        temp_best_eval = eval
                        temp_best_action = action
        except IllegalActionException:
            pass
            
        # Update best_eval and best_action if a better action was found
        # if maximizing_player and temp_best_eval > best_eval:
        #     best_eval = temp_best_eval
        #     best_action = temp_best_action
        # elif not maximizing_player and temp_best_eval < best_eval:
        #     best_eval = temp_best_eval
        #     best_action = temp_best_action

        # Check if the allotted time has passed, break out of the loop if it has
        current_time = time.time()
        if current_time - start_time > time_limit:
            break

    return best_action

