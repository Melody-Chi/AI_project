# AI Project: Chess Board Game Optimization

## Introduction

This project focuses on a 7x7 chessboard game between two AI agents. The objective for each player is to control all cells on the board. We've employed advanced algorithms and optimization techniques to analyze the best strategies for gameplay. By understanding and utilizing game dynamics, our agents make strategic decisions to dominate the board. 

### Game Mechanics:

- **Board Class**: Represents the game state, keeping track of player’s total power and number of cells controlled on the board.
- **Actions**:
  - **SpawnAction**: Allows a player to spawn a new piece in an empty cell.
  - **SpreadAction**: Permits the spread of an existing piece to neighboring cells.
- **Rewards**: The player gets rewarded in terms of increased cell control and power.

## Approach

### Minimax Algorithm with Alpha-Beta Pruning

- **Minimax**: A recursive algorithm used in two-player games to determine the optimal move. It works by simulating all possible game states to a certain depth.
- **Alpha-Beta Pruning**: An optimization technique that reduces the computational effort of the Minimax algorithm by eliminating branches in the search tree that won't influence the final decision.

### Evaluation Function

Determines the utility of a game state. It's based on:
- Difference in power between players.
- Number of pieces controlled by each player.
- Strategic priorities, like controlling more cells or increasing the power of pieces.

## Performance Evaluation

Comparative analysis between our Minimax approach and the Monte Carlo Tree Search (MCTS) algorithm:
- **MCTS**:
  - Efficient when game state is vast.
  - Doesn’t require domain-specific knowledge or an evaluation function.
  - Can be computationally demanding.
  
Considering search speed, depth, decision quality, and complexity, our results indicate Minimax with Alpha-Beta Pruning offers a more efficient solution.

## Algorithmic Optimizations & Technical Aspects

1. **Iterative Deepening**: Regulates time complexity by limiting and gradually increasing the depth of the search tree.
2. **Move Sorting**: Prioritizes highest-scoring actions for evaluation, optimizing time complexity.
3. **Transposition Tables**: Cache to store results of previously evaluated positions, reducing computational redundancy.
4. **Adaptive Depth**: Adjusts search depth based on the game state's complexity.
5. **Bitboard Representation**: Uses a compact data structure, the bitboard, to represent the board state, enabling faster operations.
6. **Stimulus Windows**: Combines with Alpha-Beta pruning to optimize search, especially when game state isn't overly complex.

## Supporting Work

- **Training Agent**: An agent that plays randomly was developed to aid in evaluating and refining our main agent's strategies.
- **Version Control with GitHub**: Enabled tracking and managing different versions of the game program, fostering collaboration and ensuring continuity in development.

## Conclusion

Through a combination of the Minimax algorithm, heuristic evaluations, and various optimization strategies, we've created an AI game agent that is strategically adept and computationally efficient.

---

## README

### Installation:

1. Clone the repository:
```
git clone <repository-url>
```
2. Navigate to the project directory:
```
cd <project-directory>
```
3. Install the required packages (Ensure Python 3.x is installed):
```
pip install -r requirements.txt
```

### Usage:

Run the main game script:
```
python main.py
```

### Contribution:

- Fork the repository.
- Create your feature branch (`git checkout -b feature/fooBar`).
- Commit your changes (`git commit -am 'Add some fooBar'`).
- Push to the branch (`git push origin feature/fooBar`).
- Create a new Pull Request.

### License:

This project is licensed under the MIT License.

---

**Note**: Ensure you provide the actual `repository-url` and `<project-directory>` in the README instructions.
