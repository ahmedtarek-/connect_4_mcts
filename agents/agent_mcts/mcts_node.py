import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, GameState
from game_utils import get_free_columns, check_end_state, apply_player_action
from game_utils import PLAYER1, PLAYER2
from typing import Optional, Callable

class MonteCarloTreeSearchNode:
    """
    A class that represents a node of a monte carlo tree search algorithm
    a node typically has a parent and children

    Attributes
    ----------
        - action: The action taken to reach this node.
        - parent: The parent node of this node.
        - board: The game board state associated with this node.
        - player: The player associated with this node.
        - children: List of child nodes.
        - num_wins: Number of wins during simulation.
        - num_visits: Number of visits during simulation.
        - available_actions: Array of available actions for the current node.
        - model: Connect4Model representing the neural network model for predictions.

    Methods
    -------
        - get_available_actions(self): Returns an array of available actions.
        - count_visit_win(self, score): Increments the number of visits and number of wins for the current node.
        - selecting_node_nn(self): Calculates the best child based on the neural network predictions.
        - expanding_node(self, action): Expands the node by applying the player action and creating a new child.
    """

    def __init__(self, action=None, parent=None, board=None, player=None):
        self.action = action
        self.parent = parent
        self.board = board
        self.player = player
        self.children = []
        self.num_wins = 0
        self.num_visits = 0
        self.available_actions = self.get_available_actions()

    def get_available_actions(self) -> np.ndarray:
        """
        Returns an array of available actions
        """
        if check_end_state(self.board, self.player) == GameState.IS_WIN:
          return np.array([])
        return np.array(get_free_columns(self.board))

    def count_visit_win(self, score: int):
        """
        Increments the number of visits and number of wins for the current node
        """
        self.num_visits += 1
        self.num_wins += score

    def calculate_ucb_score(self, child) -> float:
        """
        Calculates the ucb score based on the node and given child
        """
        exploration_factor = np.sqrt(2)
        exploitation_term = child.num_wins / child.num_visits
        exploration_term = np.sqrt(np.log(self.num_visits) / child.num_visits)
        return exploitation_term + exploration_factor * exploration_term

    def selecting_node(self):
        """
        Calculates the best child to use it later in expanding
        """
        best_child = None
        highest_score = float('-inf')

        for child in self.children:
            score = self.calculate_ucb_score(child)
            if score > highest_score:
                best_child = child
                highest_score = score
        return best_child

    def expanding_node(self, action: PlayerAction):
        """
        Expands the node by applying the player action and creating a new child
        """
        opponent = PLAYER2 if self.player == PLAYER1 else PLAYER1
        new_board = apply_player_action(self.board.copy(), action, opponent)
        child = MonteCarloTreeSearchNode(action=action, parent=self, board=new_board, player=opponent)
        self.children.append(child)
        self.available_actions = list(set(self.available_actions) - set([action]))#not visit htis action again
        return child
