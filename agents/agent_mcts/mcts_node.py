import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, GameState
from game_utils import get_free_columns, check_end_state, apply_player_action
from game_utils import PLAYER1, PLAYER2
from typing import Optional, Callable

class MonteCarloTreeSearchNode:
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
        if check_end_state(self.board, self.player) == GameState.IS_WIN:
          return np.array([])
        return np.array(get_free_columns(self.board))

    def count_visit_win(self, score: int):
        self.num_visits += 1
        self.num_wins += score

    def calculate_ucb_score(self, child):
        exploration_factor = np.sqrt(2)
        exploitation_term = child.num_wins / child.num_visits
        exploration_term = np.sqrt(np.log(self.num_visits) / child.num_visits)
        return exploitation_term + exploration_factor * exploration_term

    def selecting_node(self):
        best_child = None
        highest_score = float('-inf')

        for child in self.children:
            score = self.calculate_ucb_score(child)
            if score > highest_score:
                best_child = child
                highest_score = score
        return best_child

    def expanding_node(self, action: PlayerAction):
        opponent = PLAYER2 if self.player == PLAYER1 else PLAYER1
        new_board = apply_player_action(self.board.copy(), action, opponent)
        child = MonteCarloTreeSearchNode(action=action, parent=self, board=new_board, player=opponent)
        self.children.append(child)
        self.available_actions = list(set(self.available_actions) - set([action]))#not visit htis action again
        return child
