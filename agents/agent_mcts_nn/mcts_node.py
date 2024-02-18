import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, Callable
from .network import Connect4Model
from game_utils import encode_state
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, GameState
from game_utils import get_free_columns, check_end_state, apply_player_action
from game_utils import PLAYER1, PLAYER2, BOARD_SHAPE, BOARD_COLS

class MonteCarloTreeSearchNode_NN:

    def __init__(self, action=None, parent=None, board=None, player=None):
        self.action = action
        self.parent = parent
        self.board = board
        self.player = player
        self.children = []
        self.num_wins = 0
        self.num_visits = 0
        self.available_actions = self.get_available_actions()
        self.model = Connect4Model(BOARD_SHAPE, 7)
        self.model.load_state_dict(torch.load('data/model_weights.pth'))

    def get_available_actions(self) -> np.ndarray:
        if check_end_state(self.board, self.player) == GameState.IS_WIN:
          return np.array([])
        return np.array(get_free_columns(self.board))

    def count_visit_win(self, score: int):
        self.num_visits += 1
        self.num_wins += score

    def selecting_node_nn(self):
        state_tensor = torch.tensor(encode_state(self.board.copy(), self.player), dtype=torch.float)
        out_pi, out_v = self.model(state_tensor.unsqueeze(0))
        action_probs = F.softmax(out_pi.view(BOARD_COLS), dim=0).detach().numpy()
        max_index = np.argmax(action_probs)
        max_value = action_probs[max_index]
        highest_score = max_value.item()
        best_child = max_index.item()

        if 0 <= best_child < len(self.children):
          return self.children[best_child], out_v
        else:
          return None, None

    def expanding_node(self, action: PlayerAction):
        opponent = PLAYER2 if self.player == PLAYER1 else PLAYER1
        new_board = apply_player_action(self.board.copy(), action, opponent)
        child = MonteCarloTreeSearchNode_NN(action=action, parent=self, board=new_board, player=opponent)
        self.children.append(child)
        self.available_actions = list(set(self.available_actions) - set([action]))#not visit htis action again
        return child
