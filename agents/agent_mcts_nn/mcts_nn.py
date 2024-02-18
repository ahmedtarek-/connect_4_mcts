import time
import numpy as np
from .mcts_node import MonteCarloTreeSearchNode_NN
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, GLOBAL_TIME
from game_utils import get_free_columns, apply_player_action, connected_four, get_opponent
from game_utils import PLAYER1, PLAYER2
from typing import Tuple, Optional, Callable


def toggle_player(current_player):
    return PLAYER2 if current_player == PLAYER1 else PLAYER1

def select_node_nn(node: MonteCarloTreeSearchNode_NN) -> MonteCarloTreeSearchNode_NN:
    value = 0
    while not np.any(node.available_actions) and node.children != []:
        next_node, next_value = node.selecting_node_nn()
        if next_node == None:
          break
        node = next_node
        value = next_value
    return node, value


def explore_node_nn(node: MonteCarloTreeSearchNode_NN) -> MonteCarloTreeSearchNode_NN:
    if np.any(node.available_actions):
        action = np.random.choice(node.available_actions)
        node = node.expanding_node(action)
    return node


def simulate_game_nn(node: MonteCarloTreeSearchNode_NN) -> tuple:
    board = node.board.copy()
    win = False
    current_player = node.player
    while np.any(get_free_columns(board)) and not win:
        if current_player == PLAYER2:
            current_player = PLAYER1
        else:
            current_player = PLAYER2
        action = np.random.choice(get_free_columns(board))
        board = apply_player_action(board, action, current_player)
        win = connected_four(board, current_player)

    return win, current_player

def backpropagation_nn(node:MonteCarloTreeSearchNode_NN, nn_output_value):
    while node is not None:
        node.count_visit_win(nn_output_value)
        node = node.parent

def find_best_move(root):
    best_move = -1
    max_score = -np.inf
    for child in root.children:
        if connected_four(child.board, child.player):
            return child.action
        score = child.num_wins / child.num_visits
        if score > max_score:
            best_move = child.action
            max_score = score
    return best_move

def MCTS_nn(board: np.ndarray, player: BoardPiece) -> PlayerAction:
    root = MonteCarloTreeSearchNode_NN(board=board, player=player)
    end_time = time.time() + GLOBAL_TIME #examplary usage for 5 seconds run

    while time.time() < end_time:
        node = root
        node,value = select_node_nn(node)
        node = explore_node_nn(node)
        win, current_player = simulate_game_nn(node)
        backpropagation_nn(node, value)

    return find_best_move(root)

def generate_move_mcts_nn(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> object:
    global PLAYER
    global OPPONENT

    PLAYER = player
    if PLAYER == PLAYER1:
        OPPONENT = PLAYER2
    else:
        OPPONENT = PLAYER1
    action = MCTS_nn(board, player)
    return action, SavedState()
