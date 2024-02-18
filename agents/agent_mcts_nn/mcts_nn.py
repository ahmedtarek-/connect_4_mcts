import time
import numpy as np
from .mcts_node import MonteCarloTreeSearchNode_NN
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, GLOBAL_TIME
from game_utils import get_free_columns, apply_player_action, connected_four, get_opponent
from game_utils import PLAYER1, PLAYER2
from typing import Tuple, Optional, Callable


def toggle_player(current_player: BoardPiece) -> BoardPiece:
    """
    Returns the other player
    """
    return PLAYER2 if current_player == PLAYER1 else PLAYER1

def select_node_nn(node: MonteCarloTreeSearchNode_NN) -> MonteCarloTreeSearchNode_NN:
    """
    Selects a node by invoking the selecting_node() method of
    MonteCarloTreeSearchNode
    """
    value = 0
    while not np.any(node.available_actions) and node.children != []:
        next_node, next_value = node.selecting_node_nn()
        if next_node == None:
          break
        node = next_node
        value = next_value
    return node, value


def explore_node_nn(node: MonteCarloTreeSearchNode_NN) -> MonteCarloTreeSearchNode_NN:
    """
    Explores a node by invoking the expanding_node() method of
    MonteCarloTreeSearchNode
    """
    if np.any(node.available_actions):
        action = np.random.choice(node.available_actions)
        node = node.expanding_node(action)
    return node


def simulate_game_nn(node: MonteCarloTreeSearchNode_NN) -> tuple:
    """
    Simulates a game by invoking the simulate_game() method of
    MonteCarloTreeSearchNode
    """
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
    """
    Runs the back propagation of a mcts tree by invoking the count_visit_win() method of
    MonteCarloTreeSearchNode
    """
    while node is not None:
        node.count_visit_win(nn_output_value)
        node = node.parent

def find_best_move(root):
    """
    Finds the best move given a root node
    """
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
    """
    Runs the mcts alogrithm by selecting a node, exploring it, simulating a game
    and running back_propagation to update values.
    """
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
    """
    Generates an action based on the mcts algorithm.
    """
    global PLAYER
    global OPPONENT

    PLAYER = player
    if PLAYER == PLAYER1:
        OPPONENT = PLAYER2
    else:
        OPPONENT = PLAYER1
    action = MCTS_nn(board, player)
    return action, SavedState()
