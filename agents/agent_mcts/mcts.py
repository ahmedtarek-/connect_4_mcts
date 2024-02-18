import time
import numpy as np
from .mcts_node import MonteCarloTreeSearchNode
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, GLOBAL_TIME
from game_utils import get_free_columns, apply_player_action, connected_four
from game_utils import PLAYER1, PLAYER2
from typing import Tuple, Optional, Callable

def toggle_player(current_player: BoardPiece) -> BoardPiece:
    """
    Returns the other player
    """
    return PLAYER2 if current_player == PLAYER1 else PLAYER1

def select_node(node: MonteCarloTreeSearchNode) -> MonteCarloTreeSearchNode:
    """
    Selects a node by invoking the selecting_node() method of
    MonteCarloTreeSearchNode
    """
    while node.children and not np.any(node.available_actions):
        node = node.selecting_node()
    return node

def explore_node(node: MonteCarloTreeSearchNode) -> MonteCarloTreeSearchNode:
    """
    Explores a node by invoking the expanding_node() method of
    MonteCarloTreeSearchNode
    """
    if np.any(node.available_actions):
        action = np.random.choice(node.available_actions)
        node = node.expanding_node(action)
    return node

def simulate_game(node: MonteCarloTreeSearchNode) -> Tuple[bool, np.int8]:
    """
    Simulates a game by invoking the simulate_game() method of
    MonteCarloTreeSearchNode
    """
    board, current_player = node.board.copy(), node.player
    while np.any(get_free_columns(board)):
        action = np.random.choice(get_free_columns(board))
        board = apply_player_action(board, action, current_player)
        if connected_four(board, current_player):
            return True, current_player
        current_player = toggle_player(current_player)
    return False, current_player

def back_propagation(node: MonteCarloTreeSearchNode, win: bool, current_player: np.int8):
    """
    Runs the back propagation of a mcts tree by invoking the count_visit_win() method of
    MonteCarloTreeSearchNode
    """
    score = 1 if win and current_player == node.player else -1 if win else 0
    while node:
        node.count_visit_win(score)
        node = node.parent

def find_best_move(root: MonteCarloTreeSearchNode) -> int:
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

def MCTS(board: np.ndarray, player: BoardPiece) -> PlayerAction:
    """
    Runs the mcts alogrithm by selecting a node, exploring it, simulating a game
    and running back_propagation to update values.
    """
    root = MonteCarloTreeSearchNode(board=board, player=player)
    end_time = time.time() + GLOBAL_TIME #examplary usage for 5 seconds run

    while time.time() < end_time:
        node = root
        node = select_node(node)
        node = explore_node(node)
        win, current_player = simulate_game(node)
        back_propagation(node, win, current_player)

    return find_best_move(root)

def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, SavedState]:
    """
    Generates an action based on the mcts algorithm.
    """
    action = MCTS(board, player)
    return action, SavedState()
