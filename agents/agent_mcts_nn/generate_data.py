import numpy as np
from agents.agent_mcts import MonteCarloTreeSearchNode
from agents.agent_mcts import select_node, explore_node, simulate_game, back_propagation
from game_utils import encode_state

from game_utils import BOARD_COLS
from game_utils import initialize_game_state, apply_player_action, check_end_state, get_free_columns
from game_utils import GameState, get_opponent
from game_utils import PLAYER1, PLAYER2

NUM_MCTS_SIMULATIONS = 100

def MCTS_for_data_generation():
  train_examples = []
  current_player = PLAYER1
  state = initialize_game_state()

  while True:
      root = MonteCarloTreeSearchNode(board=state.copy(), player=current_player)
      simulate_mcts_search(root)
      action_probs = calculate_action_probabilities(root)
      train_examples.append((state.copy(), current_player, action_probs))
      action = select_action(action_probs, state)
      state = apply_player_action(state, action, current_player)
      game_result = check_end_state(state, current_player)

      if game_result != GameState.STILL_PLAYING:
          boards = []
          for history_state, history_current_player, history_action_probs in train_examples:
              encoded_board = encode_state(history_state.copy(),history_current_player)
              reward = 1 if game_result == GameState.IS_WIN and history_current_player == current_player else -1
              boards.append((encoded_board, history_action_probs, reward))
          return boards
      else:
          current_player = get_opponent(current_player)

def simulate_mcts_search(root):
    for _ in range(NUM_MCTS_SIMULATIONS):
        node = root
        node = select_node(node)
        node = explore_node(node)
        win, simulated_player = simulate_game(node)
        back_propagation(node, win, simulated_player)

def calculate_action_probabilities(root):
    action_probs = np.zeros(BOARD_COLS)
    for child in root.children:
        action_probs[child.action] = child.num_visits
    total_visits = np.sum(action_probs)
    if total_visits > 0:
        action_probs = action_probs / total_visits
    else:
        action_probs = np.ones(BOARD_COLS) / BOARD_COLS
    return action_probs

def select_action(action_probs, d_board):
    free_columns = get_free_columns(d_board)
    if not free_columns:
        return None

    filtered_probs = [action_probs[col] for col in free_columns]
    normalized_probs = np.array(filtered_probs) / np.sum(filtered_probs)

    action = np.random.choice(free_columns, p=normalized_probs)
    return action
