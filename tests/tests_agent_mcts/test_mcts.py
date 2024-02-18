import unittest
import numpy as np
from agents.agent_mcts import MonteCarloTreeSearchNode, find_best_move
from agents.agent_mcts import toggle_player, select_node, explore_node, simulate_game, back_propagation
from game_utils import get_free_columns, get_opponent, get_current_player, encode_state
from game_utils import initialize_game_state, PLAYER1, PLAYER2


class TestMCTS(unittest.TestCase):
  def test_toggle_player(self):
    test_player = PLAYER1
    tested_player = toggle_player(test_player)
    assert tested_player == PLAYER2

  def test_toggle_player_2(self):
    test_player = PLAYER2
    tested_player = toggle_player(test_player)
    assert tested_player == PLAYER1

  def test_select_node_no_children(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER1
      node = MonteCarloTreeSearchNode(board = mock_board, player = mock_player)
      node.available_actions = np.array([])
      selected_node = select_node(node)
      assert selected_node is node

  def test_select_node_with_available_action(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER1
      node = MonteCarloTreeSearchNode(board = mock_board, player = mock_player)
      node.available_actions = np.array([4,5])
      child_node = MonteCarloTreeSearchNode(board = mock_board,parent=node)
      child_node.num_visits = 1
      child_node.num_win = 0
      child_node.available_actions = np.array([3,5])
      node.children.append(child_node)
      selected_node = select_node(node)
      assert selected_node is node

  def test_select_node(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER1
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      node.num_visits = 4
      node.num_wins = 1
      node.available_actions = np.array([])
      child_node = MonteCarloTreeSearchNode(board = mock_board,parent=node)
      child_node.num_visits = 1
      child_node.num_win = 0
      child_node.available_actions = np.array([3,5])
      node.children.append(child_node)
      selected_node = select_node(node)
      assert selected_node is child_node



  def test_explore_node(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER2
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      node.available_actions = np.array([3])
      explored_node = explore_node(node)
      assert len(node.children) == 1

  def test_explore_node_2(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER2
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      node.available_actions = []
      explored_node = explore_node(node)
      assert len(explored_node.children) == 0

  def test_explore_node_3(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER2
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      node.available_actions = np.array([3,4,5])
      explored_node = explore_node(node)
      assert len(node.children) == 1

  def test_simulate_game(self):
      mock_board =  np.array([[1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],], dtype=np.int8)

      mock_player = PLAYER1
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      is_winner, current_player = simulate_game(node)
      assert is_winner is True
      assert current_player == PLAYER1

  def test_simulate_game_loose(self):
      mock_board =  np.array([[1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],], dtype=np.int8)

      mock_player = PLAYER2
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      is_winner, current_player = simulate_game(node)
      assert is_winner is True
      assert current_player == PLAYER1

  def test_simulate_game_draw(self):
      mock_board = np.array([
          [1, 2, 1, 2, 1, 2, 1],
          [1, 2, 1, 2, 1, 2, 1],
          [2, 1, 2, 1, 2, 1, 2],
          [2, 1, 2, 1, 2, 1, 2],
          [1, 2, 1, 2, 1, 2, 1],
          [1, 2, 1, 2, 1, 2, 1]
      ], dtype = np.int8)
      mock_player = PLAYER1
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      is_winner, winning_player = simulate_game(node)
      assert is_winner is False
      assert winning_player is PLAYER1

  def test_back_propagation(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER1
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      back_propagation(node, win=True, current_player=mock_player)
      while node:
          assert node.num_wins == 1
          assert node.num_visits == 1
          node = node.parent

  def test_back_propagation_2(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER2
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      back_propagation(node, win=True, current_player=mock_player)
      while node:
          assert node.num_wins == 1
          assert node.num_visits == 1
          node = node.parent

  def test_back_propagation_draw(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER1
      node = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      back_propagation(node, win=False, current_player=mock_player)
      while node:
          assert node.num_wins == 0
          assert node.num_visits == 1
          node = node.parent

  def test_find_best_move(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER1
      root = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      winning_child = MonteCarloTreeSearchNode(parent=root, board=mock_board.copy(), player=mock_player)
      winning_child.num_visits = 3
      winning_child.num_wins = 3
      winning_child.action = 3
      root.children.append(winning_child)
      best_move = find_best_move(root)
      assert best_move == 3

  def test_find_best_move_2(self):
      mock_board = initialize_game_state()
      mock_player = PLAYER1
      root = MonteCarloTreeSearchNode(board=mock_board, player=mock_player)
      winning_child1 = MonteCarloTreeSearchNode(parent=root, board=mock_board.copy(), player=mock_player)
      winning_child1.num_visits = 3
      winning_child1.num_wins = 2
      winning_child1.action = 0
      non_winning_child2 = MonteCarloTreeSearchNode(parent=root, board=mock_board.copy(), player=mock_player)
      non_winning_child2.action = 1
      non_winning_child2.num_visits = 3
      non_winning_child2.num_wins = 0
      root.children.extend([winning_child1, non_winning_child2])
      best_move = find_best_move(root)
      assert best_move == 0
