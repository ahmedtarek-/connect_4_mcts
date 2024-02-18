import unittest
import numpy as np
from agents.agent_mcts_nn import MonteCarloTreeSearchNode_NN
from game_utils import initialize_game_state, PLAYER1, PLAYER2

class TestMCTSNodeNN(unittest.TestCase):
    def test_count_visit_win_nn(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        initial_num_wins = node.num_wins
        initial_num_visits = node.num_visits
        node.count_visit_win(1)
        assert node.num_wins == initial_num_wins + 1
        assert node.num_visits == initial_num_visits + 1

    def test_count_visit_loss_nn(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        initial_num_wins = node.num_wins
        initial_num_visits = node.num_visits
        node.count_visit_win(0)
        assert node.num_wins == initial_num_wins
        assert node.num_visits == initial_num_visits + 1


    def test_selecting_node_no_children(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        best_child = node.selecting_node_nn()
        assert best_child is None

    def test_selecting_node_nn(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        node.num_wins = 2
        node.num_visits = 4
        child_node = MonteCarloTreeSearchNode_NN(board=mock_board,parent=node)
        child_node.num_wins = 1
        child_node.num_visits = 2
        node.children.append(child_node)
        best_child = node.selecting_node_nn()
        assert best_child is child_node

    def test_selecting_node_nn_2(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        node.num_wins = 2
        node.num_visits = 4
        child_node1 = MonteCarloTreeSearchNode_NN(board=mock_board, parent=node)
        child_node2 = MonteCarloTreeSearchNode_NN(board=mock_board, parent=node)
        child_node1.num_wins = 5
        child_node1.num_visits = 10
        child_node2.num_wins = 8
        child_node2.num_visits = 12
        node.children.extend([child_node1, child_node2])
        best_child = node.selecting_node_nn()
        assert best_child is child_node2


    def test_expanding_node_nn(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player = PLAYER1)
        action = 3
        child = node.expanding_node(action)
        assert child.parent is node
        assert child.board is not None
        assert child.action == action


    def test_selecting_node_no_children_nn(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        best_child = node.selecting_node_nn()
        assert best_child is None

    def test_selecting_node_nn(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        node.num_wins = 2
        node.num_visits = 4
        child_node = MonteCarloTreeSearchNode_NN(board=mock_board,parent=node)
        child_node.num_wins = 1
        child_node.num_visits = 2
        node.children.append(child_node)
        best_child = node.selecting_node_nn()
        assert best_child is child_node

    def test_selecting_node_nn_2(self):
        mock_board = initialize_game_state()
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=PLAYER1)
        node.num_wins = 2
        node.num_visits = 4
        child_node1 = MonteCarloTreeSearchNode_NN(board=mock_board, parent=node)
        child_node2 = MonteCarloTreeSearchNode_NN(board=mock_board, parent=node)
        child_node1.num_wins = 5
        child_node1.num_visits = 10
        child_node2.num_wins = 8
        child_node2.num_visits = 12
        node.children.extend([child_node1, child_node2])
        best_child = node.selecting_node_nn()
        assert best_child is child_node2
