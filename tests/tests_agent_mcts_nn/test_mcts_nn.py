import unittest
import numpy as np
from agents.agent_mcts_nn import MonteCarloTreeSearchNode_NN
from agents.agent_mcts_nn import explore_node_nn, simulate_game_nn
from game_utils import initialize_game_state, PLAYER1, PLAYER2

class TestMCTSNN(unittest.TestCase):
    def test_explore_node_nn(self):
        mock_board = initialize_game_state()
        mock_player = PLAYER2
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=mock_player)
        node.available_actions = np.array([3])
        explored_node = explore_node_nn(node)
        assert len(node.children) == 1

    def test_explore_node_nn_2(self):
        mock_board = initialize_game_state()
        mock_player = PLAYER2
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=mock_player)
        node.available_actions = []
        explored_node = explore_node_nn(node)
        assert len(explored_node.children) == 0

    def test_explore_node_nn_3(self):
        mock_board = initialize_game_state()
        mock_player = PLAYER2
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=mock_player)
        node.available_actions = np.array([3,4,5])
        explored_node = explore_node_nn(node)
        assert len(node.children) == 1

    def test_simulate_game_nn(self):
        mock_board =  np.array([[1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],], dtype=np.int8)

        mock_player = PLAYER1
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=mock_player)
        is_winner, current_player = simulate_game_nn(node)
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
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=mock_player)
        is_winner, current_player = simulate_game_nn(node)
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
        node = MonteCarloTreeSearchNode_NN(board=mock_board, player=mock_player)
        is_winner, winning_player = simulate_game_nn(node)
        assert is_winner is False
        assert winning_player is PLAYER1
