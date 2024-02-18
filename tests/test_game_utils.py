import unittest
import numpy as np
from game_utils import get_free_columns, get_opponent, get_current_player, encode_state
from game_utils import initialize_game_state, PLAYER1, PLAYER2

class TestGameUtils(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_get_free_columns_1(self):
        sample_board = np.array([[0, 2, 2, 1, 1, 0, 0],
                                 [0, 2, 1, 2, 2, 0, 0],
                                 [0, 1, 2, 1, 2, 0, 0],
                                 [0, 1, 1, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0],
                                 [0, 2, 0, 0, 0, 0, 0],], dtype=np.int8)

        available_cols = get_free_columns(sample_board)
        assert available_cols == [0,2,3,4,5,6]

    def test_get_free_columns_2(self):
        sample_board = np.array([[0, 2, 2, 1, 1, 0, 0],
                                 [0, 2, 1, 2, 2, 0, 0],
                                 [0, 1, 2, 1, 2, 0, 0],
                                 [0, 1, 1, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0],
                                 [1, 2, 2, 0, 0, 0, 1],], dtype=np.int8)

        available_cols = get_free_columns(sample_board)
        assert available_cols == [3,4,5]

    def test_get_free_columns_3(self):
        sample_board = np.array([[0, 2, 2, 1, 1, 0, 0],
                                 [0, 2, 1, 2, 2, 0, 0],
                                 [0, 1, 2, 1, 2, 0, 0],
                                 [0, 1, 1, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0],
                                 [1, 2, 1, 2, 1, 1, 1],], dtype=np.int8)

        available_cols = get_free_columns(sample_board)
        assert available_cols == []

    def test_get_opponent(self):
      test_player = PLAYER1
      tested_player = get_opponent(test_player)
      assert tested_player == PLAYER2

    def test_get_opponent_2(self):
      test_player = PLAYER2
      tested_player = get_opponent(test_player)
      assert tested_player == PLAYER1

    def test_get_current_player(self):
        mock_board = initialize_game_state()
        current_player = get_current_player(mock_board)
        assert current_player == PLAYER1

    def test_get_current_player_2(self):
        mock_board = initialize_game_state()
        mock_board[0][0] = 1
        current_player = get_current_player(mock_board)
        assert current_player == PLAYER2


    def test_get_current_player_3(self):
        mock_board = initialize_game_state()
        mock_board[0][0] = 1
        mock_board[0][1] = 2
        current_player = get_current_player(mock_board)
        assert current_player == PLAYER1

    def test_encode_state(self):
        mock_board = initialize_game_state()
        encoded = encode_state(mock_board, PLAYER1)
        assert encoded.shape == (3, 6, 7)
        assert encoded.dtype == np.float32

    def test_encode_state_2(self):
        mock_board =  np.array([[1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],], dtype=np.int8)
        first_board =  np.array([[0, 0, 0, 0, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],], dtype=np.int8)
        encoded = encode_state(mock_board, PLAYER1)
        assert np.all(encoded[0] == first_board)
        assert np.all(encoded[1] == mock_board)
        assert np.all(encoded[2] == np.zeros((6,7),dtype = np.int8))

    def test_encode_state_3(self):
        mock_board =  np.array([[1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],], dtype=np.int8)
        first_board =  np.array([[0, 0, 0, 0, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],], dtype=np.int8)
        encoded = encode_state(mock_board, PLAYER2)
        assert np.all(encoded[0] == first_board)
        assert np.all(encoded[1] == np.zeros((6,7),dtype = np.int8))
        assert np.all(encoded[2] == mock_board)
