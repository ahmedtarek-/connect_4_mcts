from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable


BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

# PLAYER = NO_PLAYER
# OPPONENT = NO_PLAYER
GLOBAL_TIME = 5

class GameState(Enum):
    """
    An enum representing whether a game is won, draw, or still going.

    Enum Values
    -----------
        - IS_WIN: Represents the game state where a player has won.
        - IS_DRAW: Represents the game state where the game is a draw.
        - STILL_PLAYING: Represents the game state where the game is still in progress.
    """
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    initial_board_state = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
    return initial_board_state

class MoveStatus(Enum):
    """
    An enum to represent possible different cases for user input.

    Enum Values
    -----------
        - IS_VALID: Represents a valid move.
        - WRONG_TYPE: Represents an invalid move due to input not being a number.
        - NOT_INTEGER: Represents an invalid move due to input not being an integer
          or not equal to an integer in value.
        - OUT_OF_BOUNDS: Represents an invalid move due to input being out of bounds.
        - FULL_COLUMN: Represents an invalid move due to the selected column being full.
    """
    IS_VALID = 1
    WRONG_TYPE = 'Input is not a number.'
    NOT_INTEGER = ('Input is not an integer, or isn\'t equal to an integer in '
                   'value.')
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    board_strings = []

    # Add a separator line at the beginning
    separator = '=' * (BOARD_COLS * 2)
    board_strings.append(f"|{separator}|")

    for row in  reversed(board):
        row_str = '|'
        for piece in row:
            if piece == NO_PLAYER :
                row_str += NO_PLAYER_PRINT
            elif piece == PLAYER1:
                row_str += PLAYER1_PRINT
            elif piece == PLAYER2:
                row_str += PLAYER2_PRINT
            row_str += ' '
        row_str += '|'
        board_strings.append(row_str)

    board_strings.append(f"|{separator}|")
    column_numbers = '| 0 1 2 3 4 5 6 |'
    board_strings.append(column_numbers)

    return '\n'.join(board_strings)


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    lines = pp_board.strip().split('\n')[::-1]

    board_data = []

    for line in lines[2:-1]:  # Skip the first, last, and column number lines
        row_data = []
        i = 1
        while i < len(line):
            char = line[i]
            if char == NO_PLAYER_PRINT:
                piece = 0
            elif char == PLAYER1_PRINT:
                piece = 1
            elif char == PLAYER2_PRINT:
                piece = 2
            else:
                i += 2  # Skip any other characters
                continue
            row_data.append(piece)
            i += 2  # Move to the next character

        board_data.append(row_data)

    reversed_board_data = board_data[::-1]  # Reverse the order of rows
    board = np.array(reversed_board_data, dtype=np.int8)
    return board



def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).
    """
    if action < 0 or action >= BOARD_COLS:
        raise ValueError("Invalid action. Action must be in the range [0, 6]")

    column_index = action

    for row in range(BOARD_ROWS):
        if board[row, column_index] == NO_PLAYER:
            board[row, column_index] = player
            return board

   # raise ValueError("Column is already full. Invalid action.")



def check_horizontal(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if a win happened in the horizontal dimension
    """
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS - 3):  # Check all columns
            if all(board[row, col + i] == player for i in range(4)):
                return True
    return False


def check_vertical(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if a win happened in the vertical dimension
    """
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 3):
            if all(board[row + i, col] == player for i in range(4)):
                return True
    return False

def check_diagonal_bottom_left_top_right(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if a win happened in a diagonal
    """
    for row in range(3, BOARD_ROWS):
        for col in range(BOARD_COLS - 3):
            if all(board[row - i, col + i] == player for i in range(4)):
                return True
    return False

def check_diagonal_top_left_bottom_right(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if a win happened in a diagonal
    """
    for row in range(BOARD_ROWS - 3):
        for col in range(BOARD_COLS - 3):
            if all(board[row + i, col + i] == player for i in range(4)):
                return True
    return False

def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    return (
        check_horizontal(board, player)
        or check_vertical(board, player)
        or check_diagonal_bottom_left_top_right(board, player)
        or check_diagonal_top_left_bottom_right(board, player)
    )


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN

    if np.all(board != NO_PLAYER):
        return GameState.IS_DRAW

    return GameState.STILL_PLAYING

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

def user_move(board: np.ndarray,
              _player: BoardPiece,
              saved_state: Optional[SavedState]) -> tuple[PlayerAction, SavedState]:
    """
    Handles when a user makes a move
    """
    is_valid_move = False
    while not is_valid_move:
        input_move_string = query_user(input)
        try:
            is_valid_move = handle_illegal_moves(board, input_move_string)
        except TypeError:
            print('Not the right format, try an integer.')
        except IndexError:
            print('Selected integer is not in the range of possible columns (0 - 6).')
        except ValueError:
            print('Selected column is full.')
    input_move_integer = PlayerAction(input_move_string)
    return input_move_integer, saved_state


def query_user(prompt_function: Callable) -> str:
    """
    Prompt user for input
    """
    usr_input = prompt_function("Column? ")
    return usr_input

def handle_illegal_moves(board: np.ndarray, column: PlayerAction) -> bool:
    """
    Handles a user input that doesn't make sense
    """
    try:
        column = PlayerAction(column)
    except:
        raise TypeError

    is_in_range = PlayerAction(0) <= column <= PlayerAction(6)
    if not is_in_range:
        raise IndexError

    is_open = board[-1, column] == NO_PLAYER
    if not is_open:
        raise ValueError

    return True


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    """
    Simulates the game between two agents, usually 
    between a human agent (prompt) and a smart agent
    (minimax or mcts)
    """
    import time

    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                print(f"Move time: {time.time() - t0:.3f}s")
                board = apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                    playing = False
                    break
                print("\n")

def encode_state(state, current_player: BoardPiece) -> np.ndarray:
    """
    Encodes the state of the board to three arrays:
        - One representing the current player
        - One representing the opposite player
        - One representing the empty space
    """
    other_player = get_opponent(current_player)
    encoded_state = np.stack((state == 0, state == current_player, state == other_player)).astype(np.float32)
    if len(state.shape) == 3:
        encoded_state = np.swapaxes(encoded_state, 0, 1)
    return encoded_state


def get_free_columns(d_board: np.ndarray) -> list:
    """
    Retrieve a list of free columns

    :parameter d_board: the playing board of type np.ndarray

    :return: returns a list of column index numbers.
    """
    col_list = []
    for col in range(7):
        if d_board[-1, col] == NO_PLAYER:
            col_list.append(col)
    return col_list

def get_opponent(player: BoardPiece) -> BoardPiece:
    """
    Get the opponent (human) player.

    Parameters
    ----------
    player : BoardPiece
        The current player.

    Returns
    -------
    BoardPiece
        The opponent (human) player.
    """

    return PLAYER1 if player == PLAYER2 else PLAYER2

def get_current_player(board: np.ndarray) -> BoardPiece:
    num_moves = np.sum(board != NO_PLAYER)
    return PLAYER1 if num_moves % 2 == 0 else PLAYER2
