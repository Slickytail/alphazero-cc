import numpy as np
from scipy.signal import convolve2d
from typing import List, Tuple

BOARD_W = 9
BOARD_H = 7

class Game(object):
    """
    Represents a single game, all its history, and any information
    known about the child games.
    """

    NUM_ACTIONS = BOARD_W
    INPUT_SHAPE = (BOARD_H, BOARD_W, 3)
    # The board is 17x17=289 squares.
    # If we were to make the output be layered (rather than flat),
    # we would need 5.7 times more actions; and all of the extras would be impossible.

    def __init__(self, history: List[int]=None):
        self.history = []
        self.child_visits = []
        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.int)
        self.winner = 0
        self.turn = 1

        if history is not None:
            for move in history:
                self.apply(move)

    def terminal(self) -> bool:
        return self.winner > 0

    def terminal_value(self, player) -> float:
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def legal_actions(self) -> np.array:
        if self.terminal():
            return np.zeros_like(self.board)
        return (self.board[0] == 0).astype(np.float)

    def apply(self, action: int):
        # Put the move into the history
        self.history.append(action)
        slot = BOARD_H - np.argwhere(self.board[::-1,action] == 0)[0,0] - 1
        self.board[slot, action] = self.turn

        # Check for winner
        if has_won(self.board, self.turn):
            self.winner = self.turn
        else:
            self.turn = self.turn % 2 + 1

    def unapply(self, action: int, board):
        slot = np.argwhere(board[:,action] != 0)[0,0]
        board[slot, action] = 0

    def clone(self):
        return Game(list(self.history))

    def store_search_statistics(self, root):
        total_visits = sum(child.visit_count for child in root.children.values())
        visit_probs = np.zeros(self.NUM_ACTIONS)
        for (action, child) in root.children.items():
            visit_probs[action] = child.visit_count / total_visits
        self.child_visits.append(visit_probs)

    def make_image(self, state_index: int) -> np.array:
        board = self.board.copy()
        turn = self.turn

        if state_index < 0:
            state_index += len(self.history) + 1
            # state index -1 means last state.
        for col in reversed(self.history[state_index:]):
            self.unapply(col, board)
            turn = turn % 2 + 1
        opponent = turn % 2 + 1
        # Now board is the state of the board at that index.
        # 3 layers for tile state (empty, player piece, opponent piece)
        planes = np.array([
            board == 0,
            board == turn,
            board == opponent
        ], dtype=np.float)

        # We want to use channels_last convention, where the axes are (batches, y, x, channels).
        # Right now this is (C, Y, X). So to follow convention we'll move the channel axis to the end.
        return np.moveaxis(planes, 0, -1)


    def make_target(self, state_index: int) -> Tuple[float, np.array]:
        # Directly copied from pseudocode
        # Need to check to make sure we're getting the right index ...
        return (self.terminal_value(state_index % 2 + 1),
                self.child_visits[state_index])

    def to_play(self):
        return self.turn


horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.int)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

def has_won(board, player):
    for kernel in detection_kernels:
        if (convolve2d(board == player, kernel, mode="valid") == 4).any():
            return True
    return False
