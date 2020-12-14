import numpy as np
from checkers import CheckersGame, FULL_BOARD, Move
from itertools import product
from typing import List, Tuple

Action = int
class Game(object):
    """
    Represents a single game, all its history, and any information
    known about the child games.
    """

    NUM_ACTIONS = 121**2
    INPUT_SHAPE = (17, 17, 8)
    # The board is 17x17=289 squares.
    # If we were to make the output be layered (rather than flat),
    # we would need 5.7 times more actions; and all of the extras would be impossible.

    def __init__(self, history: List[Move]=None, board=None):
        self.history = history or []
        self.game = board or CheckersGame.new_game(2)
        # If we made a new board, apply the history.
        # Otherwise, assume that the history and board are matching.
        if board is None:
            for action in self.history:
                self.game.move(*action, validate=False)
        self.child_visits = []

    def terminal(self) -> bool:
        return self.game.winner is not None

    def terminal_value(self, player) -> float:
        # Might need to be in range [0, 1] instead...
        return 1.0 if self.game.winner == player else -1.0

    def legal_actions(self) -> np.array:
        actions = np.zeros(Game.NUM_ACTIONS)
        np.put(actions, [move_to_index[m] for m in self.game.get_legal(self.game.player_turn)], 1.0)
        return actions

    def apply(self, action: int):
        # Get the game-world representation instead of the move index
        move = index_to_move[action]
        # Put the move into the history
        self.history.append(move)
        # Apply the move to the current version of the game
        # Trust that the move was validated already
        self.game.move(*move, validate=False)

    def clone(self):
        return Game(list(self.history), self.game.clone())

    def store_search_statistics(self, root):
        total_visits = sum(child.visit_count for child in root.children.values())
        visit_probs = np.zeros(self.NUM_ACTIONS)
        for (action, child) in root.children.items():
            visit_probs[action] = child.visit_count / total_visits
        self.child_visits.append(visit_probs)

    def make_image(self, state_index: int) -> np.array:
        board = self.game._board.copy()
        turn = self.game.player_turn
        # TODO: Check for off-by-one error here.
        for (start, end) in self.history[state_index::-1]:
            board[start] = board[end]
            board[end] = 0

            turn = (turn + 1) % 2
        opponent = (turn + 1) % 2
        # Now board is the state of the board at that index.
        # Layout is as follows:
        # 3 layers for tile state (empty, player piece, opponent piece)
        #   (optionally repeated T times for history)
        # 3 layers for marking special tile types (not part of the board, player goal, opponent goal)
        # 1 layer with total move count (as a real number -- nothing special here)
        # 1 layer with whether the current moved first or second.
        planes = np.array([
            # The current player's goal
            _goals[turn],
            # The opponent's goal
            _goals[opponent],
            # Whether each space is blocked or not
            _blocked_spaces,
            # Number of moves made so far
            np.full(FULL_BOARD.shape, len(self.history)), # divided by something???
            # the index of the current player
            np.full(FULL_BOARD.shape, turn),
            
            # Now the potentially-repeated planes
            # empty spaces
            board == 0,
            # player's pieces
            board == self.game.colors[turn],
            # opponent's pieces
            board == self.game.colors[opponent]
        ], dtype=np.float32)

        # We want to use channels_last convention, where the axes are (batches, y, x, channels).
        # Right now this is (C, Y, X). So to follow convention we'll move the channel axis to the end.
        return np.moveaxis(planes, 0, -1)


    def make_target(self, state_index: int) -> Tuple[float, np.array]:
        # Directly copied from pseudocode
        # Need to check to make sure we're getting the right index ...
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return self.game.player_turn

# We'll precompute some features here
_blocked_spaces = FULL_BOARD == -1
_goals = [FULL_BOARD == CheckersGame.opposite(c) for c in CheckersGame.PLAYER_COLOR[2]]

# I was using START_ZONES but numpy HATES ME
# Literally I want to bash my head in.
# I had a 2d array and a python list of 2-tuples.
# And I wanted to set the value in the 2d array to 1 for each tuple in the list.
# and numpy hates me,

index_to_move = [(start, end) for (start, end) in product(np.ndindex(FULL_BOARD.shape), repeat=2)
                 if FULL_BOARD[start] != -1 and FULL_BOARD[end] != -1]
move_to_index = {v: i for (i, v) in enumerate(index_to_move)}
