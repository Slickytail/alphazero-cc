from typing import List, Tuple
from tensorflow import keras
import numpy as np
import os

from constants import Config
from game_az_wrapper import Game
from search import mcts
import network

class ReplayBuffer:
    def __init__(self, config: Config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.rng = config.rng
        self.buffer = []

    def save_game(self, game: Game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def __next__(self):
        """
        Sample a bunch of moves from the game history,
        with each state having equal probability (ie, rather than each game).
        """
        game_lengths = np.array([len(g.history) for g in self.buffer])
        games = self.rng.choice(
            self.buffer,
            size=self.batch_size,
            p = game_lengths / sum(game_lengths))
        game_positions = ((g, self.rng.randint(len(g.history))) for g in games)
        X = np.stack([g.make_image(i) for (g, i) in game_positions])
        V, P = zip(*[g.make_target(i) for (g, i) in game_positions])
        targets = [np.stack(V), np.stack(P)]
        return X, targets


def train(config: Config):
    """
    Run the training loop.
    This means doing a number of self-play games,
    saving the states of those games,
    and then running a few batches of gradient descent on the network.
    """
    model = network.get_network(config)
    replays = ReplayBuffer(config)

    # Make a directory to save callbacks in
    os.makedirs(config.checkpoint_dir, exists_ok=True)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(config.checkpoint_dir, config.checkpoint_fname),
        save_freq = config.checkpoint_interval
        )
    # Single threaded version.
    for i in range(config.training_steps):
        # Self-play some games
        for _ in range(config.games_per_step):
            replays.save_game(self_play(network, config))
        model.fit(replays, steps_per_epoch=config.batches_per_step,
                callbacks = [checkpoint_callback])

def self_play(network, config: Config) -> Game:
    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = mcts(game, network, config)
        game.apply(action)
        game.store_search_statistics(root)
    return game
