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

    def sample_batch(self) -> Tuple[np.array, List[np.array]]:
        """
        Sample a bunch of moves from the game history,
        with each state having equal probability (ie, rather than each game).
        """
        game_lengths = np.array([len(g.history) for g in self.buffer])
        games = self.rng.choice(
            self.buffer,
            size = self.batch_size,
            p = game_lengths / sum(game_lengths))
        game_positions = [(g, self.rng.integers(len(g.history))) for g in games]
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
    model = network.get_network(True, config)
    replays = ReplayBuffer(config)

    def replays_generator():
        while True:
            b = replays.sample_batch()
            (X, (V, P)) = b
            print()
            print((V, np.max(P), np.mean(P)))
            print(model(X))
            yield b

    # Make a directory to save callbacks in
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(config.checkpoint_dir, config.checkpoint_fname),
            save_freq = config.checkpoint_interval
        )
    # Single threaded version.
    for i in range(config.training_steps // config.batches_per_step):
        # Self-play some games
        progbar = keras.utils.Progbar(config.games_per_step)
        for g in range(config.games_per_step):
            replays.save_game(self_play(model, config))
            progbar.update(g+1)
        # Might want to create a better training loop than this
        model.fit(replays_generator(), callbacks = [checkpoint_callback],
                steps_per_epoch=config.batches_per_step,
                initial_epoch = i, epochs = i+1)

def self_play(model: keras.layers.Layer, config: Config) -> Game:
    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = mcts(game, model, config)
        game.apply(action)
        game.store_search_statistics(root)
    return game

config = Config()
train(config)
