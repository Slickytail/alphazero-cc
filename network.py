import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2

import os

# the Game class there will tell us the action space size
# and the size of the network's input
from game_az_wrapper import Game
from constants import Config

def make_network(config: Config) -> keras.Model:
    """
    Create a Keras model.
    """
    tf.random.set_seed(config.seed)
    # Maybe we set dtype here?
    planes = keras.Input(shape=Game.INPUT_SHAPE, name="game_state")
    # AlphaZero uses 18 residual blocks.
    # We'll use five.
    # First, we have to have an initial non-residual layer to change the number of filters. 
    x = layers.Conv2D(64, kernel_size = (1, 1), padding='same',
            kernel_regularizer = L2(config.l2_decay))(planes)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    for _ in range(5):
        x = residual_block(x, 64, reg=config.l2_decay)
    
    # Policy Head
    p = layers.Conv2D(2, kernel_size=(1, 1), padding='same',
            kernel_regularizer = L2(config.l2_decay))(x)
    p = layers.BatchNormalization()(p)
    p = layers.LeakyReLU()(p)
    p = layers.Flatten()(p)
    p = layers.Dense(Game.NUM_ACTIONS, name='policy',
            kernel_regularizer = L2(config.l2_decay))(p)

    # Value head
    v = layers.Conv2D(1, kernel_size=(1, 1), padding='same',
            kernel_regularizer = L2(config.l2_decay))(x)
    v = layers.BatchNormalization()(v)
    v = layers.LeakyReLU()(v)
    v = layers.Flatten()(v)
    v = layers.Dense(64,
            kernel_regularizer = L2(config.l2_decay))(v)
    v = layers.LeakyReLU()(v)
    v = layers.Dense(1, activation = 'tanh', name='value',
            kernel_regularizer = L2(config.l2_decay))(v)

    model = keras.Model(inputs = planes, outputs = [v, p])
    model.compile(
        optimizer = keras.optimizers.Adam(
            keras.optimizers.schedules.ExponentialDecay(
                config.lr_init,
                config.lr_steps,
                config.lr_multiplier)
            ),
        loss = [
            keras.losses.MeanSquaredError(name="value_loss"), # for value
            keras.losses.CategoricalCrossentropy(from_logits=True, name="policy_loss") # for policy
        ]
    )

    return model

def residual_block(y, nb_channels: int, reg: float=0):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), padding='same',
            kernel_regularizer = L2(reg))(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), padding='same',
            kernel_regularizer = L2(reg))(y)
    y = layers.BatchNormalization()(y)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

def get_network(checkpoint: bool, config: Config) -> keras.Model:
    """
    Either load the latest checkpoint or make a new network.

    If 0 < epoch, then try to load a specific checkpoint file.
    If epoch == -1, then load the most recent checkpoint file (ie, the furthest epoch).
    If epoch is None, create a new model.
    """
    # First we have to load the network architecture
    model = make_network(config)
    # If we're going to load saved weights
    if checkpoint:
        latest = tf.train.latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            model.load_weights(latest)
    return model

