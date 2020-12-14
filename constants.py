import numpy as np
import datetime

class Config:
    def __init__(self):
        ### Self-Play
        self.search_batch_size = 8
        self.num_sampling_moves = 15
        self.max_moves = 600
        self.num_simulations = 32

        # Root exploration noise
        # Figure out average number of legal moves in chinese checkers
        # then alpha = 10/n_moves
        self.root_alpha = 0.2
        self.root_noise_scale = 0.15

        ### Training
        self.training_steps = int(1e3)
        self.games_per_step = 30
        self.batches_per_step = 16

        self.checkpoint_interval = 50
        self.window_size = 200
        self.batch_size = 64

        # Model saving
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_fname = "chinese-checkers-training.ckpt"

        # Logging
        self.log_dir = "logs"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Coefficient for L2 regularization
        self.l2_decay = 1e-4
        self.lr_init = 1e-1
        self.lr_multiplier = 0.5
        self.lr_steps = 3e3

        ### RNG
        self.seed = 3257840388504953787
        self.rng = np.random.default_rng(self.seed)
