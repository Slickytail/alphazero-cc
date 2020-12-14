import numpy as np
import datetime

class Config:
    def __init__(self):
        ### Self-Play
        self.search_batch_size = 8
        self.num_sampling_moves = 20
        self.max_moves = 400
        self.num_simulations = 128

        # Root exploration noise
        # Figure out average number of legal moves in chinese checkers
        # then alpha = 10/n_moves
        self.root_alpha = 0.3
        self.root_noise_scale = 0.25

        # UCB
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(7e4)
        self.games_per_step = 100
        self.batches_per_step = 10

        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e4)
        self.batch_size = 128

        # Model saving
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_fname = "chinese-checkers-training-{epoch:05d}.ckpt"

        # Logging
        self.log_dir = "logs"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Coefficient for L2 regularization
        self.l2_decay = 1e-4
        self.lr_init = 2e-1
        self.lr_multiplier = 0.5
        self.lr_steps = 3e3

        ### RNG
        self.seed = 3257840388504953787
        self.rng = np.random.default_rng(self.seed)
