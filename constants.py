import numpy as np

class Config:
    def __init__(self):
        ### Self-Play
        self.num_sampling_moves = 20
        self.max_moves = 400
        self.num_simulations = 100

        # Root exploration noise
        # Figure out average number of legal moves in chinese checkers
        # then alpha = 10/n_moves
        self.root_alpha = 0.3
        self.root_noise_scale = 0.25

        # UCB
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(1e4)
        self.checkpoint_interval = int(1e2)
        # What does this do?
        self.window_size = int(1e6)
        self.batch_size = 64

        self.regularization = 1e-4
        self.momentum = 0.9
        self.lr_base = 1e-1

        ### RNG
        self.seed = 3257840388504953787
        self.rng = np.random.default_rng(self.seed)
