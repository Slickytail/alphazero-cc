from typing import List, Tuple, Dict
import numpy as np
import math

from game_az_wrapper import Game, Action
from constants import Config

class Node(object):
    """
    Dataclass to store a node in a search tree.
    """
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[Action, Node] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

def mcts(game: Game, network, config: Config):
    """
    Monte-Carlo tree search.
    """
    # Create the root node
    root = Node(0)
    # Tell it who's playing
    root.to_play = game.to_play()
    # Fill in the probabilities of its children according to the current NN
    _, root_policy = evaluate(network,
            np.expand_dims(game.make_image(-1), 0),
            np.expand_dims(game.legal_actions(), 0))
    # Add noise to the priors first!
    root_policy = add_exploration_noise(root_policy[0], config)
    create_children(root, root_policy)

    # Do a number of playouts
    for _ in range(config.num_simulations // config.search_batch_size):
        # Initialize an empty stack of searches
        actions = np.empty((config.search_batch_size, Game.NUM_ACTIONS), dtype=np.float32)
        images = np.empty((config.search_batch_size, *Game.INPUT_SHAPE), dtype=np.float32)
        search_paths = []
        
        # fill up a batch of searches.
        # this could be done in parallel, but because of the overhead of creating threads/processes
        # it might actually be slower to do it naively -- ie, without keeping the threads long-term
        for i in range(config.search_batch_size):
            node = root
            # Create a copy of the game to try moves in 
            scratch_game = game.clone()
            search_path = [node]
            # Randomly pick moves until we reach the end of the known search tree
            while node.expanded():
                action, node = select_child(node, config)
                scratch_game.apply(action)
                search_path.append(node)
            # TODO: Add edge case: what if the game is won?
            node.to_play = scratch_game.to_play()

            images[i,:,:,:] = scratch_game.make_image(-1)
            actions[i,:] = scratch_game.legal_actions()
            search_paths.append(search_path)
       
        # Use the neural network to get suggested priors and value
        values, policies = evaluate(network, images, actions)
        # Store the NN's estimation in the search tree
        for i in range(config.search_batch_size):
            # Add children to the node
            path = search_paths[i]
            node = path[-1]
            create_children(node, policies[i])
            # Backpropagate the value
            to_play = node.to_play
            value = values[i]
            for node in path:
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
    # Return the chosen action as well as the search tree.
    return select_action(game, root, config), root


def evaluate(evaluator, images, actions):
    """
    Use a neural network evaluator to compute a suggested policy at a node.
    """
    # Let's make sure that passing a negative int to game.make_image works
    value, policy_logits = evaluator(images, training=False)
    # Zero out the illegal actions
    policy = np.exp(policy_logits * actions)
    # Renormalize the probabilities
    # policies has shape (batches, actions) and we want to sum over actions
    policy_sum = np.sum(policy, axis=1, keepdims=True)
    policy = policy / policy_sum
    # Return the values and policies
    return value[:,0], policy

def create_children(node, priors):
    """
    Given a list of prior probabilities, create children for a node
    """
    for (a, p) in enumerate(priors):
        # p is only set to 0 if the move is actually illegal
        if p != 0.0:
            node.children[a] = Node(p)

def add_exploration_noise(priors: np.array, config: Config) -> np.array:
    """
    Add noise to an array of priors
    """
    noise = config.rng.gamma(config.root_alpha, 1, priors.shape)
    frac = config.root_noise_scale
    return priors * (1 - frac) + noise * frac

def select_child(node: Node, config: Config) -> Tuple[Action, Node]:
    """
    Compute UCB scores for a node's children and return the best one.
    """
    # First, we package the relevant things into a numpy array.
    # This will let us quickly do operations over it

    # This relies on the fact that action == int
    children = np.array([[action, child.visit_count, child.prior, child.value()]
            for (action, child) in node.children.items()])
    
    # This computes the UCB constant as well as the numerator of the "underexplored" fraction
    pb_c = (math.log((node.visit_count + config.pb_c_base + 1) /
            config.pb_c_base) + config.pb_c_init) * math.sqrt(node.visit_count)
    # Next, we compute the prior -- filling in the denominator of the "underexplored" fraction, and multiplying by the prior value
    prior_score = pb_c * children[:,2] / (children[:,1] + 1)
    # Finally we get the ucb by adding this to the Q value
    ucb_score = prior_score + children[:,3]
    # And then we find the index which gives us the best score, and get the action name (ie, an int) 
    best_action = Action(children[np.argmax(ucb_score),0])
    return (best_action, node.children[best_action])


def select_action(game: Game, root: Node, config: Config) -> Action:
    """
    Choose an action to take based on search statistics.
    Chooses proportionally in the beginning of the game,
    and greedily in the rest of the game.
    """
    visit_counts = [(child.visit_count, i) for (i, child) in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        # Softmax sampling
        probs = np.exp([c for (c, _) in visit_counts])
        action = config.rng.choice([i for (_, i) in visit_counts], p=probs/np.sum(probs))
    else:
        _, action = max(visit_counts)
    return action

# Note to self: exploiting symmetry (ie, augmenting training data by flipping) will make training much faster when measured in real-time
# Need to find out: does AlphaZero flip the board to be from the current player's position?

