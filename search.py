from typing import List, Tuple, Dict
import numpy as np
import math

from game import Game
from constants import Config
from scipy.special import softmax

class Node(object):
    """
    Dataclass to store a node in a search tree.
    """
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    # This is the value for the player whose turn it is at this state.
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
    root_legals = game.legal_actions()
    _, root_policy = evaluate(network,
            np.expand_dims(game.make_image(-1), 0),
            np.expand_dims(root_legals, 0))
    # Add noise to the priors first!
    root_policy = add_exploration_noise(root_policy[0], config) * root_legals
    create_children(root, root_policy)

    # Do a number of playouts
    for _ in range(config.num_simulations):
            node = root
            # Create a copy of the game to try moves in 
            scratch_game = game.clone()
            search_path = [node]
            # Randomly pick moves until we reach the end of the known search tree
            while node.expanded() and not scratch_game.terminal():
                action, node = select_child(node)
                scratch_game.apply(action)
                search_path.append(node)
            node.to_play = scratch_game.to_play()


            if scratch_game.terminal():
                value = scratch_game.terminal_value(node.to_play)
            else:
                value, policy = evaluate(network,
                        np.stack([scratch_game.make_image(-1)]),
                        np.stack([scratch_game.legal_actions()]))
                value = float(value[0])
                policy = policy[0]

                # Store the NN's estimation in the search tree
                create_children(node, policy)
            # Backpropagate the value
            backpropagate(search_path, value)
    # Return the chosen action as well as the search tree.
    return select_action(game, root, config), root

def evaluate(evaluator, images, actions):
    """
    Use a neural network evaluator to compute a suggested policy at a node.
    """
    # Let's make sure that passing a negative int to game.make_image works
    value, policy_logits = evaluator(images, training=False)
    # Zero out the illegal actions and then softmax
    policy = softmask(policy_logits, actions, 1)
    # Return the values and policies
    return value[:,0].numpy(), policy

def create_children(node, priors):
    """
    Given a list of prior probabilities, create children for a node
    """

    for a in np.flatnonzero(priors):
        # p is only set to 0 if the move is actually illegal
        node.children[int(a)] = Node(priors[a])

def backpropagate(path, value):
    to_play = path[-1].to_play
    for parent in path:
        parent.value_sum += value if parent.to_play == to_play else -value
        parent.visit_count += 1

def add_exploration_noise(priors: np.array, config: Config) -> np.array:
    """
    Add noise to an array of priors
    """
    noise = config.rng.gamma(config.root_alpha, 1.0, priors.shape)
    frac = config.root_noise_scale
    return priors * (1 - frac) + noise * frac

def select_child(node: Node) -> Tuple[int, Node]:
    """
    Compute UCB scores for a node's children and return the best one.
    """
    # First, we package the relevant things into a numpy array.
    # This will let us quickly do operations over it

    # This relies on the fact that action == int
    children = np.array([[action, child.visit_count, child.prior, -child.value()]
            for (action, child) in node.children.items()])
    
    # Next, we compute the prior -- filling in the denominator of the "underexplored" fraction, and multiplying by the prior value
    prior_score = math.sqrt(node.visit_count) * children[:,2] / (children[:,1] + 1)
    # Finally we get the ucb by adding this to the Q value
    ucb_score = prior_score + children[:,3]
    # And then we find the index which gives us the best score, and get the action name (ie, an int) 
    best_action = int(children[np.argmax(ucb_score),0])
    return (best_action, node.children[best_action])


def select_action(game: Game, root: Node, config: Config) -> int:
    """
    Choose an action to take based on search statistics.
    Chooses proportionally in the beginning of the game,
    and greedily in the rest of the game.
    """
    visit_counts = [(child.visit_count, i) for (i, child) in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        # Softmax sampling
        probs = softmax(np.array([c for (c, _) in visit_counts]), 0)
        action = config.rng.choice([i for (_, i) in visit_counts], p=probs)
    else:
        _, action = max(visit_counts)
    return action

def softmask(x, mask, axis=None):
    x = softmax(x * mask, axis=axis) * mask
    return x / np.sum(x, axis=axis, keepdims=True)

