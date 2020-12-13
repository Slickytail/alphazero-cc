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
    Single threaded.
    """
    root = Node(0)
    evaluate(root, game, network)
    add_exploration_noise(root, config)

    # Do a number of playouts
    for _ in range(config.num_simulations):
        node = root
        # Create a copy of the game to try moves in 
        scratch_game = game.clone()
        search_path = [node]
        # Randomly pick moves until we reach the end of the known search tree
        while node.expanded():
            action, node = select_child(node, config)
            scratch_game.apply(action)
            search_path.append(node)
       
        # To improve performance (GPU batching)
        # In theory you could also 
        # Run the loop only up to this point B times (probably 2^n)
        # And on each run just save the search path and the game's image, player, and legal actions
        # save all of these into a list.
        # Then stack up the B images and the B arrays of legal actions.
        # Send the images to the neural network.
        # Then you can multiply the stack of policy_logits by the legal_actions arrays very fast
        # do a softmax (be sure to get the axis summation right)
        # and then finally unstack them, create children, etc.

        # Use the neural network to get suggested priors and value
        value = evaluate(node, scratch_game, network)
        # Store the NN's estimation in the search tree
        backpropagate(search_path, value, scratch_game.to_play())
    # Return the chosen action as well as the search tree.
    return select_action(game, root, config), root


def evaluate(node: Node, game: Game, evaluator) -> float:
    """
    Use a neural network evaluator to compute a suggested policy at a node.
    """
    # Let's make sure that passing a negative int to game.make_image works
    value, policy_logits = evaluator.predict(game.make_image(-1))

    # Record the policy prediction in the children
    node.to_play = game.to_play()
    # Zero out the illegal actions
    policy = np.exp(policy_logits * game.legal_actions())
    # Renormalize the probabilities
    policy_sum = np.sum(policy)
    policy = policy / policy_sum
    # Add children to the node
    for (i, p) in enumerate(policy):
        if p != 0.0:
            node.children[i] = Node(p)
    # Return the position's value
    return value


def add_exploration_noise(node: Node, config: Config):
    """
    Add noise to the priors of a node's children.
    """
    actions = node.children.keys()
    noise = config.rng.gamma(config.root_alpha, 1, len(actions))
    frac = config.root_noise_scale
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

def select_child(node: Node, config: Config) -> Tuple[Action, Node]:
    """
    Compute UCB scores for a node's children and return the best one.
    """
    # First, we package the relevant things into a numpy array.
    # This will let us quickly do operations over it

    # This relies on the fact that action == int
    children = np.array([action, child.visit_count, child.prior, child.value()]
            for (action, child) in node.children.items())
    
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

def backpropagate(search_path: List[Node], value: float, to_play: int):
    """
    Given a value at the end of a search path, store that value in the path.
    """
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1

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

