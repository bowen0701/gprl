from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Environment::
    """Environment class for k-armed bandit."""

    def __init__(self, k):
        # Simulate k means from standard normal N(0, 1).
        self.k = k
        self.means = np.random.normal(0, 1, self.k)

    def get_actions(self):
        """Get possible actions."""
        pass


class MultiArmedBanditAgent:
    """Agent class for multi-armed bandit."""

    def __init__(self, k, optim_initial_values=None):
        # Init k action-values Q(A) and counts N(A) for action A.
        self.k = k
        self.optim_initial_values = optim_initial_values
        pass

    def init_action_values(self):
        self.Q = [0] * self.k
        self.N = [0] * self.k

    def _exploit_and_explore(self, env, actions, UCB=False):
        """Exploit and explore by the epsilon-greedy strategy:
          - Take exploratory moves in the p% of times. 
          - Take greedy moves in the (100-p)% of times.
        where p% is epsilon. 
        If epsilon is zero, then use the greedy strategy.
        """
        pass

    def select_action(self):
        pass

    def estimate_action_values(self):
        pass


def k_armed_testbed():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
