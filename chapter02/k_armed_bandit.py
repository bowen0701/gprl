from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


ACTION_VAL_MEAN = 0
ACTION_VAL_VAR = 1
REWARD_VAR = 1


class Environment::
    """Environment class for k-armed bandit."""

    def __init__(self, k):
        # Simulate k means from standard normal N(0, 1).
        self.k = k
        self.means = np.random.normal(ACTION_VAL_MEAN, ACTION_VAL_VAR, self.k)

    def get_actions(self):
        """Get possible (fixed) actions."""
        return list(range(self.k))

    def get_reward(self, action):
        """Get reward given action."""
        return np.random.normal(self.means[action], REWARD_VAR)


class MultiArmedBanditAgent:
    """Agent class for multi-armed bandit."""

    def __init__(self, 
                 k, 
                 step_size=0.01, 
                 epsilon=0.01,
                 optim_init_values=None):
        # Init k action-values Q(A) and counts N(A) for action A.
        self.k = k
        self.step_size = step_size
        self.epsilon = epsilon
        self.optim_init_values = optim_init_values

    def init_action_values(self):
        """Initialize action values."""
        self.Q = [0] * self.k
        self.N = [0] * self.k

    def _exploit_and_explore(self, actions):
        """Exploit and explore by the epsilon-greedy strategy:
          - Take exploratory moves in the p% of times. 
          - Take greedy moves in the (100-p)% of times.
        where p% is epsilon. 
        If epsilon is zero, then use the greedy strategy.
        """
        vals_actions = []
        for a in actions:
            v = self.Q[a]
            vals_actions.append((v, a))

        p = np.random.random()
        if p > self.epsilon:
            # Exploit by selecting the action with the greatest value and
            # breaking ties randomly.
            np.random.shuffle(vals_actions)
            vals_actions.sort(key=lambda x: x[0], reverse=True)
            action = vals_actions[0][1]
        else:
            # Explore by selecting action randomly.
            n = len(vals_actions)
            action = vals_actions[np.random.randint(n)][1]

        return action

    def select_action(self):
        """Select an action from possible actions."""
        # Get next actions from environment.
        actions = env.get_actions()

        # Exloit and explore by the epsilon-greedy strategy.
        action = self._exploit_and_explore(actions)
        return action

    def backup_action_value(self):
        """Backup action value."""
        pass


def k_armed_testbed():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
