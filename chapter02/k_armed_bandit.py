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
    """Agent class for stationary multi-armed bandit."""

    def __init__(self, 
                 k, 
                 epsilon=0.01,
                 optim_init_values=None):
        self.k = k
        self.epsilon = epsilon

        if optim_init_values:
            self.optim_init_values = optim_init_values
        else:
            self.optim_init_values = 0

        self.actions = []
        self.rewards = []

    def init_action_values(self):
        """Initialize action values."""
        self.Q = [0 + self.optim_init_values] * self.k
        self.N = [0] * self.k

    def _explore(self, actions):
        """Random exploration."""
        np.random.shuffle(actions)
        n = len(actions)
        action = actions[np.random.randint(n)]
        return action

    def _exploit_and_explore(self, actions):
        """Exploit and explore by the epsilon-greedy strategy:
          - Take exploratory moves in the p% of times. 
          - Take greedy moves in the (100-p)% of times.
        where p% is epsilon. 
        If epsilon is zero, then use the greedy strategy.
        """
        p = np.random.random()
        if p > self.epsilon:
            # Exploit by selecting the action with the greatest value and
            # breaking ties randomly.
            vals_actions = []
            for a in actions:
                v = self.Q[a]
                vals_actions.append((v, a))
            np.random.shuffle(vals_actions)
            vals_actions.sort(key=lambda x: x[0], reverse=True)
            action = vals_actions[0][1]
        else:
            # Explore by selecting action randomly.
            action = self._explore(actions)

        return action

    def select_action(self):
        """Select an action from possible actions."""
        # Get next actions from environment.
        actions = env.get_actions()

        # Exloit and explore by the epsilon-greedy strategy.
        action = self._exploit_and_explore(actions)
        self.actions.append(action)
        return action

    def backup_action_value(self, reward):
        """Backup action value."""
        self.rewards.append(reward)

        action = self.actions[-1]
        self.N[action] += 1
        self.Q[action] += 1 / self.N[action] * (reward - self.Q[action])


def k_armed_testbed():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
