from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


K = 10


class Environment:
    """Environment class for k-armed bandit."""

    def __init__(self, K):
        # Simulate k means from standard normal N(0, 1).
        self.K = K
        self.reward_means = np.random.randn(self.K)
        self.optim_action = np.argmax(self.reward_means)

    def get_actions(self):
        """Get possible (fixed) actions."""
        return list(range(self.K))

    def step(self, action):
        """Step by action to get reward."""
        return np.random.randn() + self.reward_means[action]


class MultiArmedBanditAgent:
    """Agent class for stationary multi-armed bandit."""

    def __init__(self, K, epsilon=0.1, optim_init_values=None):
        self.K = K
        self.epsilon = epsilon

        if optim_init_values:
            self.optim_init_values = optim_init_values
        else:
            self.optim_init_values = 0

        self.actions = []
        self.rewards = []

    def init_action_values(self):
        """Initialize action values."""
        self.Q = [0 + self.optim_init_values] * self.K
        self.N = [0] * self.K

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

    def select_action(self, env):
        """Select an action from possible actions."""
        # Get next actions from environment.
        actions = env.get_actions()

        # Exloit and explore by the epsilon-greedy strategy.
        action = self._exploit_and_explore(actions)
        self.actions.append(action)
        return action

    def backup_action_value(self, reward):
        """Backup action value for stationary problem."""
        self.rewards.append(reward)

        action = self.actions[-1]
        self.N[action] += 1
        self.Q[action] += 1 / self.N[action] * (reward - self.Q[action])


def k_armed_testbed(K, bandits, runs=2000, steps=1000, print_per_runs=100):
    n_bandits = len(bandits)
    rewards = np.zeros((n_bandits, runs, steps))
    optimal_actions = np.zeros((n_bandits, runs, steps))

    for i, bandit in enumerate(bandits):
        for r in range(runs):
            env = Environment(K)
            bandit.init_action_values()

            for s in range(steps):
                # Environment and agent interact with each other.
                action = bandit.select_action(env)
                reward = env.step(action)
                bandit.backup_action_value(reward)

                # Store reward and optimal action indicator.
                rewards[i, r, s] = reward
                if action == env.optim_action:
                    optimal_actions[i, r, s] = 1

            if (r + 1) % print_per_runs == 0:
                print('Run {} for agent {} is completed.'.format(r, i))

    # Average along runs.
    avg_rewards = rewards.mean(axis=1)
    avg_optim_actions = optimal_actions.mean(axis=1)
    return avg_rewards, avg_optim_actions


def figure2_1():
    plt.violinplot(dataset=np.random.randn(K) + np.random.randn(200, K))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.hlines(y=0, xmin=0.5, xmax=10.5, linestyles='dashed')
    plt.savefig("../images/figure2.1.png")
    plt.close()


def figure2_2():
    epsilons = [0, 0.01, 0.1]
    bandits = [MultiArmedBanditAgent(K, epsilon) for epsilon in epsilons]
    avg_rewards, avg_optim_actions = k_armed_testbed(
        K, bandits, runs=2000, steps=1000)

    # TODO: Plot figure.
    pass


def main():
    figure2_1(K)


if __name__ == '__main__':
    main()
