from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np



class Environment::
	"""Environment class for k-armed bandit."""

	def __init__(self, K):
		# Simulate K means from standard normal N(0, 1).
		self.means = np.random.normal(0, 1, K)


class MultiArmedBanditAgent:
	"""Agent class for k-armed bandit."""

	def __init__(self, K):
		# Init K action-values Q(A) and counts N(A) for action A.
		self.Q = [0] * K
		self.N = [0] * K


def k_armed_testbed():
	pass


def main():
	pass


if __name__ == '__main__':
	main()
