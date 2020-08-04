from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np


NMARKS = 3
BOARD_NROWS = BOARD_NCOLS = 3
BOARD_SIZE = BOARD_NROWS * BOARD_NCOLS

CROSS = 1
CIRCLE = -1
EMPTY = 0

np.random.seed(seed=71)


def hash(board):
    return ','.join([str(x) for x in list(board.reshape(BOARD_SIZE))])


def unhash(state):
    return (np.fromstring(state, dtype=int, sep=',')
              .reshape((BOARD_NROWS, BOARD_NCOLS)))

class Environment(object):
    def __init__(self):
        self.steps_left = BOARD_SIZE
        self.board = (np.array([EMPTY] * BOARD_SIZE)
                        .reshape((BOARD_NROWS, BOARD_NCOLS)))
        self.state = hash(self.board)
        self.winner = None

    def _judge(self):
        """Judge winner based on the current board."""
        # Check rows.
        for r in range(BOARD_NROWS):
            row = self.board[r, :]
            symbol = row[0]
            if symbol != EMPTY and np.sum(row) == symbol * NMARKS:
                self.winner = symbol
                return self
        
        # Check columns.
        for c in range(BOARD_NCOLS):
            col = self.board[:, c]
            symbol = col[0]
            if symbol != EMPTY and np.sum(col) == symbol * NMARKS:
                self.winner = symbol
                return self
        
        # Check diagonals.
        mid = BOARD_NROWS // 2
        symbol = self.board[mid][mid]
        if symbol != EMPTY: 
            diag1, diag2 = [], []
            for i in range(BOARD_NROWS):
                diag1.append(self.board[i][i])
                diag2.append(self.board[i][BOARD_NROWS - i - 1])

            diag1, diag2 = np.array(diag1), np.array(diag1)
            if np.sum(diag1) == symbol * NMARKS or np.sum(diag2) == symbol:
                self.winner = symbol
                return self

    def is_done(self):
        return self.steps_left == 0

    def step(self, r, c, symbol):
        """Take next step as state with symbol."""
        self.board[r][c] = symbol
        self.state = hash(self.board)
        self._judge()
        self.steps_left -= 1
        return self.state, self.is_done(), self.winner

    def copy(self):
        env_copy = Environment()
        env_copy.steps_left = self.steps_left
        env_copy.board = self.board.copy()
        env_copy.state = self.state
        env_copy.winner = self.winner
        return env_copy

    def show_board(self):
        """Show board."""
        board = self.board.tolist()
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if board[r][c] == CROSS:
                    board[r][c] = 'X'
                elif board[r][c] == CIRCLE:
                    board[r][c] = 'O'
                else:
                    board[r][c] = ' '

        print('Board: is_done={}, winner={}'
              .format(self.is_done(), self.winner))
        for r in range(BOARD_NROWS):
            print(board[r])


def _dfs_states(cur_symbol, env, all_state_envs):
    """DFS for next state by recursion."""
    for r in range(BOARD_NROWS):
        for c in range(BOARD_NCOLS):
            if env.board[r][c] == EMPTY:
                env_copy = env.copy()
                env_copy.step(r, c, cur_symbol)
                if env_copy.state not in all_state_envs:
                    all_state_envs[env_copy.state] = env_copy

                    # If game is not ended, continue DFS.
                    if not env_copy.is_done():
                        _dfs_states(-cur_symbol, env_copy, all_state_envs)


def get_all_states():
    """Get all states from the init state."""
    # The player who plays first always uses 'X'.
    cur_symbol = CROSS

    # Apply DFS to collect all states.
    env = Environment()
    all_state_envs = dict()
    all_state_envs[env.state] = env
    _dfs_states(cur_symbol, env, all_state_envs)
    return all_state_envs


ALL_STATE_ENV = get_all_states()


class Agent(object):
    def __init__(self, player='X', step_size=0.01, epsilon=0.1):
        if player == 'X':
            self.symbol = CROSS
        else:
            self.symbol = CIRCLE
        self.step_size = step_size
        self.epsilon = epsilon

        # Create an afterstate-value table V: state->value.
        self.V = dict()
        self.init_state_value_table()

        # Memoize all states played by two players.

        # Momoize action state, its parent state & greedy bool.
        self.states = []
        self.state_parent_d = dict()
        self.state_isgreedy_d = dict()

    def init_state_value_table(self):
        """Init state-value table."""
        for s in ALL_STATE_ENV:
            env = ALL_STATE_ENV[s]
            if env.winner == self.symbol:
                self.V[s] = 1.0
            elif env.winner == -self.symbol:
                self.V[s] = 0.0
            else:
                self.V[s] = 0.5

    def _get_actions(self, env, symbol):
        """Get possible action positions given current board."""
        action_positions = dict()
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if env.board[r][c] == EMPTY:
                    action_positions[(r, c)] = env_copy
        return action_positions

    def _play_strategy(self, action_positions, env):
        """Play with strategy. Here we use epsilon-greedy strategy.

        Epsilon-greedy strategy: 
          - Take exploratory moves in the p% of times. 
          - Take greedy moves in the (100-p)% of times.
        where p% is epsilon. 
        If epsilon is zero, then use the greedy strategy.
        """
        p = np.random.random()
        if p > self.epsilon:
            # Exploit.
            next_r, next_c, next_state = None, None, None
            value = -float('inf')
            for (r, c) in action_positions:
                env_copy = env.copy()
                env_copy.step(r, c, self.symbol)
                s = env_copy.state
                v = self.V[s]
                if v > value:
                    next_r, next_c, next_state = r, c, s
            is_greedy = True
        else:
            # Explore.
            (next_r, next_c) = np.random.choice(action_positions)
            env_copy = env.copy()
            env_copy.step(next_r, next_c, self.symbol)
            next_state = env_copy.state
            is_greedy = False
        return (next_r, next_c, next_state, is_greedy)
    
    def act(self, env):
        """Play a move from possible states given current state."""
        # Get possible actions from environment.
        action_positions = self._get_actions(env, self.symbol)

        # Apply epsilon-greedy strategy.
        (next_r, next_c, next_state, is_greedy) = self._play_strategy(
            action_positions)
        self.state_parent_d[next_state] = self.states[-1]
        self.state_isgreedy_d[next_state] = is_greedy
        self.states.append(next_state)
        return next_r, next_c, self.symbol

    def backup_value(self, state, reward):
        """Back up value by a temporal-difference learning after a greedy move.
        
        Temporal-difference learning:
          V(S_t) <- V(S_t) + a * [V(S_{t+1}) - V(S_t)]
        where a is the step size, and V(S_t) is the state-value function
        at time step t.
        """
        s = state.state
        s_before = self.state_parent[s].state
        is_greedy = self.state_isgreedy[s]
        if is_greedy:
            self.V[s_before] += self.step_size * (self.V[s] - self.V[s_before])

    def reset_episode(self):
        """Rreset moves in a played episode."""
        self.states = []
        self.state_parent = dict()
        self.state_isgreedy = dict()

    def save_state_values(self):
        """Save learned state-value table."""
        if self.symbol == CROSS:
            json.dump(self.V, open("state_value_x.json", 'w'))
        else:
            json.dump(self.V, open("state_value_o.json", 'w'))

    def load_state_values(self):
        """Load learned state-value table."""
        if self.symbol == CROSS:
            self.V = json.load(open("state_value_x.json"))
        else:
            self.V = json.load(open("state_value_o.json"))
