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


def hash(board):
    return ','.join([str(x) for x in list(board.reshape(BOARD_SIZE))])


class State:
    def __init__(self):
        self.board = np.array([EMPTY] * BOARD_SIZE).reshape((BOARD_NROWS, BOARD_NCOLS))
        self.state = hash(self.board)
        self.winner = None
        self.is_end = False
    
    def judge(self):
        """Judge winner and is_end based on the current state."""
        # Iterate each row to judge winner.
        for r in range(BOARD_NROWS):
            row = self.board[r, :]
            symbol = row[0]
            if symbol != EMPTY and np.sum(row) == symbol * NMARKS:
                self.winner = symbol
                self.is_end = True
                return self
        
        # Iterate each col to judge winner.
        for c in range(BOARD_NCOLS):
            col = self.board[:, c]
            symbol = col[0]
            if symbol != EMPTY and np.sum(col) == symbol * NMARKS:
                self.winner = symbol
                self.is_end = True
                return self
        
        # Check 2 diagonals to judge winner.
        symbol = self.board[1][1]
        if symbol != EMPTY: 
            diag1, diag2 = [], []
            for i in range(BOARD_NROWS):
                diag1.append(self.board[i][i])
                diag2.append(self.board[i][BOARD_NROWS - i - 1])

            diag1, diag2 = np.array(diag1), np.array(diag1)
            if np.sum(diag1) == symbol * NMARKS or np.sum(diag2) == symbol:
                self.winner = symbol
                self.is_end = True
                return self

        # Judge tie with no winner.
        if np.sum(np.abs(self.board)) == BOARD_SIZE:
            self.is_end = True
            return self

    def next_state(self, r, c, symbol):
        """Create the next state at position (r, c)."""
        next_state = State()
        next_state.board = self.board.copy()
        next_state.board[r][c] = symbol
        next_state.state = hash(next_state.board)
        next_state.judge()
        return next_state


def _dfs_states_recur(cur_symbol, cur_state, states_d):
    """DFS for next state by recursion."""
    for r in range(BOARD_NROWS):
        for c in range(BOARD_NCOLS):
            if cur_state.board[r][c] == ' ':
                next_state = cur_state.next_state(r, c, cur_symbol)
                if next_state.state not in states_d:
                    states_d[next_state.state] = next_state

                    # If game is not ended, continue DFS.
                    if not next_state.is_end:
                        if cur_symbol == 'X':
                            _dfs_states_recur('O', next_state, states_d)
                        else:
                            _dfs_states_recur('X', next_state, states_d)


def get_all_states():
    """Get all states from the init state."""
    # The player who plays first always uses 'X'.
    cur_symbol = 'X'
    cur_state = State()
    states_d = dict()
    states_d[cur_state.state] = cur_state
    _dfs_states_recur(cur_symbol, cur_state, states_d)
    return states_d


class Agent:
    def __init__(self, symbol, step_size=0.01, epsilon=0.1):
        self.symbol = symbol
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.state_parent_d = dict()
        self.state_isgreedy_d = dict()

    def init_state_values(self):
        """Init state-value table."""
        self.state_value_table = None
        pass
    
    def load_state_values(self):
        """Load learned state-value table."""
        self.state_value_table = None
        pass

    def reset_episode(self):
        """Rreset moves in a played episode."""
        pass
    
    def play(self, state):
        """Play a move from possible states given a state."""
        pass
    
    def backup_value(self, state, reward):
        """Back up value by a temporal-difference learning after an opisode."""
        pass


# TODO: Continue implementing tic-tac-toe.
