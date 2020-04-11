from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np


NMARKS = 3
BOARD_NROWS = BOARD_NCOLS = 3
BOARD_SIZE = BOARD_NROWS * BOARD_NCOLS


def hash(board):
    return ''.join(list(board.reshape(BOARD_SIZE)))

def unhash(state):
    return np.array(list(state)).reshape((BOARD_NROWS, BOARD_NCOLS))


class State:
    def __init__(self):
        self.board = np.array([' '] * BOARD_SIZE).reshape((BOARD_NROWS, BOARD_NCOLS))
        self.state = hash(self.board)
        self.winner = None
        self.is_end = False
    
    def judge(self):
        """Judge winner and is_end based on the current state."""
        # Iterate each row to judge winner.
        for r in range(BOARD_NROWS):
            row = ''.join(self.board[r])
            symbol = row[0]
            if symbol != ' ' and row.count(symbol) == NMARKS:
                self.winner = symbol
                self.is_end = True
        
        # Iterate each col to judge winner.
        for c in range(BOARD_NCOLS):
            col = ''.join(self.board[:][c])
            symbol = col[0]
            if symbol != ' ' and col.count(symbol) == NMARKS:
                self.winner = symbol
                self.is_end = True
        
        # Check 2 diagonals to judge winner.
        symbol = self.board[1][1]
        if symbol != ' ': 
            diag1, diag2 = [], []
            for i in range(BOARD_NROWS):
                diag1.append(self.board[i][i])
                diag2.append(self.board[i][BOARD_NROWS - i - 1])

            diag1, diag2 = ''.join(diag1), ''.join(diag2)
            if diag1.count(symbol) == NMARKS or diag2.count(symbol) == NMARKS:
                self.winner = symbol
                self.is_end = True

        # Judge tie with no winner.
        if self.state.count('X') + self.state.count('O') == BOARD_SIZE:
            self.is_end = True
    
    def next_state(self, r, c, symbol):
        """Create the next state at position (r, c)."""
        next_state = State()
        next_state.board = self.board.copy()
        next_state.board[r][c] = symbol
        next_state.state = hash(next_state.board)
        next_state.judge()
        return next_state

    def _dfs_all_states_recur(self, symbol):
        states_d = dict()
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if self.board[r][c] == ' ':
                    next_state = self.next_state(r, c, symbol)
    
    def dfs_all_states(self, symbol):
        states_d = dict()
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if self.board[r][c] == ' ':
                    next_state = self.next_state(r, c, symbol)
                    if next_state.state not in states_d:
                        pass


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
