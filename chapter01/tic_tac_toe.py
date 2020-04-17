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
    return np.fromstring(state, dtype=int, sep=',')


class State:
    def __init__(self):
        self.board = np.array([EMPTY] * BOARD_SIZE).reshape((BOARD_NROWS, BOARD_NCOLS))
        self.state = hash(self.board)
        self.is_end = False
        self.winner = None
    
    def judge(self):
        """Judge winner and is_end based on the current state."""
        # Iterate each row to judge winner.
        for r in range(BOARD_NROWS):
            row = self.board[r, :]
            symbol = row[0]
            if symbol != EMPTY and np.sum(row) == symbol * NMARKS:
                self.is_end = True
                self.winner = symbol
                return self
        
        # Iterate each col to judge winner.
        for c in range(BOARD_NCOLS):
            col = self.board[:, c]
            symbol = col[0]
            if symbol != EMPTY and np.sum(col) == symbol * NMARKS:
                self.is_end = True
                self.winner = symbol
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
                self.is_end = True
                self.winner = symbol
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

    def show_board(self):
        board = self.board.tolist()
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if board[r][c] == CROSS:
                    board[r][c] = 'X'
                elif board[r][c] == CIRCLE:
                    board[r][c] = 'O'
                else:
                    board[r][c] = ' '

        print('Board: is_end={}, winner={}'.format(self.is_end, self.winner))
        [print(board[r]) for r in range(BOARD_NROWS)]


def _dfs_states(cur_symbol, cur_state, states_d):
    """DFS for next state by recursion."""
    for r in range(BOARD_NROWS):
        for c in range(BOARD_NCOLS):
            if cur_state.board[r][c] == EMPTY:
                next_state = cur_state.next_state(r, c, cur_symbol)
                if next_state.state not in states_d:
                    states_d[next_state.state] = next_state

                    # If game is not ended, continue DFS.
                    if not next_state.is_end:
                        _dfs_states(-cur_symbol, next_state, states_d)


def get_all_states():
    """Get all states from the init state."""
    # The player who plays first always uses 'X'.
    cur_symbol = CROSS
    cur_state = State()
    states_d = dict()
    states_d[cur_state.state] = cur_state
    _dfs_states(cur_symbol, cur_state, states_d)
    return states_d


class Agent:
    def __init__(self, symbol, step_size=0.01, epsilon=0.1):
        self.symbol = symbol
        self.step_size = step_size
        self.epsilon = epsilon

        # Create an afterstate-value table.
        self.state_value_table = dict()
        self.init_state_value()

        # Memoize all states played by two players.
        self.states = []

        # Momoize action states and their parent states & is_greedy bools.
        self.actstate_parent_isgreedy_d = dict()

    def init_state_values(self):
        """Init state-value table."""
        for state in ALL_STATES_D:
            if state.winner == self.symbol:
                self.state_value_table[state] = 1.0
            elif state.winner == -self.symbol:
                self.state_value_table[state] = 0.0
            else:
                self.state_value_table[state] = 0.5
    
    def load_state_values(self):
        """Load learned state-value table."""
        self.state_value_table = None
        pass

    def set_state(self, state):
        """State the latest state."""
        self.states.append(state)

    def _get_possible_moves(self):
        """Get possible moves from possible states given current state."""
        state = self.states[-1]
        next_states = []

        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if state.board[r][c] == EMPTY:
                    next_state = state.next_state(r, c, self.symbol)
                    next_states.append(next_state)
        return next_states

    def _play_strategy(self, next_states):
        """Play by strategy. Here we use epsilon-greedy strategy.

        Epsilon-greedy strategy: 
        - Take exploratory moves in the p% of times. 
        - Take greedy moves in the (100-p)% of times.
        where p% is epsilon. 
        If epsilon is zero, then use the greedy strategy.
        """
        p = np.random.random()
        if p > self.epsilon:
            # Exploit.
            values_states = [(self.state_value_table[st], st)
                             for st in next_states]
            values_states.sort(reverse=True)
            next_state = values_states[0]
            is_greedy = True
        else:
            # Explore.
            np.random.shuffle(next_states)
            next_state = next_states[0]
            is_greedy = False
        return (next_state, is_greedy)
    
    def act(self, state):
        """Play a move from possible states given current state."""
        # Get possible moves given a state by replacing EMPTY cells.
        next_states = self._get_possible_moves()

        # Apply epsilon-greedy strategy.
        (next_state, is_greedy) = self._play_strategy(next_states)
        self.actstate_parent_isgreedy_d[next_state.state] = {
            'parent': self.states[-1],
            'is_greedy': is_greedy
        }
        return next_state

    # TODO: Continue implementing tic-tac-toe.

    def backup_value(self, state, reward):
        """Back up value by a temporal-difference learning after a move.
        
        Temporal-difference learning:
          V(S_t) <- V(S_t) + a * [V(S_{t+1}) - V(S_t)]
        where a is the step size, and V(S_t) is the state-value function
        at time step t.
        """
        pass

    def reset_episode(self):
        """Rreset moves in a played episode."""
        self.states = []
        self.actstate_parent_isgreedy_d = dict()
