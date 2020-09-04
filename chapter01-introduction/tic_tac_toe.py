from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re

import numpy as np


NMARKS = 3
BOARD_NROWS = BOARD_NCOLS = 3
BOARD_SIZE = BOARD_NROWS * BOARD_NCOLS

CROSS = 1
CIRCLE = -1
EMPTY = 0


def hash(board):
    return ','.join([str(x) for x in list(board.reshape(BOARD_SIZE))])


class Environment:
    """Environment class for Tic-Tac Toe."""

    def __init__(self):
        self.steps_left = BOARD_SIZE
        self.board = (np.array([EMPTY] * BOARD_SIZE)
                        .reshape((BOARD_NROWS, BOARD_NCOLS)))
        self.state = hash(self.board)
        self.winner = EMPTY

    def get_positions(self):
        """Get possible action positions given current board."""
        positions = []
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if self.board[r][c] == EMPTY:
                    positions.append((r, c))
        return positions

    def _judge(self):
        """Judge winner based on the current board."""
        # Check rows.
        for r in range(BOARD_NROWS):
            row = self.board[r, :]
            symbol = row[0]
            if symbol != EMPTY and np.sum(row) == symbol * NMARKS:
                self.winner = symbol
                self.steps_left = 0
                return self

        # Check columns.
        for c in range(BOARD_NCOLS):
            col = self.board[:, c]
            symbol = col[0]
            if symbol != EMPTY and np.sum(col) == symbol * NMARKS:
                self.winner = symbol
                self.steps_left = 0
                return self

        # Check diagonals.
        mid = BOARD_NROWS // 2
        symbol = self.board[mid][mid]
        if symbol != EMPTY: 
            diag1, diag2 = [], []
            for i in range(BOARD_NROWS):
                diag1.append(self.board[i][i])
                diag2.append(self.board[i][BOARD_NROWS - i - 1])

            diag1, diag2 = np.array(diag1), np.array(diag2)
            if (np.sum(diag1) == symbol * NMARKS or 
                np.sum(diag2) == symbol * NMARKS):
                self.winner = symbol
                self.steps_left = 0
                return self

    def is_done(self):
        """Check the game is done."""
        return self.steps_left == 0

    def step(self, r, c, symbol):
        """Take a step with symbol."""
        env_next = self._copy()
        env_next.board[r][c] = symbol
        env_next.state = hash(env_next.board)
        env_next.steps_left -= 1
        env_next._judge()
        return env_next

    def _copy(self):
        """Copy to a new Environment instance."""
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

        print('Board: is_done={}, steps_left={}, winner={}'
              .format(self.is_done(), self.steps_left, self.winner))
        for r in range(BOARD_NROWS):
            print(board[r])


def _dfs_states(cur_symbol, env, all_state_env_d):
    """DFS for next state by recursion."""
    for r in range(BOARD_NROWS):
        for c in range(BOARD_NCOLS):
            if env.board[r][c] == EMPTY:
                env_next = env.step(r, c, cur_symbol)
                if env_next.state not in all_state_env_d:
                    all_state_env_d[env_next.state] = env_next

                    # If game is not ended, continue DFS.
                    if not env_next.is_done():
                        _dfs_states(-cur_symbol, env_next, all_state_env_d)


def get_all_states():
    """Get all states from the init state."""
    # The player who plays first always uses 'X'.
    cur_symbol = CROSS

    # Apply DFS to collect all states.
    env = Environment()
    all_state_env_d = dict()
    all_state_env_d[env.state] = env
    _dfs_states(cur_symbol, env, all_state_env_d)
    return all_state_env_d


ALL_STATE_ENV_D = get_all_states()


class Agent:
    """Agent class for Tic-Tac-Toe game."""

    def __init__(self, player='X', step_size=0.01, epsilon=0.01):
        self.player = player
        if self.player == 'X':
            self.symbol = CROSS
        elif self.player == 'O':
            self.symbol = CIRCLE
        else:
            raise InputError("Input player should be 'X' or 'O'")

        self.step_size = step_size
        self.epsilon = epsilon

        # Create a state-value table V:state->value.
        self.V = dict()

        # Memoize action state, its parent state & is_greedy bool:
        # state_parent_d:state->parent state & state_isgreedy_d:state->is_greedy bool.
        self.states = []
        self.state_parent_d = dict()
        self.state_isgreedy_d = dict()

    def init_state_value_table(self):
        """Init state-value table."""
        all_state_env_d = ALL_STATE_ENV_D

        for s, env in all_state_env_d.items():
            if env.winner == self.symbol:
                # If agent is winner, it gets reward 1.
                self.V[s] = 1.0
            elif env.winner == -self.symbol or env.steps_left == 0:
                # If agent is loser or tied, it gets reward 0.
                self.V[s] = 0.0
            else:
                # For other cases, agent get reward 0.5.
                self.V[s] = 0.5

    def reset_episode(self, env):
        """Init episode."""
        self.states.append(env.state)
        self.state_parent_d[env.state] = None
        self.state_isgreedy_d[env.state] = False

    def _exploit_and_explore(self, env, positions):
        """Exploit and explore by the epsilon-greedy strategy:
          - Take exploratory moves in the p% of times. 
          - Take greedy moves in the (100-p)% of times.
        where p% is epsilon. 
        If epsilon is zero, then use the greedy strategy.
        """
        vals_positions = []
        for (r, c) in positions:
            env_next = env.step(r, c, self.symbol)
            s = env_next.state
            v = self.V[s]
            vals_positions.append((v, (r, c)))

        # Break state-value ties randomly.
        np.random.shuffle(vals_positions)
        vals_positions.sort(key=lambda x: x[0], reverse=True)

        p = np.random.random()
        if p > self.epsilon:
            # Exploit by selecting the move with the greatest value.
            (r, c) = vals_positions[0][1]
            is_greedy = True
        else:
            # Explore by selecting randomly from among the other moves instead.
            if len(vals_positions) > 1:
                vals_positions_explore = vals_positions[1:]
                n = len(vals_positions_explore)
                (r, c) = vals_positions_explore[np.random.randint(n)][1]
            else:
                (r, c) = vals_positions[0][1]
            is_greedy = False

        env_next = env.step(r, c, self.symbol)
        state_next = env_next.state
        return (r, c, state_next, is_greedy)

    def add_state(self, state_next, is_greedy):
        state = self.states[-1]
        self.state_parent_d[state_next] = state
        self.state_isgreedy_d[state_next] = is_greedy
        self.states.append(state_next)
        return self

    def select_position(self, env):
        """Select a action position by the epsilon-greedy strategy.

        Agent gets candidate positions given current state.
        Then it selects a position by strategy.
        """
        # Get next action positions from environment.
        positions = env.get_positions()

        # Exloit and explore by the epsilon-greedy strategy.
        (r, c, state_next, is_greedy) = self._exploit_and_explore(
            env, positions)

        # Add state.
        self.add_state(state_next, is_greedy)
        return r, c, self.symbol

    def backup_state_value(self):
        """Back up value by a temporal-difference learning after a greedy move.

        Temporal-difference learning:
          V(S_t) <- V(S_t) + a * [V(S_{t+1}) - V(S_t)]
        where a is the step size, and V(S_t) is the state-value function
        at time step t.
        """
        s = self.states[-1]
        s_par = self.state_parent_d[s]
        is_greedy = self.state_isgreedy_d[s]
        if is_greedy:
            self.V[s_par] += self.step_size * (self.V[s] - self.V[s_par])

    def save_state_value_table(self):
        """Save learned state-value table."""
        if self.symbol == CROSS:
            json.dump(self.V, open("state_value_x.json", 'w'))
        else:
            json.dump(self.V, open("state_value_o.json", 'w'))

    def load_state_value_table(self):
        """Load learned state-value table."""
        if self.symbol == CROSS:
            self.V = json.load(open("state_value_x.json"))
        else:
            self.V = json.load(open("state_value_o.json"))


def self_train(epochs=100000, step_size=0.1, epsilon=0.1, print_per_epochs=100):
    """Self train an agent by playing games against itself."""
    agent1 = Agent(player='X', step_size=step_size, epsilon=epsilon)
    agent2 = Agent(player='O', step_size=step_size, epsilon=epsilon)
    agent1.init_state_value_table()
    agent2.init_state_value_table()

    n_agent1_wins = 0
    n_agent2_wins = 0
    n_ties = 0

    for i in range(1, epochs + 1):
        # Reset both agents after epoch was done.
        env = Environment()
        agent1.reset_episode(env)
        agent2.reset_episode(env)

        while not env.is_done():
            # Agent 1 plays one step.
            r1, c1, symbol1 = agent1.select_position(env)
            env = env.step(r1, c1, symbol1)
            agent1.backup_state_value()

            if env.is_done():
                break

            # Agent 2 plays the next step.
            r2, c2, symbol2 = agent2.select_position(env)
            env = env.step(r2, c2, symbol2)
            agent2.backup_state_value()

        # Set final state with is_greedy=True to backup value.
        is_greedy = True
        if env.winner == CROSS:
            agent2.add_state(env.state, is_greedy)
            agent2.backup_state_value()
            n_agent1_wins += 1
        elif env.winner == CIRCLE:
            agent1.add_state(env.state, is_greedy)
            agent1.backup_state_value()
            n_agent2_wins += 1
        else:
            n_ties += 1

        # Print board.
        if i % print_per_epochs == 0:
            print('Epoch {}: Agent1 wins {}, Agent2 wins {}, ties {}'
                  .format(i,
                          round(n_agent1_wins / i, 2), 
                          round(n_agent2_wins / i, 2), 
                          round(n_ties / i, 2)))
            env.show_board()
            print('---')

    agent1.save_state_value_table()
    agent2.save_state_value_table()


class Human:
    """Human class for Tic-Tac-Toe game."""

    def __init__(self, player='X'):
        self.player = player
        if self.player == 'X':
            self.symbol = CROSS
        elif self.player == 'O':
            self.symbol = CIRCLE

    def select_position(self, env):
        """Select a position given current state."""
        # Get human input position.
        input_position_re = re.compile('^[0-2],[0-2]$')
        positions = set(env.get_positions())
        while True:
            input_position = input(
                'Please input position for {} in the format: "row,col" with '
                .format(self.player) +
                'row/col: 0~{}:\n'.format(BOARD_NROWS - 1))

            if not input_position_re.match(input_position):
                print('Input position style is incorrect!\n')
                continue

            input_position = tuple([int(x) for x in input_position.split(',')])
            if input_position in positions:
                break
            else:
                print('Input position was occupied, please input "row,col" again!\n')

        (r, c) = input_position
        return r, c, self.symbol


def human_agent_compete():
    """Human compete with agent."""
    # Get human player.
    human_name = input('Please input your name:\n')
    while True:
        human_player = input('Please input your player: ' + 
                             '1st player (X), 2nd player (O):\n')
        if human_player in ['X', 'O']:
            break

    env = Environment()
    env.show_board()
    print('---')

    # Set up human & agent as player1 or player2.
    if human_player == 'X':
        # Player1: human, player2: agent.
        human, agent = Human(player='X'), Agent(player='O', epsilon=0)
        player1, player2 = human, agent
        player1_name, player2_name = human_name, 'Robot'
    else:
        # Player1: agent, player2: human.
        agent, human = Agent(player='X', epsilon=0), Human(player='O')
        player1, player2 = agent, human
        player1_name, player2_name = 'Robot', human_name
    agent.reset_episode(env)
    agent.load_state_value_table()

    # Start competition.
    while not env.is_done():
        # Player1 plays one step.
        r1, c1, symbol1 = player1.select_position(env)
        env = env.step(r1, c1, symbol1)
        print('Player1, {} ({}), puts ({}, {})'
              .format(player1_name, player1.player, r1, c1))
        env.show_board()
        print('---')

        if env.is_done():
            break

        # Player2 plays the next step.
        r2, c2, symbol2 = player2.select_position(env)
        env = env.step(r2, c2, symbol2)
        print('Player2, {} ({}), puts ({}, {})'
              .format(player2_name, player2.player, r2, c2))
        env.show_board()
        print('---')

    # Judge the winner.
    if env.winner == human.symbol:
        print('Congrats {}, you win!'.format(human_name))
    elif env.winner == -human.symbol:
        print('{} loses to Robot...'.format(human_name))
    else:
        print('{} and Robot tie.'.format(human_name))


def main():
    while True:
        cmd = input('Train robot (T) or play game (P)? ')
        if cmd in ['T', 'P']:
            break

    if cmd == 'T':
        self_train(epochs=100000, step_size=0.1, epsilon=0.1, 
                   print_per_epochs=500)
    elif cmd == 'P':
        human_agent_compete()


if __name__ == '__main__':
    main()
