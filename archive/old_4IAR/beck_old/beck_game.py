import sys
sys.path.append('..')

# Global imports
import numpy as np
from scipy.ndimage import convolve

# Local imports
from game import Game

class BeckGame(Game):
    def __init__(self, m=4, n=9, k=4):
        assert k <= m and k <= n, "n-in-a-row must fit on the board all four ways!"
        self.m, self.n, self.k = m, n, k
        self.valid_wins = [
            np.identity(k),
            np.rot90(np.identity(k)),
            np.ones((1,k)),
            np.ones((k,1))
        ]
        self.players = [1, 2]

    def get_initial_state(self):
        return np.zeros((self.m, self.n))

    def get_next_state(self, state, player, action):
        state = np.copy(state)
        state[action // self.n, action % self.n] = player
        return state, 3 - player

    def get_actions_size(self):
        return self.m * self.n

    def get_allowed_actions(self, state, player=1):
        return (state == 0).ravel()
        # allowed_coordinates = np.argwhere(state == 0)
        # return [x[0] * self.m + x[1] for x in allowed_coordinates]

    def get_is_terminal_state(self, state, player=1):
        if (state != 0).sum() == self.m * self.n:
            return True

        is_terminal_state = False
        for p in self.players:
            filtered_board = (state == p).astype(int)
            for win in self.valid_wins:
                if self.k in convolve(
                    filtered_board,
                    win,
                    mode='constant',
                    cval=0.0,
                    origin=0
                ):
                    is_terminal_state = True
                    break
            if is_terminal_state:
                break

        return is_terminal_state

    def get_result(self, state, player=1):
        winner = None

        for p in self.players:
            filtered_board = (state == p).astype(int)
            for win in self.valid_wins:
                if self.k in convolve(
                    filtered_board,
                    win,
                    mode='constant',
                    cval=0.0,
                    origin=0
                ):
                    winner = p
                    break
            if winner is not None:
                break

        if winner is not None:
            return 1 if winner == player else -1

        if (state != 0).sum() == self.m * self.n:
            return 0

        return None

    def get_canonical_form(self, state, player):
        canonical_state = np.copy(state)
        if player == 2:
            ones, twos = canonical_state == 1, canonical_state == 2
            canonical_state[ones] = 2
            canonical_state[twos] = 1
        return canonical_state

    def get_symmetries(self, state, pi):
        pi_board = np.reshape(pi, (self.m, self.n))
        return [
            (state, pi),
            (np.rot90(state, 2), np.rot90(pi_board, 2).ravel()),
            (np.flip(state, axis=0), np.flip(pi_board, axis=0).ravel()),
            (np.flip(state, axis=1), np.flip(pi_board, axis=1).ravel())
        ]

    def get_hash_of_state(self, state):
        return hash(state.tostring())
