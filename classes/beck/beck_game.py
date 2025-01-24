from __future__ import print_function
import sys
from os.path import dirname, abspath
import numpy as np
from scipy.ndimage import convolve
import pygame
from game import Game

# Add parent directory to path for imports
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

class BeckGame(Game):
    """
    Implementation of the Beck variant of m,n,k-game (a generalization of tic-tac-toe).
    Players take turns placing pieces on an m×n board, trying to get k pieces in a row.
    
    Attributes:
        m (int): Number of rows on the board
        n (int): Number of columns on the board
        k (int): Number of pieces in a row needed to win
        valid_wins (list): List of numpy arrays representing winning patterns
        square_content (dict): Mapping of piece values to their string representations
    """

    square_content = {
        -1: "X",  # Player 1's pieces
        +0: "-",  # Empty space
        +1: "O"   # Player 2's pieces
    }

    def __init__(self, m, n, k):
        """
        Initialize the game board with dimensions m x n and winning condition k.

        Args:
            m (int): Number of rows
            n (int): Number of columns
            k (int): Number of pieces in a row needed to win
            
        Raises:
            AssertionError: If k is larger than either board dimension
        """
        assert k <= m and k <= n, "n-in-a-row must fit on the board all four ways!"
        self.m = m
        self.n = n
        self.k = k
        self.valid_wins = [
            np.identity(k),               # Diagonal
            np.rot90(np.identity(k)),     # Anti-diagonal
            np.ones((1,k)),              # Horizontal
            np.ones((k,1))               # Vertical
        ]

    @staticmethod
    def getSquarePiece(piece):
        """Convert numeric piece representation to string representation."""
        return BeckGame.square_content[piece]

    def getInitBoard(self, initboard=None):
        """
        Create initial game board or use provided board.

        Args:
            initboard (numpy.ndarray, optional): Initial board state

        Returns:
            numpy.ndarray: Initial board state
        """
        if initboard is None:
            pieces = [[0] * self.n for _ in range(self.m)]
            return np.array(pieces)
        return initboard

    def getBoardSize(self):
        """Return board dimensions as (rows, columns)."""
        return (self.m, self.n)

    def getActionSize(self):
        """Return total number of possible actions (board positions)."""
        return self.m * self.n

    def getNextState(self, board, player, action):
        """
        Get the next board state after player takes an action.

        Args:
            board (numpy.ndarray): Current board state
            player (int): Current player (1 or -1)
            action (int): Action index (0 to m*n-1)

        Returns:
            tuple: (new_board, next_player)
        """
        x, y = divmod(action, self.n)
        new_board = np.copy(board)
        new_board[x][y] = player
        return (new_board, -player)

    def getValidMoves(self, board, player):
        """
        Get vector of valid moves for current board state.

        Returns:
            numpy.ndarray: Binary vector where 1 indicates valid move
        """
        return (board == 0).flatten().astype(int)

    def getValidMovesBatch(self, board_b, player):
        """
        Get valid moves for a batch of board states.

        Args:
            board_b (numpy.ndarray): Batch of board states
            player (int): Current player

        Returns:
            numpy.ndarray: Boolean array of valid moves for each board
        """
        return (board_b == 0).reshape(board_b.shape[0], -1).astype(bool)

    def get_next_step_boards(self, board_batch):
        """
        Generate all possible next board states for a batch of current states.

        Args:
            board_batch (numpy.ndarray): Batch of current board states [b x game.m x game.n]

        Returns:
            tuple: (input_expanded, valids)
                - input_expanded: All possible next states [(b x action_size) x game.m x game.n]
                - valids: Valid moves mask [b x action_size]
        """
        valids = self.getValidMovesBatch(board_batch, 1)
        action_size = self.getActionSize()
        input_expanded = np.repeat(board_batch, action_size, axis=0)
        inds_to_be_placed = np.tile(np.arange(action_size), board_batch.shape[0])
        xs, ys = inds_to_be_placed // self.n, inds_to_be_placed % self.n
        expanded_batch_inds = np.arange(input_expanded.shape[0])
        input_expanded[expanded_batch_inds, xs, ys] = 1
        return input_expanded, valids

    def getGameEnded(self, board, player):
        """
        Check if the game has ended and return the result.

        Args:
            board (numpy.ndarray): Current board state
            player (int): Current player (1 or -1)

        Returns:
            float: 
                0 if game ongoing
                1 if player won
                -1 if player lost
                small non-zero value for draw
        """
        for p in [player, -player]:
            filtered_board = (board == p).astype(int)
            for win in self.valid_wins:
                if self.k in convolve(filtered_board, win, mode='constant', cval=0.0, origin=0):
                    return (p * player)

        if (board != 0).sum() == self.m * self.n:
            return 0.0001
        return 0

    def getCanonicalForm(self, board, player):
        """Return the canonical form of the board from player's perspective."""
        return player * board

    def getSymmetries(self, board, pi):
        """
        Get all symmetric board positions and their corresponding policies.

        Args:
            board (numpy.ndarray): Current board state
            pi (numpy.ndarray): Policy vector

        Returns:
            list: List of (board, policy) tuples for all symmetries
        """
        assert len(pi) == self.m * self.n
        pi_board = np.reshape(pi, (self.m, self.n))
        return [
            (board, pi),
            (np.rot90(board, 2), np.rot90(pi_board, 2).ravel()),
            (np.flip(board, axis=0), np.flip(pi_board, axis=0).ravel()),
            (np.flip(board, axis=1), np.flip(pi_board, axis=1).ravel())
        ]

    def stringRepresentation(self, board):
        """Convert board to string representation for storage."""
        return board.tostring()

    def stringRepresentationReadable(self, board):
        """Convert board to human-readable string representation."""
        return "".join(self.square_content[square] for row in board for square in row)

    def encode_pieces(self, board):
        """
        Encode board state as integers for black and white pieces.

        Returns:
            dict: Encoded positions for black ('bp') and white ('wp') pieces
        """
        players = ['bp', 'wp']
        piece = [1, -1]
        encoded = {}
        for ii, p in enumerate(players):
            row, col = np.nonzero(board == piece[ii])
            enc = np.sum(2 ** (row * self.n + col))
            encoded[p] = enc
        return encoded

    @staticmethod
    def str_rep_to_array(board):
        """
        Convert string representation back to numpy array.

        Args:
            board: String or bytes representation of board

        Returns:
            numpy.ndarray: Board state array
        """
        if isinstance(board, (bytes, bytearray)):
            return np.frombuffer(board, dtype=int)
        
        board_arr = []
        for row in board.split('\n'):
            row_arr = []
            for s in row.split(' '):
                s = s.lstrip('[').rstrip(']')
                if s.lstrip('-').isnumeric():
                    row_arr.append(int(s))
            board_arr.append(row_arr)
        return np.array(board_arr)

    @staticmethod
    def get_board_from_xo_str(board_str):
        """
        Convert display string representation to board array.

        Args:
            board_str (str): String representation with X, O, and - characters

        Returns:
            numpy.ndarray: Board state array with 1, -1, and 0 values
        """
        board_arr = np.array(board_str.split(' '))
        board_arr = np.array([s for s in board_arr if ('X' in s) or ('O' in s) or ('-' in s)]).reshape(4, -1).astype('object')
        board_arr = np.array([-1 if 'X' in s else 1 if 'O' in s else 0 for s in board_arr.flatten()])
        return board_arr.reshape(4, -1).astype(int)

    @staticmethod
    def display(board, turn, player, game_num, gameover):
        """
        Display the game board using Pygame.

        Args:
            board (numpy.ndarray): Current board state
            turn (int): Current turn number
            player (int): Current player
            game_num (int): Game number
            gameover (bool): Whether the game has ended
        """
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        DARK_BROWN = (222, 184, 135)
        LIGHT_BROWN = (255, 248, 220)
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE/2 - 5)
        
        m, n = board.shape
        height = SQUARESIZE * (m + 1)
        width = SQUARESIZE * n
        
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('4-in-a-row')
        pygame.draw.rect(screen, LIGHT_BROWN, (0, 0, width, height))

        font = pygame.font.Font('freesansbold.ttf', 32)
        player_color = "black" if player == 1 else "white" if player == -1 else "draw"

        if not gameover:
            text = font.render(f"Game: {game_num}/32  Turn: {turn}  Player: {player_color}", True, BLACK, LIGHT_BROWN)
        elif player_color == 'draw':
            text = font.render(f"Game over! Turn: {turn}  Result: {player_color}", True, BLACK, LIGHT_BROWN)
        else:
            text = font.render(f"Game over! Turn: {turn}  Result: {player_color} wins", True, BLACK, LIGHT_BROWN)

        textRect = text.get_rect()
        textRect.center = (width//2, height * 0.9)
        screen.blit(text, textRect)

        for row in range(m):
            for col in range(n):
                colour = BLACK if board[row, col] == 1 else WHITE if board[row, col] == -1 else DARK_BROWN
                pygame.draw.circle(screen, colour, 
                                 (int(col*SQUARESIZE + SQUARESIZE/2), 
                                  int(row*SQUARESIZE + SQUARESIZE/2)), 
                                 RADIUS)
        
        pygame.display.update()