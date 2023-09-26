from __future__ import print_function
import sys
# sys.path.append('..')
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__))) # get parent dir in a robust way
sys.path.append(d)

from game import Game
# from beck.beck_logic import Board
import numpy as np
from scipy.ndimage import convolve
import pygame
class BeckGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return BeckGame.square_content[piece]

    def __init__(self, m, n, k):
        # import fiarrl.classes.beck
        # import classes.beck
        assert k <= m and k <= n, "n-in-a-row must fit on the board all four ways!"
        self.m = m
        self.n = n
        self.k = k
        self.valid_wins = [
            np.identity(k),
            np.rot90(np.identity(k)),
            np.ones((1,k)),
            np.ones((k,1))
        ]

    def getInitBoard(self,initboard=None):
        # return initial board (numpy board)
        if initboard is None:
            pieces = [None]*self.m
            for i in range(self.m):
                pieces[i] = [0]*self.n
            return np.array(pieces)
        else:
            return initboard

    def getBoardSize(self):
        # (a,b) tuple
        return (self.m, self.n)

    def getActionSize(self):
        # return number of actions
        return self.m*self.n

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # if action == self.n*self.n:    < --- no passing
        #     return (board, -player)
        x, y = (int(action/self.n), action%self.n)
        new_board = np.copy(board)
        new_board[x][y] = player
        return (new_board, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = (board == 0).flatten().astype(int)
        return np.array(valids)

    def getValidMovesBatch(self,board_b,player):
        #[SZ] b x nvalidmoves
        valids=(board_b==0).reshape(board_b.shape[0],-1).astype(bool)
        return np.array(valids)

    def get_next_step_boards(self,board_batch):
        '''
        [SZ]
        board_batcch: b x game.m x game.n
        input_expanded: (b x action_size) x game.m x game.n
        valids: b x action_size
        '''
        
        valids = self.getValidMovesBatch(board_batch, 1)
        action_size = self.getActionSize()
        input_expanded = np.repeat(board_batch,action_size, axis=0) # repeat for : 1 2 -> 1 1 2 2
        inds_to_be_placed = np.tile(np.arange(action_size),board_batch.shape[0]) # tile for: 1 2 -> 1 2 1 2
        xs, ys = inds_to_be_placed // self.n, inds_to_be_placed % self.n
        expanded_batch_inds = np.arange(input_expanded.shape[0])
        input_expanded[expanded_batch_inds, xs, ys] = 1
        return input_expanded, valids

    def getGameEnded(self, board, player):        
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        for p in [player, -player]:
            filtered_board = (board == p).astype(int)
            for win in self.valid_wins:
                if self.k in convolve(
                    filtered_board,
                    win,
                    mode='constant',
                    cval=0.0,
                    origin=0
                ):
                    return (p * player)

        if (board != 0).sum() == self.m * self.n:
            return 0.0001
        
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.m*self.n)  # 1 for pass
        pi_board = np.reshape(pi, (self.m, self.n))
        return [
            (board, pi),
            (np.rot90(board, 2), np.rot90(pi_board, 2).ravel()),
            (np.flip(board, axis=0), np.flip(pi_board, axis=0).ravel()),
            (np.flip(board, axis=1), np.flip(pi_board, axis=1).ravel())
        ]

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    
    def encode_pieces(self,board):
        '''
        return a dic of integer encoding of the black and white pieces on the board
        '''
        players=['bp','wp']
        piece = [1,-1]
        encoded = {}
        for ii,p in enumerate(players):
            row,col = np.nonzero(board==piece[ii])
            enc = np.sum(2**(row * self.n + col))
            encoded[p] = enc
        return encoded


    @staticmethod
    def str_rep_to_array(board):
        if isinstance(board, (bytes, bytearray)): # mcts boards encoded as bytes-like
            return np.frombuffer(board, dtype=int)
        
        else: # bfts boards encoded as readable strings
            board_arr = []
            for row in board.split('\n'):
                row_arr = []
                for s in row.split(' '):
                    s = s.lstrip('[')
                    s = s.rstrip(']')
                    if s.lstrip('-').isnumeric():
                        row_arr.append(int(s))
                board_arr.append(row_arr)
            return np.array(board_arr)
        # str_rep_to_array = np.array([[int(i.rstrip(']')) for i in row.split(' ') if (i.lstrip('-').isnumeric() or i.rstrip(']').isnumeric() or i.rstrip(']').lstrip('-').isnumeric() or i.rstrip(']]').isnumeric() or i.rstrip(']]').lstrip('-').isnumeric()) ] for row in board.split('\n') ])
        # return str_rep_to_array

    @staticmethod
    def get_board_from_xo_str(board_str):
        '''
        [SZ]
        get the board, np array with 1/-1/0,
        from a string like:
        0 |- X - O X X O - - |
    1 |- X O O O X - - - |
    2 |- - - O - O X - - |
    3 |- - - X O - X O - |
         that comes from the display of a gameplay
        '''
        board_arr = np.array(board_str.split(' '))
        board_arr=np.array([s for s in board_arr if ('X' in s) or ('O' in s) or ('-' in s) ]).reshape(4,-1).astype('object')

        board_arr = np.array([-1 if 'X' in s else 1 if 'O' in s else 0 for s in board_arr.flatten()])
        board_arr = board_arr.reshape(4,-1).astype(int)

        return board_arr

    # @staticmethod
    # def display(board):
    #     m = board.shape[0]
    #     n = board.shape[1]
    #     print("   ", end="")
    #     for y in range(n):
    #         print(y, end=" ")
    #     print("")
    #     print("-----------------------")
    #     for y in range(m):
    #         print(y, "|", end="")    # print the row #
    #         for x in range(n):
    #             piece = board[y][x]    # get the piece to print
    #             print(BeckGame.square_content[piece], end=" ")
    #         print("|")

    #     print("-----------------------")


    @staticmethod
    def display(board,turn,player,game_num,gameover):

        m = board.shape[0] #row num
        n = board.shape[1] #col num
        BLACK = (0,0,0)
        WHITE = (255,255,255)
        DARK_BROWN = (222, 184, 135)
        LIGHT_BROWN = (255, 248, 220)
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE/2 - 5)
        height = SQUARESIZE * (m+1)
        width = SQUARESIZE * n
        size = (width, height)
        pygame.init()
        screen = pygame.display.set_mode(size)
        # print('screen')
        pygame.display.set_caption('4-in-a-row')
        pygame.draw.rect(screen, LIGHT_BROWN, (0,0,width,height))

        font = pygame.font.Font('freesansbold.ttf', 32)
        if player == 1:
            player_color = "black"
        elif player ==-1: 
            player_color = "white"
        else:
            player_color = 'draw'

        if gameover == False:
            text = font.render("Game: " + str(game_num) +"/32  "+ "Turn: " + str(turn) + "  Player: "+ player_color, True, BLACK, LIGHT_BROWN)
        elif gameover == True and player_color == 'draw':
            text = font.render("Game over! Turn: "+ str(turn)+ "  Result: "+ player_color ,True, BLACK, LIGHT_BROWN) 
        else: 
            text = font.render("Game over! Turn: "+ str(turn)+ "  Result: "+ player_color +" wins",True, BLACK, LIGHT_BROWN) 
        textRect = text.get_rect()
        textRect.center = (width//2, height*0.9)
        # print('check1')
        screen.blit(text, textRect)

        for row in range(m):
            for col in range(n):

                colour = BLACK if board[row,col] == 1 else WHITE if board[row,col] == -1 else DARK_BROWN
                pygame.draw.circle(screen, colour, (int(col*SQUARESIZE+SQUARESIZE/2), int(row*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()
