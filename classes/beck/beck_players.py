import numpy as np
import math
import pygame
import sys

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanBeckPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        action_made = False
        # for i in range(len(valid)):
        #     if valid[i]:
        #         print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while action_made == False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:

                    sys.exit()     
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # print('beckplayer1')

                    posx = event.pos[0]   
                    posy = event.pos[1]
                    col = int(math.floor(posx/100))
                    row = int(math.floor(posy/100))
                    #input_move = input()
                    input_a = [row, col]
                    print('input,',input_a)
                    #input_a = input_move.split(" ")
                    if len(input_a) == 2:
                        # try:
                        x,y = [int(i) for i in input_a]
                        if ((0 <= x) and (x < self.game.m) and (0 <= y) and (y < self.game.n)):
                        # or \
                        # ((x == self.game.m) and (y == 0)):
                            a = self.game.n * x + y if x != -1 else self.game.n ** 2
                            if valid[a]:
                                # print('beckplayercheck2')

                                action_made=True
                                    # break
                        # except ValueError:
                            # Input needs to be an integer
                            # 'Invalid integer'
                            # else:
                            #     print('Invalid move')
        return a


class GreedyBeckPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:  # If the move is not valid...
                continue        # don't consider it
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getGameEnded(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


class NNPolicyPlayer():
    '''
    [SZ] nnet without mcts, using policy
    '''
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        p,v = self.nnet.predict(board)
        p *= valids
        return np.random.choice(np.nonzero(p==np.max(p))[0])

class NNValuePlayer():
    '''
    [SZ] nnet without mcts, using value
    '''
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet

    def play(self, board, curPlayer):
        valids = self.game.getValidMoves(board, 1)
        v_l = []
        for action,valid in enumerate(valids): # valids is a mask of len action size
            if valid:
                board_next, curPlayer_next = self.game.getNextState(board, curPlayer, action)
                p,v = self.nnet.predict(board_next * curPlayer_next) # to get the canonical board
            else:
                v = 1000 # a ridiculously large number that cannot be the min
            v_l.append(np.squeeze(v))
        
        return np.random.choice(np.nonzero(v_l==np.min(v_l))[0]), v_l


