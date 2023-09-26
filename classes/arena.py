import logging

from tqdm import tqdm
import numpy as np
from copy import copy
from utils import *
import pdb
import pygame
import sys

log = logging.getLogger(__name__)

def refresh(tree1,tree2):
    for tree in [tree1, tree2]:
        if tree is not None:
            tree.refresh()

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, tree1=None,tree2=None,display=None,track_color=[False,False],flip_color=False,game_num=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.tree1 = tree1
        self.tree2 = tree2
        self.track_color = track_color
        self.flip_color = flip_color
        self.game_num = game_num
        

    def playGameSave(self,verbose=False,nnet=None,subjectID_l=['default1','default2'],fd='../models/moves/'):
        '''
        [SZ] playGame once and save; helper func for tournament
        '''
        win_result, moves_result = self.playGame(verbose=verbose, nnet=nnet, is_save_moves=True)
        for i in range(2):
            # save two players moves one by one
            save_moves(moves_result,tosave=i,subjectID=subjectID_l[i],model_class=None, model_instance=None, temp=None,
                fd = fd)
        return win_result


    def playGame(self, verbose=False, nnet=None, is_save_moves=False,initboard=None,game_num=None):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        [SZ] nnet can be None, a single network, or a list of two networks
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        # if initboard is None:
        #     board = self.game.getInitBoard(initboard=initboard)
        # else:
        #     board=initboard
        board = self.game.getInitBoard(initboard=initboard)
        it = 0

        ##[SZ]for saving moves
        moves_list = [[],[]] # for p1(black), p2(white), history current moves
        pieces_list = [[],[]] # for both p1, p2, history of codes for black, white pieces on the board
        pieces =[0,0] # initial code for black, white pieces on the board
        value_list = [[],[]]

        while self.game.getGameEnded(board, curPlayer) == 0:
            curP_ind = int(0.5 - curPlayer/2)
            ##[SZ] getting, displaying and saving nnet value
            # if nnet is not None:
            #     if isinstance(nnet,list):
            #         if nnet[curP_ind] is not None:
            #             value = nnet[curP_ind].predict(self.game.getCanonicalForm(board, curPlayer))[1]
            #         else:
            #             value = -404 #non existent value hard coded -404
            #     else:
            #         value = nnet.predict(self.game.getCanonicalForm(board, curPlayer))[1]

            #     value_list[curP_ind].append(value) # if one nnet, saving its value for both player's states; if two nnets, saving the respective values for their own states

            #     # print(f"Value of player to go: {nnet.predict(self.game.getCanonicalForm(board, curPlayer))[1]}")
            #     print(f"Value of player to go: {value}")
            # else:
            #     value_list[curP_ind].append(-404) #non existent value hard coded -404

            it += 1

            ##[DL] for exiting pygame GUI
            if verbose:
                assert self.display
                #print("Turn ", str(it), "Player ", str(curPlayer))
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         sys.exit()
                self.display(board, it, curPlayer, self.game_num,gameover=False)


            # if verbose:
            #     assert self.display
            #     print("Turn ", str(it), "Player ", str(curPlayer))
            #     self.display(board)
            
            if self.track_color[curP_ind]:
                curP_color = np.abs(1 - curP_ind) # 1 for black
                if self.flip_color:
                    curP_color = np.abs(1- curP_color) # 0 for black
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer), curP_color)
                # print('action')
            else:
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))
                # print('action')

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            ##[SZ] for saving moves
            encoded_move = 2**action
            
            moves_list[curP_ind].append(encoded_move)
            pieces_list[curP_ind].append(copy(pieces))
            pieces[curP_ind] += encoded_move

            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            print(board)

        if verbose:
            assert self.display
            #print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            game_result_txt = self.game.getGameEnded(board, 1)
            print(game_result_txt)
            self.display(board, it, game_result_txt,self.game_num,gameover=True)
            pygame.time.wait(3000)

        win_result = curPlayer * self.game.getGameEnded(board, curPlayer)

        moves_result = [] # list of two lists
        if is_save_moves:
            for player in range(2): # 0 and 1, corresponding to 1 and -1, black and white
                pieces_list_1p = np.array(pieces_list[player]).reshape(-1,1)
                nmoves = pieces_list_1p.shape[0]
                color_list = np.ones((nmoves,1)) * player #color using number now
                moves_list_1p = np.array(moves_list[player]).reshape(-1,1)
                value_list_1p = np.array(value_list[player]).reshape(-1,1) # saving value; additional column not used in fitting
                moves_result_1p = np.hstack([pieces_list_1p,color_list,moves_list_1p,value_list_1p]) # np array: cols are: code for black pieces; code for white pieces; color(in 0/1) for the player; code for the move; 
                                                                            # adding response time and player ID later when pooling everything together
                moves_result.append(moves_result_1p)

            return win_result, moves_result
        else:
            return win_result


    def playGames(self, num, verbose=False, nnet=None, is_save_moves=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        ##[SZ] init moves_result_multigame
        moves_result_multigame = [[],[]] # list of two lists, for p1 and p2; each will be np arrays concatenated

        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            if is_save_moves:
                gameResult, moves_result = self.playGame(verbose=verbose, nnet=nnet, is_save_moves=True)
                moves_result_multigame[0].append(moves_result[0])
                moves_result_multigame[1].append(moves_result[1])
            else:
                gameResult = self.playGame(verbose=verbose, is_save_moves=False)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

            refresh(self.tree1,self.tree2)
            

        self.player1, self.player2 = self.player2, self.player1
        self.track_color = [self.track_color[1],self.track_color[0]]

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            if is_save_moves:
                gameResult, moves_result = self.playGame(verbose=verbose, nnet=nnet, is_save_moves=True)
                moves_result_multigame[0].append(moves_result[1]) #now player's color reversed
                moves_result_multigame[1].append(moves_result[0])
            else:
                gameResult = self.playGame(verbose=verbose, is_save_moves=False)

            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

            refresh(self.tree1,self.tree2)
        
        if is_save_moves:
            moves_result_multigame[0] = np.vstack(moves_result_multigame[0])
            moves_result_multigame[1] = np.vstack(moves_result_multigame[1])

            return oneWon, twoWon, draws, moves_result_multigame
        else:
            return oneWon, twoWon, draws