import numpy as np
import sys,os,copy,pdb,importlib
import pandas as pd
print(pd.__version__)
from tqdm import tqdm
# from keras import backend as K

# from beck.beck_nnet import NNetWrapper as nn

from pickle import Pickler, Unpickler
import pickle

import beck.beck_game
importlib.reload(beck.beck_game)
from beck.beck_game import BeckGame as Game
from beck.beck_players import HumanBeckPlayer, RandomPlayer
from beck.beck_display import BeckDisplay 
import arena
importlib.reload(arena)
from arena import Arena
m = 4
n = 9
game = Game(4,9,4)
TOURNAMENT_NAME='gui_test'
database_dir = '../tournaments/all_players.pkl'
moves_dir = '../tournaments/result/testmove'
def getInitBoard(initboard=None):
    # return initial board (numpy board)
    if initboard is None:
        pieces = [None]*m
        for i in range(m):
            pieces[i] = [0]*n
        return np.array(pieces)
    else:
        return initboard

board = getInitBoard()
#print(board)

player=1
human_p = HumanBeckPlayer(game)
p_human = lambda x:human_p.play(x)
rand_p = RandomPlayer(game)
rand_player = lambda x: rand_p.play(x)
players = [rand_player, p_human]


arena = Arena(players[0],players[1], game, display=game.display)
res=arena.playGame(verbose=True,initboard=None)
