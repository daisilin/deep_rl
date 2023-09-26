import numpy as np
from classes/beck import beck.beck_game
importlib.reload(beck.beck_game)
from beck.beck_game import BeckGame as Game
from beck.beck_players import HumanBeckPlayer
import arena
importlib.reload(arena)
from arena import Arena
m = 4
n = 9
game = Game(4,9,4)

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
print(board)

player=1
human_p = HumanBeckPlayer(game)
p_human = lambda x:human_p.play(x)
players = [p_human, None, p_human]
def getCanonicalForm(board, player):
    # return state if player==1, else return -state if player==-1
    return player*board
print(getCanonicalForm(board, player))
action = players[player + 1](getCanonicalForm(board, player))
print('action', action)
