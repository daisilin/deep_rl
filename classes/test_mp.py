import importlib
import datetime
import numpy as np
import os
from keras import backend as K
import pandas as pd

from arena import Arena

import cog_related
importlib.reload(cog_related)
from cog_related import cog_value_net as cvn


from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn 
import mcts

importlib.reload(mcts)
from mcts import MCTS
from bfts import BFTS
from utils import *

import time
from itertools import repeat

import create_database as cd

from multiprocessing import get_context, Pool

game = Game(4,9,4)
all_p = pd.read_pickle(cd.DATABASE_LOC)

prev_p_id = 'tournament_5;mcts100;cpuct2;id-0;best'
# prev_p_id = 'tournament_1;mcts100;cpuct3;best'
# prev_p_id = 'tournament_5;mcts100;cpuct2;id-1;27'
prev_p_row = cd.select_row_by_id(prev_p_id,all_p)
p1_row = prev_p_row

import copy

import ray

p1_row = copy.copy(p1_row)
iter = 35
tournament=6
p1_row.value_func_iter = iter
p1_row.mcts_iter = iter
p1_row.id = f'tournament_{tournament};mcts100;cpuct2;id-0;{iter}'
p1_row.tournament=tournament
p1_row.mcts_location = f'/scratch/zz737/fiar/tournaments/tournament_{tournament}/checkpoints_mcts100_cpuct2_id_0/checkpoint_{iter}.pth.tar'
p1_row.value_func_location = f'/scratch/zz737/fiar/tournaments/tournament_{tournament}/checkpoints_mcts100_cpuct2_id_0/checkpoint_{iter}.pth.tar'

p2_row = copy.copy(p1_row)
iter = 50
tournament=6
id=2
p2_row.value_func_iter = iter
p2_row.mcts_iter = iter
p2_row.id = f'tournament_{tournament};mcts100;cpuct2;id-{id};{iter}'
p2_row.tournament=tournament
p2_row.mcts_location = f'/scratch/zz737/fiar/tournaments/tournament_{tournament}/checkpoints_mcts100_cpuct2_id_{id}/checkpoint_{iter}.pth.tar'
p2_row.value_func_location = f'/scratch/zz737/fiar/tournaments/tournament_{tournament}/checkpoints_mcts100_cpuct2_id_{id}/checkpoint_{iter}.pth.tar'


one_info = p1_row

nnet = nn(game)
load_folder_file = one_info.value_func_location
folder='/'.join(load_folder_file.split('/')[:-1])
file=load_folder_file.split('/')[-1]
nnet.load_checkpoint(folder, file)

n_mcts = one_info.n_mcts
cpuct = one_info.cpuct
args = dotdict({
'numMCTSSims': n_mcts,
'cpuct': cpuct,
})
tree = MCTS(game,nnet,args)

ray.init()

@ray.remote
def getactprob(x):
    print(f'len Qsa {len(tree.Qsa.keys())}')
    return tree.search(x)

def main():
    temp=0
    g = game.getInitBoard()
    # with get_context("spawn").Pool(2) as pool:
    # # with Pool() as pool:
        # print(pool.starmap(getactprob,zip([g],repeat(tree))))
        # print(pool.map(getactprob,[g]*100))
    futures=[getactprob.remote(gg) for gg in [g]*100]
    print(ray.get(futures))
    # for _ in range(100):
    #     print(getactprob(g))

if __name__ == '__main__':
    st = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f'{st-end} duration')
