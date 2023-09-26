import pandas as pd
import sys,os,copy,pdb,importlib
sys.path.append('../classes')
sys.path.append('../analysis')
import numpy as np
import matplotlib.pyplot as plt

import tournament_new as tn
import create_database as cd
importlib.reload(tn)

import beck.beck_game
from importlib import reload
reload(beck.beck_game)
from beck.beck_game import BeckGame as Game
game = Game(4,9,4)

from arena import Arena
from mcts import MCTS
importlib.reload(tn)
game = Game(4,9,4)
all_p = pd.read_pickle(cd.DATABASE_LOC)

# res = tn.merge_res_to_base()
tournament_res = pd.read_pickle('/scratch/zz737/fiar/tournaments/ai_all_player_round_robin_base.pkl')

import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

import keras
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import tensorflow as tf
from tensorflow import Tensor

sys.path.append('..')
from neural_net import NeuralNet
from utils import *

reload(beck.beck_nnet)
from beck.beck_nnet import OthelloNNet, NNetWrapper, NNetWrapper_color

import beck.beck_nnet_pvcorr as bnp
reload(bnp)

from collections import OrderedDict


from pickle import Pickler, Unpickler

import supervised_learning as sl 
from supervised_learning import OthelloNNet_resnet
from beck.beck_nnet_pvcorr import NNetWrapper_pvcorr_color, OthelloNNet_with_PVcorr_color, OthelloNNet_with_PVcorr,NNetWrapper_pvcorr

nepochs = 80
othello_dict = OrderedDict()
nnet_dict = OrderedDict()
# othello_dict['res15_color_pvcorr'] = OthelloNNet_resnet(game, args=sl.get_args(n_res=15, epochs=nepochs, num_channels=256, track_color=True),return_compiled=False) #return_compiled=False important!
# nnet_dict['res15_color_pvcorr'] = NNetWrapper_pvcorr_color(game, args=othello_dict['res15_color_pvcorr'].args, nnet=OthelloNNet_with_PVcorr_color(game,args=othello_dict['res15_color_pvcorr'].args,sub_model=othello_dict['res15_color_pvcorr'].model))     


othello_dict['res3_pvcorr'] = OthelloNNet_resnet(game, args=sl.get_args(n_res=3, epochs=nepochs, num_channels=256, track_color=False),return_compiled=False) #return_compiled=False important!
nnet_dict['res3_pvcorr'] = NNetWrapper_pvcorr(game, args=othello_dict['res3_pvcorr'].args, nnet=OthelloNNet_with_PVcorr(game,args=othello_dict['res3_pvcorr'].args,sub_model=othello_dict['res3_pvcorr'].model))     


def main(i):
    
    name = list(nnet_dict.keys())[i]
    print(f'training {name}...')
    nnet = nnet_dict[name]
    if 'track_color' in nnet.args.keys():
        tc = nnet.args.track_color
    else:
        tc = False
    trainEx = sl.load_data(track_color=tc)

    # reduce trainex
    nte = len(trainEx)
    to_reduce = 1
    nte = nte // to_reduce
    trainEx = trainEx[:nte]

    nnet.args.batch_size = 64
    

    folder = '/scratch/zz737/fiar/sl/resnet'
    # filename = f'{name}_Ex_tournament_6_mcts100_cpuct2_id_1_iter55.pth.tar'
    # sub_folder = f'{name}_Ex_tournament_{ex_tournament}_{ex_id}_{ex_fn_part}'
    sub_folder = f'{name}_Ex_tournament_{sl.ex_tournament}_{sl.ex_id}_{sl.ex_fn_part}_reduce{to_reduce}'
    # filename = f'{name}_Ex_tournament_{ex_tournament}_{ex_id}_{ex_fn_part}'
    filename = 'final.pth.tar'
    
    # nnet.nnet.model.save(os.path.join(folder,filename))
    final_folder = os.path.join(folder,sub_folder)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    args_fn = os.path.join(final_folder, 'args.p')
    with open(args_fn, "wb") as f:
        Pickler(f).dump(nnet.args)

    nnet.train(trainEx,checkpoint_dir=final_folder)
    nnet.save_checkpoint(folder=final_folder, filename=filename)

    

    # nnet.save_checkpoint(folder='/scratch/zz737/fiar/sl', filename=f'{name}_Ex_tournament_6_mcts100_cpuct2_id_1_iter55.pth.tar')

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
