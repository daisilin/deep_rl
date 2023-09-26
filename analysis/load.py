import sys
import os
sys.path.insert(0,'../classes')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import logging
import numpy as np

import coloredlogs

from arena import Arena
from coach import Coach
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer
from mcts import MCTS
from utils import *
log = logging.getLogger(__name__)

from keras import backend as K

import tournament


def read_one_subject(sub_dir,params_names=None,name='_'):
    if params_names is None:
        params_names = ['pruning_thresh','stop_p','feat_drop','lapse','active_passive','center','two_con','two_uncon','three','four','lltest']
    params_l = []
    for i in range(1,6):
        fn = os.path.join(sub_dir,'params'+str(i)+'.csv')
        try:
            params = np.loadtxt(fn,delimiter=',')
            ll_fn = os.path.join(sub_dir,f'lltest{i}.csv')
            ll = np.mean(np.loadtxt(ll_fn))
            params=np.append(params,ll)

            params_l.append(params)
        except:
            pass
    
    if len(params_l)==0: # cog model not fitted on this subject yet
        return
    params_l = np.vstack(params_l)
    params_df = pd.DataFrame(params_l,columns=params_names)
    
    if 'mcts' in name:
        subname, iter = name.split(';')
        iter = int(iter)        
        mcts = int(subname.split('_')[0][4:])
        cpuct = int(subname.split('_')[1][5:])

    
        
        params_df['mcts'] = mcts
        params_df['cpuct'] = cpuct
        params_df['iter'] = iter
    
    params_names.extend(['mcts','cpuct','iter'])
    params_df = params_df.reindex(params_names,axis=1)
    
#     if isdepth:
#         subj_index = int(sub_dir.split('/')[-1])
    
#     params_df.loc[:,'mcts':'iter'] = params_df.loc[:,'mcts':'iter'].astype('int16')
    
    return params_df

def get_subject_name(sub_dir):
    fn = os.path.join(sub_dir,'1.csv')
    moves = np.loadtxt(fn,delimiter=',', dtype='str')
    name = moves[0].split('\t')[-2]
    return name

def read_all_subjects(splits_dir,params_names=None):
    params_all_subj_dict = {}
    for sub_dir in os.listdir(splits_dir):
        sub_dir = os.path.join(splits_dir, sub_dir)
        name = get_subject_name(sub_dir)
        try:
	        params_all_subj_dict[name] = read_one_subject(sub_dir,params_names=params_names,name=name)
        except:
        	print(f'{name} cannot be loaded')
    
    try:
        params_all_subj_dict = pd.concat(params_all_subj_dict,keys=params_all_subj_dict.keys())
    
        return params_all_subj_dict
    except:
        print('cog model not fit')
        return None
    

