'''
Usage: 
python compute_depth_parallel.py {i} {depth / entropy}
i: iloc in participants
depth or entropy to compute
or policy_value_correlation "pvcorr"
or "NsWeightedMeanEntropy"
'''
import sys
sys.path.append('../analysis')
sys.path.append('./analysis')

import pandas as pd 
import numpy as np 
import create_database as cd 
import tournament_new as tn 
import value_analysis as va 


import datetime
import numpy as np
import os
from keras import backend as K
import pandas as pd

from arena import Arena
from mcts import MCTS
from cog_related import cog_value_net as cvn

from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from mcts import MCTS
from bfts import BFTS
from utils import *



# agent type
tree_type = 'bfts'#'mcts'#'bfts'#
val_func_type = 'nn'#'cog'#'nn'#

# tree related params
n_tree_str = 'n_mcts' if tree_type=='mcts' else 'n_bfts' # NB this requires bfts to be the only tree_type alternative
N_tree = 100 # for the primary tree search counts, whether mcts or bfts
N_mcts = 100 # for when the tree search is bfts but val_func_type=='nn'
CPUCT = 2 if (val_func_type =='nn' or tree_type =='mcts') else None
TOURNAMENT = 5 if (val_func_type =='nn' or tree_type =='mcts') else None
# additional ID filters
nn_mcts_ID = 'id-0' if (val_func_type =='nn' or tree_type =='mcts') else ''
cog_id = 'cog_id_1' if val_func_type=='cog' else ''
bfts_id = 'prune2' if tree_type=='bfts' else ''

# selecting mcts100_cpuct2 tournament6 id-0
all_players = pd.read_pickle(cd.DATABASE_LOC)
mask = all_players['value_func_type'] == val_func_type
mask &= all_players['tree_type'] == tree_type
mask &= all_players[n_tree_str] == N_tree
if (tree_type=='mcts'): # CPUCT is only assigned when tree_type is mcts; but the variable CPUCT should be used for the path name
    mask &= all_players['cpuct'] == CPUCT
if TOURNAMENT is not None:
    mask &= all_players['tournament'] == TOURNAMENT
participants = all_players.loc[mask]
mask = [((nn_mcts_ID in pid) and (cog_id in pid) and (bfts_id in pid)) for pid in participants.id]
participants = participants.iloc[mask]

if (tree_type=='mcts') and (val_func_type=='nn'):
    DEPTH_RES_DIR = f'/scratch/zz737/fiar/depth/tournament_{TOURNAMENT}_mcts{N_tree}_cpuct{CPUCT}_{nn_mcts_ID}' # no _{nn_mcts_ID} for tournament1; or whenever no ID is needed
elif (tree_type=='mcts') and (val_func_type=='cog'):
    DEPTH_RES_DIR = f'/scratch/zz737/fiar/depth/hybrid_{cog_id}_tournament_{TOURNAMENT}_{tree_type}{N_tree}_cpuct{CPUCT}_{nn_mcts_ID}' # no _{nn_mcts_ID} for tournament1; or whenever no ID is needed
elif (tree_type=='bfts') and (val_func_type=='nn'):
    DEPTH_RES_DIR = f'/scratch/zz737/fiar/depth/hybrid_tournament_{TOURNAMENT}_mcts{N_mcts}_cpuct{CPUCT}_{nn_mcts_ID}_bfts{N_tree}_{bfts_id}' # no _{nn_mcts_ID} for tournament1; or whenever no ID is needed
elif (tree_type=='bfts') and (val_func_type=='cog'):
    DEPTH_RES_DIR = f'/scratch/zz737/fiar/depth/{cog_id}_bfts_{N_tree}_{bfts_id}' # no _{nn_mcts_ID} for tournament1; or whenever no nn_mcts_ID is needed


print(f'{len(participants)} participants')



def read_depth_entropy_result(to_read='depth'):
    '''
    read result; now only support if each result correspond to a value_func_iter
    '''
    participants = pd.read_pickle(os.path.join(DEPTH_RES_DIR,'participants.pkl'))

    # depth_df_all ={}
    df_all = []
    for fn in os.listdir(DEPTH_RES_DIR):
        if fn.startswith(to_read):
            num = int(fn.split('_')[1].split('.')[0])
            iter = participants['mcts_iter'].iloc[num] # prioritizing mcts_iter, if nan then go with value_func_iter; could be more principled; need to be careful about potential conflict in the future!
            if np.isnan(iter):
                iter = participants['value_func_iter'].iloc[num]
            if iter!='best':
                df_one = pd.read_pickle(os.path.join(DEPTH_RES_DIR,fn))
                df_one['iter'] = iter
                df_one['id'] = participants['id'].iloc[num]
                df_all.append(df_one)
    df_all = pd.concat(df_all)
    return df_all

def read_multi_result(to_read_l=['depth'],save=True):
    '''
    based on read_depth_entropy_result;
    combined analysis results from different modalities
    '''
    res_multi_df = []
    res_multi_df = read_depth_entropy_result(to_read=to_read_l[0])
    if len(to_read_l) > 1:
        for to_read in to_read_l[1:]:
            res = read_depth_entropy_result(to_read=to_read)
            res_multi_df = res_multi_df.join(res.set_index(['bp','wp','iter','npieces','id']),on=['bp','wp','iter','npieces','id'],how='inner')
    fn = '_'.join(to_read_l)
    fn = fn+'_combined.pkl'
    fn = os.path.join(DEPTH_RES_DIR,fn)
    if save:
        res_multi_df.to_pickle(fn)
    return res_multi_df


def main(i,to_compute_l=['depth']):
    game = Game(4,9,4)
    try:
        ai,val_func,tree = tn.get_player(game,participants.iloc[i])
    except:
        print(f'error in get_player')
        return 

    opt_boards, opt_values=va.load_opt_value_test_boards(filter_3iar=True)
    
    for to_compute in to_compute_l:
    
        if to_compute=='depth':
            depth_df_one_model = va.get_depth_one_move_all_boards_one_model(opt_boards,tree,tot = opt_boards.shape[0])
            # depth_df_one_model = va.get_depth_one_move_all_boards_one_model(opt_boards,tree,tot = 1)
            depth_df_one_model.to_pickle(os.path.join(DEPTH_RES_DIR,f'depth_{i}.pkl'))
            # return depth_df_one_model
        elif to_compute=='entropy':
            entropy_df_one_model = va.get_entropy_one_model_all_boards(opt_boards,val_func)
            entropy_df_one_model.to_pickle(os.path.join(DEPTH_RES_DIR,f'entropy_{i}.pkl'))
            # return entropy_df_one_model
        elif to_compute =='pvcorr':

            pvcorr_df_one_model = va.policy_val_correlation_all_board_one_model(opt_boards,game,val_func,flip_val=False)
            pvcorr_df_one_model.to_pickle(os.path.join(DEPTH_RES_DIR,f'pvcorr_{i}.pkl'))

        elif to_compute =='NsWeightedMeanEntropy':
            df_one_model = va.get_Ns_weighted_mean_entropy_one_move_all_boards_one_model(opt_boards, tree, tot = opt_boards.shape[0])
            df_one_model.to_pickle(os.path.join(DEPTH_RES_DIR, f'NsWeightedMeanEntropy_{i}.pkl'))

        else:
            print('unaccetable to_compute')
            pass
    
if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    if not os.path.exists(DEPTH_RES_DIR):
        os.makedirs(DEPTH_RES_DIR)
        print(f'{DEPTH_RES_DIR} does not exist. Created!')


    participants.to_pickle(os.path.join(DEPTH_RES_DIR,'participants.pkl'))
    main(int(args[0]),args[1:])
 
