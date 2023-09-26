'''
[SZ]
player management based on create_database.py
tournament logic similar to tournament_parallel.py
**multiple games between two stochastic agents!!!**
'''

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

import create_database_sz as cd
importlib.reload(cd)

from tournament_new import get_player

# TOURNAMENT_BASE_LOC = '/scratch/xl1005/deep-master/tournaments/ai_all_player_round_robin_base.pkl'
TOURNAMENT_BASE_LOC = cd.REPEATED_TOURNAMENT_RES_LOC
# TOURNAMENT_BASE_backup_LOC = '/scratch/xl1005/deep-master/tournaments/ai_all_player_round_robin_base_backup.pkl' # backup should be in the same folder as the original
TOURNAMENT_BASE_backup_LOC = cd.REPEATED_TOURNAMENT_RES_LOC_backup
# DATABASE_LOC = '/scratch/xl1005/deep-master/tournaments/all_players.pkl'
DATABASE_LOC = cd.DATABASE_LOC
# TOURNAMENT_NAME = 'repeated_tournament_567withhybrids_1all'
TOURNAMENT_NAME = 'repeated_tournament_moves_2' #REMEMBER TO CHANGE THIS
NEW_DATABASE_LOC = '/scratch/xl1005/deep-master/tournaments/all_players_xl_swap.pkl'
# NEW_DATABASE_LOC = '/scratch/xl1005/deep-master/tournaments/m_players_all.pkl'
moves_dir = '/scratch/xl1005/deep-master/tournaments/results/moves/raw'

def get_participants():
    # all_players=pd.read_pickle(DATABASE_LOC)
    all_players = pd.read_pickle(NEW_DATABASE_LOC)
    # player selection: can vary from each time

    # #=========#
    # mask = all_players['tournament']==1 # the original players
    # # subsample each n_mcts, cpuct combo, to only include 5 evenly spaced; best is included, but simply because it
    # # is placed at the end when database is created, not a robust way to include it;
    # gpb=all_players.loc[mask].groupby(['n_mcts','cpuct'])
    # N_TO_SAMPLE = 5
    # func = lambda x:x.iloc[np.linspace(0,len(x)-1,N_TO_SAMPLE).astype(int)]
    # inds=gpb['mcts_iter'].apply(func).index.get_level_values(-1)
    # subsampled_old_players = all_players.loc[inds]


    # mask = all_players['id']=='tournament_4;mcts100;cpuct2;id-3754964;best' # the new best 64
    # mask |= all_players['value_func_type'] =='cog' # cog value function
    # mask |= all_players['tree_type'] =='bfts' # best first search tree type
    # new_players = all_players.loc[mask]

    # participants = pd.concat([subsampled_old_players,new_players],axis=0)
    # #========#

    #========#
    # # all t567 including hybrids, cog models+bfts, t1 all
    mask = all_players.tournament ==8 
    # mask |= all_players.tournament ==7
    # mask |= all_players.tournament.isna()
    mask |= all_players.tournament==13
    mask |= all_players.tournament==15
    mask |= all_players.tournament==16
    mask |= all_players.tournament==14
    mask |= all_players.tournament==12
    mask &= all_players.other_type != 'n_mcts_changed'
    # mask &= all_players.value_func_iter == 'best'

    participants = all_players.loc[mask]
    #=========#
    # all t8,12,13,14,15,16, no hybrids
    # mask =  (all_players.tournament ==8)| (all_players.tournament ==12)| (all_players.tournament ==13)| (all_players.tournament ==14)| (all_players.tournament ==15)| (all_players.tournament ==16)
    # participants = all_players.loc[mask]
    # participants = all_players

    return participants

# def get_player(game,one_info,**kwargs):
#     '''
#     one_info: a row of all_players/participants, as pd.series 
#     '''
#     n_available_actions = game.getActionSize()
#     if one_info.value_func_type  == 'nn' or one_info.tree_type=='mcts': # both situation need the nn
#         if one_info.value_func_type  == 'nn':
#             load_folder_file = one_info.value_func_location
#         else:
#             load_folder_file = one_info.mcts_location
#         folder='/'.join(load_folder_file.split('/')[:-1])
#         file=load_folder_file.split('/')[-1]
#         nnet = nn(game)
#         nnet.load_checkpoint(folder, file)
#     if one_info.value_func_type=='nn':
#         val_func = nnet
#     elif one_info.value_func_type =='cog': #NB need to be changed; player_info should have the w and C
#         w = [0.01,0.2,0.05,2,100]
#         C = 0.1
#         args = [w,C]
#         if one_info.tree_type =='mcts':
#             cvnnet = cvn.NNetWrapper(game,nnet,args)
#         else:
#             cvnnet = cvn.NNetWrapper(game,None,args)
#         val_func = cvnnet

#     if one_info.tree_type =='mcts':
#         n_mcts = one_info.n_mcts
#         cpuct = one_info.cpuct
#         args = dotdict({
#         'numMCTSSims': n_mcts,
#         'cpuct': cpuct,
#         })
#         tree = MCTS(game,val_func,args)


#     elif one_info.tree_type =='bfts':
#         n_bfts = one_info.n_bfts
#         prune_thresh = one_info.pruning_thresh
#         args = dotdict({'numBFSsims':n_bfts,'PruningThresh':prune_thresh})
#         tree = BFTS(game,val_func,args)
    
#     else:
#         print('undefined type')
#         return

#     if 'temp' in kwargs.keys():
#         temp = kwargs['temp']
#     else:
#         temp = 0

#     det = True
    
#     def ai_func(x):
#         counts=tree.getActionProb(x,temp=temp)
#         return np.argmax(counts)

#     if 'deterministic' in kwargs.keys():
    
#         det = kwargs['deterministic']
#     if det:
#         # ai = lambda x: np.argmax(tree.getActionProb(x, temp=temp))
#         ai = ai_func
#     else:
#         ai = lambda x: np.random.choice(np.arange(n_available_actions),p=tree.getActionProb(x, temp=temp))

#     return ai, val_func, tree


def play_game(game, participant_info_1, participant_info_2, moves_dir = None):
    '''
    [sz modified] no save moves yet
    '''
    print('\n')
    #['SZ'] hiding ai info when human vs ai

    if participant_info_1.other_type=='human' or participant_info_2.other_type=='human':
        show_game = True
        print('Game beginning!')
    else:
        show_game = False
        print(f'Game beginning! {participant_info_1.id} v/s {participant_info_2.id}...')
    player1, val_func1, tree_1 = get_player(game, participant_info_1,temp=1/10, deterministic=False) # temperature here; important deterministic=False
    player2, val_func2, tree_2 = get_player(game, participant_info_2,temp=1/10, deterministic=False) # temperature here
    
    display = game.display if show_game else None
    arena = Arena(player1, player2, game, display=display,track_color=[participant_info_1.color,participant_info_2.color])

    #[SZ] if human game, use the old function, no saving moves
    
    # return arena.playGame(verbose=show_game)
    return arena.playGameSave(verbose=show_game,subjectID_l=[participant_info_1.id.to_string(),participant_info_2.id.to_string()],fd=moves_dir) # if fd not exists, should be automatically created


participants_info = get_participants()
participants_info = participants_info
# NEW_TOURNAMENT_DIR = os.path.join('/scratch/zz737/fiar/tournaments/results/',TOURNAMENT_NAME)
NEW_TOURNAMENT_DIR = os.path.join('/scratch/xl1005/deep-master/tournaments/results/',TOURNAMENT_NAME)



def play_ai_round_parallel(row_num,rep=1):

    if not os.path.exists(NEW_TOURNAMENT_DIR):
        os.makedirs(NEW_TOURNAMENT_DIR)
        # print(f'{NEW_TOURNAMENT_DIR} DOES NOT EXIST! ABORT.')
        # return
        print(f'{NEW_TOURNAMENT_DIR} created!')
        

    base_result_df = pd.read_pickle(TOURNAMENT_BASE_LOC)

    printl('Starting round robin!')
    results_name = f'round_robin_{row_num}'
    g = Game(4,9,4)

    p1 = participants_info.iloc[row_num]
    #subsample
    # sub_participants = participants_info.sample(n=430)
    #sub_participants = participants_info.iloc[[619,620,621,622,827,828,829,830,623,624,625,626,831,832,833,834]]
    #sub_participants = participants_info.sample(n=500)
    sub_participants = participants_info
    for _,p2 in sub_participants.iterrows():
    # for _,p2 in participants_info.iterrows():
        if (p1.id == p2.id) or (p1.other_type == 'human') or (p2.other_type == 'human'):
            printl(f'Skipping {p1.id} v/s {p2.id}...')
        else:
            printl(f'{p1.id} v/s {p2.id}!')
            for r in range(rep):
                if base_result_df.loc[p1.id, p2.id].sum()>=rep: # enough reps recorded
                    break
                print(f'rep {r}')
                one_res = play_game(g, p1, p2)
                print(f'res {one_res}')
                if one_res == 1:
                    base_result_df.loc[p1.id, (p2.id,'row_win')] += 1 
                elif one_res == -1:
                    base_result_df.loc[p1.id, (p2.id,'col_win')] += 1 
                elif one_res == 0.0001:
                    base_result_df.loc[p1.id, (p2.id,'draw')] += 1 
                else:
                    printl('game result not understood')
                K.clear_session() # free up memory?
        base_result_df.loc[[p1.id]].to_pickle(os.path.join(NEW_TOURNAMENT_DIR, results_name + '.pkl'))
    return base_result_df

def merge_res(save=False):
    res_l = []
    for fn in os.listdir(NEW_TOURNAMENT_DIR):
        if fn.startswith('round_robin') and not 'combined' in fn: # individual should be named round_robin_i.pkl, combined should have "combined" in the name 
            f_loc=os.path.join(NEW_TOURNAMENT_DIR,fn)
            res = pd.read_pickle(f_loc)
            res_l.append(res)
    res_l = pd.concat(res_l,axis=0,ignore_index=False)
    # res_l = res_l.loc[:,res_l.index]
    if save:
        res_l.to_pickle(os.path.join(NEW_TOURNAMENT_DIR,'round_robin_combined.p'))
    return res_l

def merge_res_to_base():
    '''
    Use this function after the round_robin; save a copy of the previous tournament_base, then update it with the new results
    '''
    old_res = pd.read_pickle(TOURNAMENT_BASE_LOC)
    old_res.to_pickle(TOURNAMENT_BASE_backup_LOC)
    print(f'back up at {TOURNAMENT_BASE_backup_LOC}')
    new_res = merge_res(save=False)
    old_res.update(new_res)
    old_res.to_pickle(os.path.join(NEW_TOURNAMENT_DIR,'ai_all_player_round_robin_base.pkl'))
    print(f'new result saved at {NEW_TOURNAMENT_DIR}')
    old_res.to_pickle(os.path.join(TOURNAMENT_BASE_LOC))
    print(f'new result saved at {TOURNAMENT_BASE_LOC}')
    return old_res

def printl(*args, flush=True, **kwargs):
    time_str = f'[{datetime.datetime.today()}]'
    print(time_str, flush=flush, *args, **kwargs)

import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    play_ai_round_parallel(int(args[0]))
