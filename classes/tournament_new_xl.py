'''
[SZ]
player management based on create_database.py
tournament logic similar to tournament_parallel.py
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
import beck.beck_nnet
importlib.reload(beck.beck_nnet)
from beck.beck_nnet import NNetWrapper, OthelloNNet_resnet, NNetWrapper_color
import supervised_learning as sl
import mcts

importlib.reload(mcts)
from mcts import MCTS, MCTS_color
from bfts import BFTS
from utils import *

import create_database as cd
import pickle


# TOURNAMENT_BASE_LOC = '/scratch/zz737/fiar/tournaments/ai_all_player_round_robin_base.pkl'
TOURNAMENT_BASE_LOC = cd.TOURNAMENT_RES_LOC
# TOURNAMENT_BASE_backup_LOC = '/scratch/zz737/fiar/tournaments/ai_all_player_round_robin_base_backup.pkl' # backup should be in the same folder as the original
TOURNAMENT_BASE_backup_LOC = cd.TOURNAMENT_RES_LOC_backup
# DATABASE_LOC = '/scratch/zz737/fiar/tournaments/all_players.pkl'
DATABASE_LOC = cd.DATABASE_LOC
TOURNAMENT_NAME = 'tournament_test'


def get_participants():
    all_players=pd.read_pickle(DATABASE_LOC)
    
    # player selection: can vary from each time
    mask = all_players['tournament']==1 # the original players
    # subsample each n_mcts, cpuct combo, to only include 5 evenly spaced; best is included, but simply because it
    # is placed at the end when database is created, not a robust way to include it;
    gpb=all_players.loc[mask].groupby(['n_mcts','cpuct'])
    N_TO_SAMPLE = 5
    func = lambda x:x.iloc[np.linspace(0,len(x)-1,N_TO_SAMPLE).astype(int)]
    inds=gpb['mcts_iter'].apply(func).index.get_level_values(-1)
    subsampled_old_players = all_players.loc[inds]


    mask = all_players['id']=='tournament_4;mcts100;cpuct2;id-3754964;best' # the new best 64
    mask |= all_players['value_func_type'] =='cog' # cog value function
    mask |= all_players['tree_type'] =='bfts' # best first search tree type
    new_players = all_players.loc[mask]

    participants = pd.concat([subsampled_old_players,new_players],axis=0)
    return participants

def get_player(game,one_info,**kwargs):
    '''
    one_info: a row of all_players/participants, as pd.series 
    '''
    n_available_actions = game.getActionSize()
    if one_info.value_func_type  == 'nn' or one_info.tree_type=='mcts': # both situation need the nn
        if one_info.value_func_type  == 'nn':
            load_folder_file = one_info.value_func_location
        else:
            load_folder_file = one_info.mcts_location
        folder='/'.join(load_folder_file.split('/')[:-1])
        file=load_folder_file.split('/')[-1]
        if one_info.n_res is None:
            nnet = NNetWrapper(game)
        else: # n_res not None
            args = pickle.load(open(os.path.join(folder,'args.p'),'rb'))
            if one_info.color: 
                args['track_color']=True
                othello_resnet = OthelloNNet_resnet(game,args,return_compiled=True)
                nnet = NNetWrapper_color(game,args=args,nnet=othello_resnet) 
            else:
                othello_resnet = OthelloNNet_resnet(game,args,return_compiled=True)
                nnet = NNetWrapper(game,args=args,nnet=othello_resnet)
                
        nnet.load_checkpoint(folder, file)
    if one_info.value_func_type=='nn':
        val_func = nnet
    elif one_info.value_func_type =='cog': #NB need to be changed; player_info should have the w and C
        w = [0.01,0.2,0.05,2,100]
        C = 0.1
        args = [w,C]
        if one_info.tree_type =='mcts':
            cvnnet = cvn.NNetWrapper(game,nnet,args)
        else:
            cvnnet = cvn.NNetWrapper(game,None,args)
        val_func = cvnnet

    if one_info.tree_type =='mcts':
        n_mcts = one_info.n_mcts
        cpuct = one_info.cpuct
        args = dotdict({
        'numMCTSSims': n_mcts,
        'cpuct': cpuct,
        })
        if one_info.color:
            tree = MCTS_color(game,val_func,args)
        else:
            tree = MCTS(game,val_func,args)


    elif one_info.tree_type =='bfts':
        n_bfts = one_info.n_bfts
        prune_thresh = one_info.pruning_thresh
        args = dotdict({'numBFSsims':n_bfts,'PruningThresh':prune_thresh})
        tree = BFTS(game,val_func,args)
    
    else:
        print('undefined type')
        return

    if 'temp' in kwargs.keys():
        temp = kwargs['temp']
    else:
        temp = 0

    det = True
    
    def ai_func(*args):
        counts=tree.getActionProb(*args,temp=temp)
        return np.argmax(counts)

    if 'deterministic' in kwargs.keys():
    
        det = kwargs['deterministic']
    if det:
        # ai = lambda x: np.argmax(tree.getActionProb(x, temp=temp))
        ai = ai_func
    else:
        ai = lambda *args: np.random.choice(np.arange(n_available_actions),p=tree.getActionProb(*args, temp=temp))

    return ai, val_func, tree


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
    player1, val_func1, tree_1 = get_player(game, participant_info_1)
    player2, val_func2, tree_2 = get_player(game, participant_info_2)
    
    display = game.display if show_game else None
    arena = Arena(player1, player2, game, display=display)

    #[SZ] if human game, use the old function, no saving moves
    
    return arena.playGame(verbose=show_game)
    

participants_info = get_participants()
participants_info = participants_info
NEW_TOURNAMENT_DIR = os.path.join('/scratch/zz737/fiar/tournaments/results/',TOURNAMENT_NAME)



def play_ai_round_parallel(row_num):

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

    for _,p2 in participants_info.iterrows():
        if (p1.id == p2.id) or (p1.other_type == 'human') or (p2.other_type == 'human') or (not pd.isnull(base_result_df.loc[p1.id, p2.id])):
            printl(f'Skipping {p1.id} v/s {p2.id}...')
        else:
            printl(f'{p1.id} v/s {p2.id}!')
            base_result_df.loc[p1.id, p2.id] = play_game(g, p1, p2)
            K.clear_session() # free up memory?
        base_result_df.loc[[p1.id]].to_pickle(os.path.join(NEW_TOURNAMENT_DIR, results_name + '.pkl'))


def merge_res_to_base():
    '''
    Use this function after the round_robin; save a copy of the previous tournament_base, then update it with the new results
    '''
    old_res = pd.read_pickle(TOURNAMENT_BASE_LOC)
    old_res.to_pickle(TOURNAMENT_BASE_backup_LOC)
    print(f'back up at {TOURNAMENT_BASE_backup_LOC}')
    for fn in os.listdir(NEW_TOURNAMENT_DIR):
        f_loc=os.path.join(NEW_TOURNAMENT_DIR,fn)
        res = pd.read_pickle(f_loc)
        old_res.update(res)
    old_res.to_pickle(os.path.join(NEW_TOURNAMENT_DIR,'ai_all_player_round_robin_base.pkl'))
    print(f'new result saved at {NEW_TOURNAMENT_DIR}')
    old_res.to_pickle(os.path.join(TOURNAMENT_BASE_LOC))
    print(f'new result saved at {TOURNAMENT_BASE_LOC}')
    return old_res

def printl(*args, flush=True, **kwargs):
    time_str = f'[{datetime.datetime.today()}]'
    print(time_str, flush=flush, *args, **kwargs)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--play_as_human', action='store_true')
parser.add_argument('--play_ai_round_robin', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    if args.play_as_human:
        play_human_games()
    if args.play_ai_round_robin:
        play_ai_round_robin()
# import sys

# if __name__ == '__main__':
#     args = sys.argv[1:]
#     print(args)
#     play_ai_round_parallel(int(args[0]))
