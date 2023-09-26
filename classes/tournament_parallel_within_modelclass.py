'''
tournement_parallel within a modelclass; used for tournament when training one model with many copies; want to tournament only within the copy
'''
import datetime
import numpy as np
import os
from keras import backend as K
import pandas as pd

import tournament

# all dirs, are general dirs, with subdirs being the copies of the model, supplemented in play_ai_round_parallel by  model_copy_name
# now fed through write_tournament_parallel_within_modelclass.py
# participants_dir = '/scratch/zz737/fiar/tournaments/tournament_4' 
# results_dir = '/scratch/zz737/fiar/tournaments/results/tournament_4' 
# moves_dir = '/scratch/zz737/fiar/tournaments/tournament_4/moves/raw/' 


from arena import Arena
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer, RandomPlayer, GreedyBeckPlayer
from mcts import MCTS
from utils import *

additional_iter = ['mcts100_cpuct2;10;cog','mcts100_cpuct2;50;cog']

def join_dirs(model_copy_name, participants_dir, results_dir, moves_dir):
    participants_dir = os.path.join(participants_dir, model_copy_name)
    results_dir = os.path.join(results_dir, model_copy_name)
    moves_dir = os.path.join(moves_dir, model_copy_name)
    return participants_dir, results_dir, moves_dir


def get_participant_iters(model_copy_name, participants_dir, results_dir, moves_dir):
    '''
    return participant_iters, len(participant_iters)
    list of participants/models in a tournament 
    '''
    participants_dir, results_dir, moves_dir = join_dirs(model_copy_name, participants_dir, results_dir, moves_dir)

    model_class = model_copy_name.split('-')[0] # model copy name: eg checkpoints_mcts100_cpuct2_id-3751934
    model_class = model_class.split('_')[1:3] #skip checkpoints, id
    model_class = '_'.join(model_class) # eg mcts100_cpuct2

    all_iters = os.listdir(participants_dir)
    all_iters_int = []
    for iter in all_iters: # eg. checkpoint_0.pth.tar.examples
        if iter.startswith('checkpoint_') and not iter.endswith('examples'):
            iter_int = int(iter.split('_')[1].split('.')[0])
            all_iters_int.append(iter_int)

    all_iters_int = set(all_iters_int)

    participant_iters = [f'{model_class};{x}' for x in all_iters_int] # eg 'mcts80_cpuct3;57'
    participant_iters =  additional_iter + participant_iters
    print(participant_iters)
    return participant_iters, len(participant_iters)


def play_ai_round_parallel(row_num, model_copy_name, participants_dir, results_dir, moves_dir, results_name='round_robin',ngames=1):
    '''
    row_num: which instance as p1
    model_copy_name: supplement the dir names to locate the copy 

    all dirs, are general dirs, with subdirs being the copies of the model, supplemented in play_ai_round_parallel by  model_copy_name
    eg
    # participants_dir = '/scratch/zz737/fiar/tournaments/tournament_4' 
    # results_dir = '/scratch/zz737/fiar/tournaments/results/tournament_4' 
    # moves_dir = '/scratch/zz737/fiar/tournaments/tournament_4/moves/raw/' 

    '''


    # get participant_iters first
    participant_iters, num_participant_iters = get_participant_iters(model_copy_name, participants_dir, results_dir, moves_dir)
    # then update the dirs
    participants_dir, results_dir, moves_dir = join_dirs(model_copy_name, participants_dir, results_dir, moves_dir)

    # create the tournament_x/model_copy folders for each model copy 
    for d in [participants_dir, results_dir, moves_dir]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True) # recursively make

    # results_df_path = os.path.join(results_dir,results_name+'.csv') #results_name needs to match write_tournament_parallel_within_modeelclass_sz.py, results_name
    # if not os.path.exists(results_df_path):
    #     # results_df = pd.DataFrame(index=participant_iters, columns=participant_iters) 
    #     results_df = pd.DataFrame(0,index=participant_iters, columns=participant_iters)  #[SZ] init to 0 now, since want to accumulate results
    #     results_df.to_csv(results_df_path)
    #     print(f'results df not exist, creating at {results_df_path}')
        
    # results_df = pd.read_csv(results_df_path)
    # no need for results_df, just create an empty df
    results_df = pd.DataFrame(0,index=participant_iters, columns=participant_iters)  #[SZ] init to 0 now, since want to accumulate results
    results_df = results_df.set_index('Unnamed: 0')

    tournament.printl('Starting round robin!')
    results_name = f'round_robin_{row_num}'
    g = Game(4,9,4)

    p1 = participant_iters[row_num]
    
    for p2 in participant_iters:
        # if (p1 == p2) or (p1 == 'human') or (p2 == 'human') or (not pd.isnull(results_df.loc[p1, p2])):
        if (p1 == p2) or (p1 == 'human') or (p2 == 'human'): # [SZ]not skipping battle for which a result already exists, since we are accumulating the results
            tournament.printl(f'Skipping {p1} v/s {p2}...')
        else:
            tournament.printl(f'{p1} v/s {p2}!')
            # results_df.loc[p1, p2] = tournament.play_game(g, participants_dir, p1, p2, moves_dir=moves_dir) #input moves_dir
            for rep in range(ngames):
                results_df.loc[p1, p2] += tournament.play_game(g, participants_dir, p1, p2, moves_dir=moves_dir) #[SZ] changed to add instead of assignment, to accumulate result
            K.clear_session() # free up memory?
        results_df.iloc[[row_num]].to_csv(os.path.join(results_dir, results_name + '.csv'))

import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    play_ai_round_parallel(int(args[0]), *args[1:-1],int(args[-1])) # last one, ngames need to be turned into int


