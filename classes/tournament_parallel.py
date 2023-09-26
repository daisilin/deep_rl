import datetime
import numpy as np
import os
from keras import backend as K
import pandas as pd

import tournament

participants_dir = '/scratch/zz737/fiar/tournaments/tournament_1' # dir for trained networks
results_dir = '/scratch/zz737/fiar/tournaments/results/tournament_3'
moves_dir = '/scratch/zz737/fiar/tournaments/tournament_3/moves/raw/'

participants = sorted([x[12:] for x in os.listdir(participants_dir)])

iters = {p: sorted(list(set([
        int(x.split('.')[0].split('_')[1]) \
        for x in os.listdir(os.path.join(participants_dir, 'checkpoints_' + p)) \
        if x.startswith('checkpoint') and x != 'checkpoint' and not x.endswith('examples')
    ]))) for p in participants
}

print(iters)

participant_iters = [f'{k};{x}' for k, v in iters.items() for x in v] + ['random', 'greedy']
# participant_iters = ['mcts25_cpuct1;1','mcts25_cpuct1;11','mcts25_cpuct1;31','mcts25_cpuct1;61','mcts80_cpuct1;1','mcts80_cpuct1;10','mcts80_cpuct1;28','mcts80_cpuct1;40'] +['random']

print(f'{len(participant_iters)} participant iterations!')

from arena import Arena
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer, RandomPlayer, GreedyBeckPlayer
from mcts import MCTS
from utils import *



def play_ai_round_parallel(row_num):
    results_df = results_dir+'/round_robin.csv'
    results_df = pd.read_csv(results_df)
    results_df = results_df.set_index('Unnamed: 0')

    tournament.printl('Starting round robin!')
    results_name = f'round_robin_{row_num}'
    g = Game(4,9,4)

    p1 = participant_iters[row_num]

    for p2 in participant_iters:
        if (p1 == p2) or (p1 == 'human') or (p2 == 'human') or (not pd.isnull(results_df.loc[p1, p2])):
            tournament.printl(f'Skipping {p1} v/s {p2}...')
        else:
            tournament.printl(f'{p1} v/s {p2}!')
            results_df.loc[p1, p2] = tournament.play_game(g, participants_dir, p1, p2)
            K.clear_session() # free up memory?
        results_df.iloc[[row_num]].to_csv(os.path.join(results_dir, results_name + '.csv'))

import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    play_ai_round_parallel(int(args[0]))


