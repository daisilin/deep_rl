import datetime
import numpy as np
from tqdm import tqdm
from random import shuffle
import os
from keras import backend as K
from cog_related import cog_value_net as cvn 

participants_dir = '/scratch/zz737/fiar/tournaments/tournament_1' # dir for trained networks
# results_dir = '/scratch/zz737/fiar/tournaments/results/tournament_3' # no need to use if just for human, and csv saved at current dir
moves_dir = '/scratch/zz737/fiar/tournaments/tournament_3/moves/raw/'

# /scratch/jt3974/tournaments

participants = sorted([x[12:] for x in os.listdir(participants_dir)])

iters = {p: sorted(list(set([
        int(x.split('.')[0].split('_')[1]) \
        for x in os.listdir(os.path.join(participants_dir, 'checkpoints_' + p)) \
        if x.startswith('checkpoint') and x != 'checkpoint' and not x.endswith('examples')
    ]))) for p in participants
}

print(iters)
# print(datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(participants_dir, participants[0], f'checkpoint_{35}.pth.tar.index'))))



import pandas as pd

from arena import Arena
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer, RandomPlayer, GreedyBeckPlayer
from mcts import MCTS
from utils import *

def get_args(participants_dir, participant, ix, mcts_sims, cpuct):
    '''
    [SZ] direct_participants_dir: if True, no need to join participants_dir and 'check_points_'+participant, just use participants; 
    used in the case of: multiple copies of the same model, each copy has its own round_robin, participants_dir point to that copy 
    currently use 'checkpoints' to tell
    '''
    if 'checkpoints' in participants_dir:
        direct_participants_dir = True
    else:
        direct_participants_dir = False

    if direct_participants_dir:
        load_folder = participants_dir
    else:
        load_folder = os.path.join(participants_dir, 'checkpoints_' + participant)

    if ix.isnumeric():
        fn = f'checkpoint_{ix}.pth.tar'
    else:
        fn = f'{ix}.pth.tar' # eg best / temp

    return dotdict({
        'tempThreshold': 15,
        'numMCTSSims': mcts_sims,
        'cpuct': cpuct,
        'load_model': True,
        'load_folder_file': (load_folder, fn),
    })

def get_player(game, participants_dir, participant_iter, is_return_mcts=False):
    '''
    [SZ] modified to return two things: player, value_function (None when not nnet)
    '''
    if participant_iter == 'human':
        player = HumanBeckPlayer(game)
        return lambda x: player.play(x), None
    elif participant_iter == 'random':
        player = RandomPlayer(game)
        return lambda x: player.play(x), None
    elif participant_iter == 'greedy':
        player = GreedyBeckPlayer(game)
        return lambda x: player.play(x), None

    #[SZ] modified to include cognitive model
    else:
        name_split = participant_iter.split(';')
        participant, ix = name_split[:2]
        mcts_sims = int(participant.split('_')[0][4:])
        cpuct = int(participant.split('_')[1][5:])
        args = get_args(participants_dir, participant, ix, mcts_sims, cpuct)
        # nnet = nn(game)
        nnet = NNetWrapper(game)

        if args.load_model:
            # log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        nmcts = MCTS(game, nnet, args)
        if len(name_split) > 2: 
            if name_split[2] == 'cog': # mcts100_cpuct2;37;cog
                w = np.array([0.1,1/3,1/3,5,10])*10
                C = 0.1
                cv_net = cvn.NNetWrapper(game,nnet,[w,C])
                cvnmcts = MCTS(game, cv_net, args)
                ai_cv = lambda x: np.argmax(cvnmcts.getActionProb(x, temp=0)) #temp=0
                return ai_cv, cv_net
        #[SZ] can return the nmcts object or not
        if is_return_mcts:
            return lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), nnet, nmcts 
        else:
            return lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), nnet 

# participant_iters = [f'{k};{x}' for k, v in iters.items() for x in v] + ['random', 'greedy', 'human']
participant_iters = [f'{k};{x}' for k, v in iters.items() for x in v] + ['random', 'greedy']
# participant_iters = ['mcts25_cpuct1;1','mcts25_cpuct1;11','mcts25_cpuct1;31','mcts25_cpuct1;61','mcts80_cpuct1;1','mcts80_cpuct1;10','mcts80_cpuct1;28','mcts80_cpuct1;40'] +['random']

print(f'{len(participant_iters)} participant iterations!')


def play_game(game, participants_dir, participant_iter_1, participant_iter_2, moves_dir = moves_dir):
    '''
    [sz modified] moves_dir in the args
    '''
    print('\n')
    #['SZ'] hiding ai info when human vs ai
    if participant_iter_1=='human' or participant_iter_2=='human':
        print('Game beginning!')
    else:
        print(f'Game beginning! {participant_iter_1} v/s {participant_iter_2}...')
    player1, val_func1 = get_player(game, participants_dir, participant_iter_1)
    player2, val_func2 = get_player(game, participants_dir, participant_iter_2)
    show_game = (participant_iter_1 == 'human') or (participant_iter_2 == 'human')
    display = game.display if show_game else None
    arena = Arena(player1, player2, game, display=display)

    #[SZ] if human game, use the old function, no saving moves
    if show_game:
        return arena.playGame(verbose=show_game)
    else:

        nnet_l = [val_func1, val_func2]

        win_res = arena.playGameSave(verbose=show_game,nnet=nnet_l,subjectID_l=[participant_iter_1,participant_iter_2],fd=moves_dir) # if fd not exists, should be automatically created

        return win_res

def play_human_games():
    results_df = pd.DataFrame(index=participant_iters, columns=participant_iters)
    results_name = 'vs_human'
    #[SZ] changed from iters to iters_human, by subsampling the iters
    iters_human = select_n_instances_each_from_iters(iters,1)
    human_vs_iters_black = [z for y in {k: [f'{k};{x}' for x in v if (x % 2 == 1) or (x == v[-1])] for k, v in iters_human.items()}.values() for z in y] + ['random', 'greedy']
    human_vs_iters_white = [z for y in {k: [f'{k};{x}' for x in v if (x % 2 == 0) or (x == v[-1])] for k, v in iters_human.items()}.values() for z in y] + ['random', 'greedy']

    # make it random
    shuffle(human_vs_iters_black)
    shuffle(human_vs_iters_white)

    g = Game(4,9,4)
    # for opponent in human_vs_iters_black:
    # [SZ] use progress bar?
    for opponent in tqdm(human_vs_iters_black, desc="human vs ai black"):
        results_df.loc[opponent, 'human'] = play_game(g, participants_dir, opponent, 'human')
        # results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))
        results_df.to_csv(os.path.join('./', results_name + '.csv')) # for other people to use, just save at the current dir
    # for opponent in human_vs_iters_white:
    for opponent in tqdm(human_vs_iters_white, desc="human vs ai white"):
        results_df.loc['human', opponent] = play_game(g, participants_dir, 'human', opponent)
        # results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))
        results_df.to_csv(os.path.join('./', results_name + '.csv')) # for other people to use, just save at the current dir

    print('Done!')

def printl(*args, flush=True, **kwargs):
    time_str = f'[{datetime.datetime.today()}]'
    print(time_str, flush=flush, *args, **kwargs)

def play_ai_round_robin():
    printl('Starting round robin!')
    results_name = 'round_robin'

    results_df = pd.DataFrame(index=participant_iters, columns=participant_iters)

    # try:
    #     results_df = pd.read_csv(os.path.join(results_dir, results_name + '.csv'), index_col=0)
    # except:
    #     results_df = pd.DataFrame(index=participant_iters, columns=participant_iters)
    
    g = Game(4,9,4)
    
    for p1 in participant_iters:
        for p2 in participant_iters:
            if (p1 == p2) or (p1 == 'human') or (p2 == 'human') or (not pd.isnull(results_df.loc[p1, p2])):
                printl(f'Skipping {p1} v/s {p2}...')
            else:
                printl(f'{p1} v/s {p2}!')
                results_df.loc[p1, p2] = play_game(g, participants_dir, p1, p2)
                K.clear_session() # free up memory?
                results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))


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



import multiprocessing as mp