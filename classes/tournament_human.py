
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
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer


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
import copy
import create_database as cd
import pickle
from random import shuffle
from tqdm import tqdm


# TOURNAMENT_BASE_LOC = '/scratch/zz737/fiar/tournaments/ai_all_player_round_robin_base.pkl'
TOURNAMENT_BASE_LOC = cd.TOURNAMENT_RES_LOC
# TOURNAMENT_BASE_backup_LOC = '/scratch/zz737/fiar/tournaments/ai_all_player_round_robin_base_backup.pkl' # backup should be in the same folder as the original
TOURNAMENT_BASE_backup_LOC = cd.TOURNAMENT_RES_LOC_backup
# DATABASE_LOC = '/scratch/zz737/fiar/tournaments/all_players.pkl'
DATABASE_LOC = cd.DATABASE_LOC
TOURNAMENT_NAME = 'tournament_test'
moves_dir = '../final_agents/results/moves/raw'

def create_human_player(name):
    all_players = all_players = pd.read_pickle(DATABASE_LOC)
    one_player = all_players.iloc[0]
    one_player = copy.copy(one_player)
    one_player.loc['tree_type'] = np.nan
    one_player.loc['value_func_type'] = np.nan
    one_player.loc['other_type'] = 'human'
    one_player.loc['n_mcts'] = np.nan
    one_player.loc['cpuct'] = np.nan
    one_player.loc['value_func_iter'] = np.nan
    one_player.loc['mcts_iter'] = np.nan
    one_player.loc['mcts_location']=np.nan
    one_player.loc['mcts_location']=np.nan
    one_player.loc['value_func_location']=np.nan
    one_player.loc['n_bfts'] = np.nan
    one_player.loc['pruning_thresh'] = np.nan
    one_player.loc['tournament'] = np.nan
    one_player.loc['better_than_last'] = np.nan
    one_player.loc['n_res'] = np.nan
    one_player.loc['model_line'] = np.nan
    one_player.loc['tempThreshold'] = np.nan
    one_player.loc['continuous_training'] = np.nan
    one_player.loc['color'] = np.nan
    one_player.loc['dir_alpha'] = np.nan

    one_player.loc['id'] = name
    COLUMNS=['tree_type','value_func_type','other_type','n_mcts','cpuct','value_func_iter','mcts_iter','n_bfts','pruning_thresh','center','2_con','2_uncon','3','4','oppo_scale','id','tournament','mcts_location','value_func_location','n_res','model_line','tempThreshold','dir_alpha','continuous_training','color']
    new_player = pd.DataFrame([],columns=COLUMNS)
    new_player = new_player.append(one_player,ignore_index=True)
    return new_player

def get_participants():
    all_players=pd.read_pickle(DATABASE_LOC)
    
    # player selection: can vary from each time

    mask = all_players['id'] == 'tournament_16;mcts100;cpuct2;id-res3-0;94'
    mask |= all_players['id'] == 'tournament_16;mcts100;cpuct2;id-res3-0;52'
    mask |= all_players['id'] == 'tournament_16;mcts100;cpuct2;id-res3-0;44'
    # sub_players = all_players.loc[mask1]
    mask |= all_players['id'] == 'tournament_14;mcts100;cpuct2;id-res9-0;42'
    mask |= all_players['id'] == 'tournament_13;mcts100;cpuct2;id-res3-0;94'
    mask |= all_players['id'] == 'tournament_12;mcts100;cpuct2;id-res3-0;91'
    mask |= all_players['id'] == 'tournament_8;mcts100;cpuct2e+00;id-res3-1;47'
    mask |= all_players['id'] == 'tournament_8;mcts100;cpuct2e+00;id-res3-1;71'
    participants = all_players.loc[mask]

    # participants = pd.concat([sub_players,new_players],axis=0)
    participants = participants.reset_index(drop=True)

    return participants

def get_player(game,one_info,**kwargs):
    '''
    one_info: a row of all_players/participants, as pd.series 
    '''

    # add human info (if other_type == "human")
    if (one_info.other_type == 'human').any():
        player = HumanBeckPlayer(game)
        # print('hello human!!!!!!!!!!!!')

        human = lambda x: player.play(x)
        return human, None, None
    else:
	    n_available_actions = game.getActionSize()
	    if (one_info.value_func_type  == 'nn').any() or (one_info.tree_type=='mcts').any(): # both situation need the nn
	        if (one_info.value_func_type).any()  == 'nn':
	            load_folder_file = one_info.value_func_location
	        else:
	            load_folder_file = one_info.mcts_location
	        # folder='/'.join(load_folder_file.split('/')[:-1])
	        # file=load_folder_file.split('/')[-1]
	        root_folder = '../final_agents/'

	        sub_folder='/'.join(np.array(load_folder_file.str.split('/'))[0][5:-1])
	        folder = os.path.join(root_folder, sub_folder)
	        file=np.array(load_folder_file.str.split('/'))[0][-1]
	        if one_info.n_res is None:
	        	# print(one_info.n_res,'nres is none')
	            nnet = NNetWrapper(game)
	        else: # n_res not None
	            args = pickle.load(open(os.path.join(folder,'args.p'),'rb'))
	            if (one_info.color).any(): 
	            	# print(one_info.color, 'color is true')
	                args['track_color']=True
	                othello_resnet = OthelloNNet_resnet(game,args,return_compiled=True)
	                nnet = NNetWrapper_color(game,args=args,nnet=othello_resnet) 
	            else:
	                othello_resnet = OthelloNNet_resnet(game,args,return_compiled=True)
	                nnet = NNetWrapper(game,args=args,nnet=othello_resnet)
	                
	        nnet.load_checkpoint(folder, file)
	    if (one_info.value_func_type=='nn').any():
	        val_func = nnet
	    elif (one_info.value_func_type =='cog').any(): #NB need to be changed; player_info should have the w and C
	        w = [0.01,0.2,0.05,2,100]
	        C = 0.1
	        args = [w,C]
	        if (one_info.tree_type =='mcts').any():
	            cvnnet = cvn.NNetWrapper(game,nnet,args)
	        else:
	            cvnnet = cvn.NNetWrapper(game,None,args)
	        val_func = cvnnet

	    if (one_info.tree_type =='mcts').any():
	        n_mcts = one_info.n_mcts
	        n_mcts = np.array(n_mcts)[0]
	        cpuct = one_info.cpuct
	        cpuct = np.array(cpuct)[0]

	        args = dotdict({
	        'numMCTSSims': n_mcts,
	        'cpuct': cpuct,
	        })
	        if (one_info.color).any():
	            tree = MCTS_color(game,val_func,args)
	        else:
	            tree = MCTS(game,val_func,args)


	    elif (one_info.tree_type =='bfts').any():
	        n_bfts = one_info.n_bfts
	        prune_thresh = one_info.pruning_thresh
	        args = dotdict({'numBFSsims':n_bfts,'PruningThresh':prune_thresh})
	        tree = BFTS(game,val_func,args)
	    
	    else:
	        print('')
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

	    # else:
	    # ai = lambda x: np.argmax(tree.getActionProb(x, temp=temp))
	    # return ai, val_func, tree

	    # if (one_info.color).any():
	    #     ai = lambda x: np.argmax(tree.getActionProb(x, color=one_info.color,temp=temp))
	    #     return ai, val_func, tree
	    # else:
	    # 	ai = ai_func
	    # 	return ai, val_func, tree
	    if det:
	        # ai = lambda x: np.argmax(tree.getActionProb(x, temp=temp))
	        # ai = lambda *args: np.argmax(tree.getActionProb(*args, temp=temp))
	        ai = ai_func

	    else:
	        ai = lambda *args: np.random.choice(np.arange(n_available_actions),p=tree.getActionProb(*args, temp=temp))

	    return ai, val_func, tree

	        # ai = lambda *args: np.random.choice(np.arange(n_available_actions),p=tree.getActionProb(*args, temp=temp))



def play_game(game, participant_info_1, participant_info_2, moves_dir = moves_dir,initboard=False,game_num=None):
    '''
    [sz modified] no save moves yet
    '''
    print('\n')
    #['SZ'] hiding ai info when human vs ai

    if (participant_info_1.other_type=='human').any() or (participant_info_2.other_type=='human').any():
        show_game = True
        print('Game beginning!')
    else:
        show_game = False
        print(f'Game beginning! {participant_info_1.id} v/s {participant_info_2.id}...')
    player1, val_func1, tree_1 = get_player(game, participant_info_1)
    player2,val_func2,tree_2 = get_player(game, participant_info_2)
    # player1,_,_ = get_player(game, participant_info_1)
    # player2,_,_ = get_player(game, participant_info_2)      

    display = game.display if show_game else None
    if (participant_info_1.other_type=='human').any():
    	arena = Arena(player1, player2, game, tree1=None,tree2=tree_2,track_color=[False,(participant_info_2.color).any()], display=display,game_num=game_num)
    else:
    	arena = Arena(player1, player2, game,  tree1=tree_1,tree2=None,track_color=[(participant_info_1.color).any(),False], display=display,game_num=game_num)

    #[SZ] if human game, use the old function, no saving moves
    nnet_l = [val_func1, val_func2]
    
    #return arena.playGame(verbose=show_game)
    # win_res = arena.playGameSave(verbose=show_game,nnet=nnet_l,subjectID_l=[participant_info_1.id.to_string(),participant_info_2.id.to_string()],fd=moves_dir) # if fd not exists, should be automatically created
    win_res = arena.playGame(verbose=show_game,initboard=initboard)
    return win_res
    

participants_info = get_participants()
# print('participant info',participants_info)
human_participant = create_human_player('test')
participant_iters= participants_info['id']
participant_double = pd.concat([participant_iters,participant_iters],ignore_index=True)
NEW_TOURNAMENT_DIR = os.path.join('../final_agents/results/',TOURNAMENT_NAME)

def play_human_games():
    results_name = 'vs_human'
    participant_iters.reset_index(drop=True, inplace=True)
    participant_double.reset_index(drop=True, inplace=True)
    results_df = pd.DataFrame(index=participant_iters) #let each ai play twice 

    # human_vs_iters_black = [id for x,id in participant_iters.items() if (x%2 ==1)]
    # human_vs_iters_white = [id for x,id in participant_iters.items() if (x%2 ==0)]
    human_vs_iters_black = [id for x,id in participant_iters.items() ]
    # print('human_vs_iters_white',human_vs_iters_black)
    human_vs_iters_white = [id for x,id in participant_iters.items() ]
    # change human_vs_iters_black and human_vs_iters_white to participant info
    # make it random
    shuffle(human_vs_iters_black)
    shuffle(human_vs_iters_white)

    g = Game(4,9,4)
    # for opponent in human_vs_iters_black:
    # [SZ] use progress bar?
    game_num =0
    for opponent in tqdm(human_vs_iters_black, desc="human white vs ai black"):
        game_num += 1

        participant_info_1 = participants_info.loc[participants_info['id'] == opponent]
        print('AI: ', opponent)
        # init = game.getInitBoard()

        results_df.loc[opponent, 'human_white'] = play_game(g,  participant_info_1, human_participant,initboard=None,game_num=game_num)
        ## change play_game 
        # results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))
        results_df.to_csv(os.path.join('../', results_name + '.csv')) # for other people to use, just save at the current dir
    
    game_num =8
    for opponent in tqdm(human_vs_iters_white, desc="human black vs ai white"):
        print('AI: ', opponent)
        game_num += 1
        participant_info_2 = participants_info.loc[participants_info['id'] == opponent]
        # print('participant info2', participant_info_2)
        init = g.getInitBoard()
        initRowB = np.random.randint(0, 3)  
        initColB = np.random.randint(1, 7)  
        initRowW = np.random.randint(0, 3)  
        initColW =  np.random.randint(1, 7) 
        while initRowB ==initRowW and initColB==initColW:
            initColB = np.random.randint(1, 7)
        # initRowB2 =  np.random.randint(0, 3)  
        # initColB2 =  np.random.randint(1, 7) 
        # if initRowB < 2: #50% opening with 2 pieces
        init[initRowB,initColB] =1
        init[initRowW,initColW] =-1
        # else:  #50% opening with 3 pieces
        #     init[initRowB,initColB] =1
        #     init[initRowW,initColW] =-1
        #     init[initRowB2,initColB2] =1
        results_df.loc[opponent,'human_black'] = play_game(g,  human_participant, participant_info_2,initboard=init,game_num=game_num)
        # results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))
        results_df.to_csv(os.path.join('../', results_name + '.csv')) # for other people to use, just save at the current dir
    game_num = 16
    for opponent in tqdm(human_vs_iters_black, desc="human white vs ai black"):
        print('AI: ', opponent)
        game_num += 1
        participant_info_1 = participants_info.loc[participants_info['id'] == opponent]
        # print('participant info1', participant_info_1)
        # init = game.getInitBoard()

        results_df.loc[opponent, 'human_white_2'] = play_game(g,  participant_info_1, human_participant,initboard=None,game_num=game_num)
        ## change play_game 
        # results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))
        results_df.to_csv(os.path.join('../', results_name + '.csv')) # for other people to use, just save at the current dir
    # for opponent in human_vs_iters_white:
    game_num = 24
    for opponent in tqdm(human_vs_iters_white, desc="human black vs ai white"):
        print('AI: ', opponent)
        game_num += 1
        participant_info_2 = participants_info.loc[participants_info['id'] == opponent]
        # print('participant info2', participant_info_2)
        init = g.getInitBoard()
        initRowB = np.random.randint(0, 3)  
        initColB = np.random.randint(1, 7)  
        initRowW = np.random.randint(0, 3)  
        initColW =  np.random.randint(1, 7)
        while initRowB ==initRowW and initColB==initColW:
            initColB = np.random.randint(1, 7) 
        # initRowB2 =  np.random.randint(0, 3)  
        # initColB2 =  np.random.randint(1, 7) 
        # if initRowB < 2: #50% opening with 2 pieces
        init[initRowB,initColB] =1
        init[initRowW,initColW] =-1
        # else:  #50% opening with 3 pieces
        #     init[initRowB,initColB] =1
        #     init[initRowW,initColW] =-1
        #     init[initRowB2,initColB2] =1
        results_df.loc[opponent,'human_black_2'] = play_game(g,  human_participant, participant_info_2,initboard=init,game_num=game_num)
        # results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))
        results_df.to_csv(os.path.join('../', results_name + '.csv')) # for other people to use, just save at the current dir

    print('Done!')

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