import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
tf.compat.v1.reset_default_graph()
import sys
sys.path.insert(0,'../classes')
sys.path.insert(0,'../analysis')
sys.path.insert(0,'../classes')
sys.path.insert(0,'../classes/cog_related')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=3,suppress=True)
import importlib
import logging
import numpy as np

import coloredlogs

from arena import Arena
from coach import Coach
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer,NNPolicyPlayer, NNValuePlayer
from mcts import MCTS
from utils import *
log = logging.getLogger(__name__)

from keras import backend as K

import tournament
import tournament_new

from importlib import reload
reload(tournament)

participant_iters = tournament.participant_iters

import cog_related
importlib.reload(cog_related)
from cog_related import cog_value_net as cvn
import rsatoolbox
import pickle

import random
random.seed(10)
board_num = 8138
board_idx = random.sample(range(board_num), 3000)

DATABASE_LOC='/scratch/xl1005/deep-master/tournaments/all_players_xl.pkl'
all_players = pd.read_pickle('/scratch/xl1005/deep-master/tournaments/all_players_xl.pkl')
tournament_res = pd.read_pickle('/scratch/xl1005/deep-master/tournaments/ai_all_player_repeated_round_robin_base.pkl')
g = Game(4, 9, 4)

###################
#helper function
###################
#get activation of all test boards for a particular layer
def get_layer_output_color(nnet,layer_name,feature_boards):
    outputs = []
    for board in feature_boards:
        intermediate_out = nnet.get_layer_activity(board,False,layer_name)
        outputs.append(intermediate_out[0])
    return outputs
def get_layer_output(nnet,layer_name,feature_boards):
    outputs = []
    for board in feature_boards:
        intermediate_out = nnet.get_layer_activity(board,layer_name)
        outputs.append(intermediate_out[0])
    return outputs       
#simpler way to get get all pairwise combinations and store it
import itertools
def get_pairwise_list(raw_activity_matrix,network_id,iter_num):
    pairs = itertools.permutations(raw_activity_matrix, 2)
    pairs = list(pairs)
    df_pairs = pd.DataFrame(pairs)
    path = f'/scratch/xl1005/deep-master/rsa/{network_id}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    df_pairs.to_pickle(os.path.join(path,f'pairs_{iter_num}.p'))
    return pairs
def get_pairwise_boards(boards,name,save=False):
    pairs = itertools.permutations(raw_activity_matrix, 2)
    pairs = list(pairs)
    if save:
        df_pairs = pd.DataFrame(pairs)
        path = f'/scratch/xl1005/deep-master/rsa/boards'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        df_pairs.to_pickle(os.path.join(path,f'pairs_{name}.p'))
    return pairs
#get eucledian distances
def get_euclidean_RDM(pairs):
    RDM = []
    for i in range(len(pairs)):
        dist = np.linalg.norm(pairs[i][0]-pairs[i][1])
        RDM.append(dist)
    return RDM
#compute RDM for one feature, one layer
def compute_RDM_one_layer(layer_name, feature_boards,network_id,iter_num):
    layer_output = get_layer_output(layer_name,feature_boards)
    pairs = get_pairwise_list(layer_output,network_id,iter_num)
    RDM = get_euclidean_RDM(pairs)
    #df_RDM = pd.DataFrame(RDM)
    #path = f'/scratch/xl1005/deep-master/rsa/{network_id}'
    #df_RDM.to_pickle(os.path.join(path,f'RDM_{iter_num}_{layer_name}.p'))
    return RDM
#compute RDM for one feature, boards
def compute_RDM_boards(boards,network_id,iter_num):
    #boards =  feature count vectors for boards
    #iter_num is the feature name
    network_id = 'boards'
    pairs = get_pairwise_list(boards,network_id,iter_num)
    RDM = get_euclidean_RDM(pairs)
    df_RDM = pd.DataFrame(RDM)
    path = f'/scratch/xl1005/deep-master/rsa/{network_id}'
    df_RDM.to_pickle(os.path.join(path,f'RDM_{iter_num}.p'))
    return RDM

def get_features(board):
    '''
    take a series from moves, turn into a series features
    '''
    # inv_d: result of cog_value_net.get_inv_dist_to_center(game)
    g = Game(4, 9, 4)
    inv_d = cvn.get_inv_dist_to_center(g)
    feat,header = cvn.get_all_feat(board,inv_d)
    all_feat=pd.DataFrame(list(feat),columns=header)

    return all_feat


##########################################
#boards


def get_all_test_boards():
	#get all test boards
	hqfd = '/scratch/xl1005/Analysis notebooks/new/Heuristic quality'
	move_stats_hvh = np.loadtxt(os.path.join(hqfd,'move_stats_hvh.txt'),dtype=int)
	feature_counts = np.loadtxt(os.path.join(hqfd,'optimal_feature_vals.txt'))[:,-35:]
	optimal_move_values = np.loadtxt(os.path.join(hqfd,'opt_hvh.txt'))[:,-36:]
	optimal_move_values_board = np.loadtxt(os.path.join(hqfd,'opt_hvh.txt'),dtype='object')[:,:2]
	string_to_int_array = lambda x: (np.array(list(x[0]),dtype=int) -np.array(list(x[1]),dtype=int)).reshape(4,9)
	optimal_move_values_board_reshaped_int = np.apply_along_axis(string_to_int_array,1,optimal_move_values_board)
	return optimal_move_values_board_reshaped_int
def get_all_features():
	#get feature counts of all test boards
	optimal_move_values_board_reshaped_int = get_all_test_boards()
	all_feature = get_features(optimal_move_values_board_reshaped_int)
	return all_feature
optimal_move_values_board_reshaped_int = get_all_test_boards()
# boards = optimal_move_values_board_reshaped_int
path_boards = f'/scratch/xl1005/deep-master/rsa/boards/'
filename_boards = os.path.join(path_boards,f'all_test_boards.p')
with open(filename_boards, 'rb') as fp:
    boards = pickle.load(fp)

# all_feature = get_all_features()
# all_feature = pd.read_pickle(os.path.join(path_boards,f'df_all_boards_features_extra_8138.pkl'))
all_feature = pd.read_pickle(os.path.join(path_boards,f'df_all_boards_features.p'))

all_boards_features = all_feature.values.tolist()
all_boards_features = [np.array(item) for item in all_boards_features]

#########################################
#networks

def get_networks(model_line):
	mask = all_players.model_line == model_line
	networks = all_players.loc[mask]
	return networks

def store_activity_df(nnet,layer_name,boards,network_id,iter_num,sampled=False,store=False):
	#get output of a particular iter and particular layer
    layer_activity = get_layer_output(nnet,layer_name,boards)
    data_dense1_activity = np.array(layer_activity)
    df_layer_activity= pd.DataFrame(layer_activity)
    path_activity = f'/scratch/xl1005/deep-master/rsa/{network_id}'
    if store == True:
        if sampled:
            df_layer_activity.to_pickle(os.path.join(path_activity,f'df_activity_{iter_num}_{layer_name}_sampled.p'))
        else:
            df_layer_activity.to_pickle(os.path.join(path_activity,f'df_activity_{iter_num}_{layer_name}.p'))
    return df_layer_activity

def get_layer_outputs(model_line,save=True):
	#######
	#outputs a dict of layer activity dataframes for all iterations of a model line
	#######
	networks = get_networks(model_line)
	activity_dict = {}
	for row in range(len(networks)):
		one_info = networks.iloc[row]
		one_info.swap = None
		one_info.fix_depth = None
		player, nnet, tree = tournament_new.get_player(g,one_info)
		names = nnet.get_layer_name()
		layer_name = names[-3] #change layer name here
		iter_num = one_info.value_func_iter
		df_layer_activity = store_activity_df(nnet,layer_name,boards,model_line,iter_num)  
		#store in dictionary for all iters
		activity_dict[str(iter_num)] =  df_layer_activity
		print('activity_dict:',str(iter_num))
	if save==True:
		path = f'/scratch/xl1005/deep-master/rsa/{model_line}'
		isExist = os.path.exists(path)
		if not isExist:
			os.makedirs(path)
		filename_activity_dict = os.path.join(path,f'activity_dict_{layer_name}.pkl')
		with open(filename_activity_dict, 'wb') as handle:
			pickle.dump(activity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)	
	return activity_dict,layer_name

def preprocess_boards():
	path_boards = '/scratch/xl1005/deep-master/rsa/boards'
	df_all_boards_features = pd.read_pickle(os.path.join(path_boards,'df_all_boards_features_extra_8138.pkl'))
	#remove 4 because it's all zero
	df_all_boards_features = df_all_boards_features.iloc[:,0:4]
	list_all_boards_features = df_all_boards_features.values.tolist()
	data_all_boards_features = [np.array(item) for item in list_all_boards_features]
	data_all_boards_features = np.array(data_all_boards_features)
	data_boards = rsatoolbox.data.Dataset(data_all_boards_features[board_idx])

	#data_boards = rsatoolbox.data.Dataset(data_all_boards_features[0:3000])
	return data_boards

def preprocess_activity_dfs(model_line):
	#convert the dict of dataframes to rsa datasets
	#######
	#outputs a dict of rsa-formatted datasets of layer activity for all iterations of a model line
	#######
	activity_dict,layer_name = get_layer_outputs(model_line)
	data_activity_dict = {}
	for key in activity_dict:
		df_layer_activity = activity_dict[key]
		list_layer_activity = df_layer_activity.values.tolist()
		data_layer_activity = [np.array(item) for item in list_layer_activity]
		data_layer_activity = np.array(data_layer_activity)
		data_activity = rsatoolbox.data.Dataset(data_layer_activity[board_idx])

		# data_activity = rsatoolbox.data.Dataset(data_layer_activity[0:3000])
		data_activity_dict[key] = data_activity
	path = f'/scratch/xl1005/deep-master/rsa/{model_line}'
	# filename_data_activity_dict = os.path.join(path,f'data_activity_dict_{layer_name}.pkl')
	# data_activity_dict.save(filename_data_activity_dict, file_type='pkl', overwrite=True)	
	return data_activity_dict,layer_name

def get_rdm(data):
    rdm = rsatoolbox.rdm.calc_rdm(data, method='euclidean', descriptor=None, noise=None)
    return rdm
def get_forward_pass(model_line,save=True):
	#get forward pass of all layer actitivity for each agent, each key in the dict is an agent, 
	#each value is a nested list, activities of all boards, then all layer
	networks = get_networks(model_line)
	forward_dict = {}

	for row in range(len(networks)-1,0,-2):
		print('network:',row)
		path = f'/scratch/xl1005/deep-master/rsa/{model_line}'
		one_info = networks.iloc[row]
		one_info.swap = None
		one_info.fix_depth = None
		player, nnet, tree = tournament_new.get_player(g,one_info)
		names = nnet.get_layer_name()
		layer_name = names[-4]
		#get outputs for all boards
		outputs = []
		for board in boards:
			all_layers_out = nnet.get_all_activity_no_dropout(board)
			outputs.append(all_layers_out)
		if save == True:
			isExist = os.path.exists(path)
			if not isExist:
				os.makedirs(path)
			filename_outputs = os.path.join(path,f'forward_outputs_{row}.pkl')
			with open(filename_outputs,'wb') as f:
				pickle.dump(outputs,f)		
		iter_num = one_info.value_func_iter
		# df_layer_activity = store_activity_df(nnet,layer_name,boards,model_line,iter_num)  
		#store in dictionary for all iters
		forward_dict[str(iter_num)] =  outputs
		print('forward_dict:',str(iter_num))

	if save == True:
		isExist = os.path.exists(path)
		if not isExist:
			os.makedirs(path)
		filename_outputs = os.path.join(path,f'forward_outputs_dict.pkl')
		with open(filename_outputs,'wb') as f:
			pickle.dump(forward_dict,f)
	return forward_dict

def get_forward_pass_2(model_line,save=True):

	networks = get_networks(model_line)
	activity_dict = {}
	for row in range(len(networks)-1,0,-2):
		print('iter_num:',row)
		path = f'/scratch/xl1005/deep-master/rsa/{model_line}'
		one_info = networks.iloc[row]
		one_info.swap = None
		one_info.fix_depth = None
		player, nnet, tree = tournament_new.get_player(g,one_info)	
		all_outs = nnet.get_all_activity_no_dropout_batch(boards,color=True) #add color=True if agents has color hp
		hm = {} #hm with key = layer_num, value = activity for all test boards; all layer, all activity for one agent/iter
		for layer in range(len(all_outs[0])):
			# print('layer_num:',layer)
			all_boards_layer = []
			for board in all_outs:
				all_boards_layer.append(board[layer]) 
			hm[layer] = all_boards_layer
		# activity_dict[row]=hm

		if save==True:
			path = f'/scratch/xl1005/deep-master/rsa/{model_line}/forward_pass_2'
			isExist = os.path.exists(path)
			if not isExist:
				os.makedirs(path)
			filename_hm = os.path.join(path,f'activity_dict_{row}.pkl')
			with open(filename_hm, 'wb') as handle:
				pickle.dump(hm, handle, protocol=pickle.HIGHEST_PROTOCOL)	
	return hm

def compute_rsa(model_line):
	data_activity_dict,layer_name = preprocess_activity_dfs(model_line)
	data_all_boards_features = preprocess_boards()
	rdm_board = get_rdm(data_all_boards_features)
	rdm_board.pattern_descriptors =  {'index':[*range(0, 3000, 1)]}
	models = []
	for key in data_activity_dict:
		print('key:',key)
		data_activity = data_activity_dict[key]
		rdm_activity = get_rdm(data_activity)
		model_activity = rsatoolbox.model.ModelFixed(f'{layer_name}_{key}', rdm_activity)
		models.append(model_activity)

	path = f'/scratch/xl1005/deep-master/rsa/{model_line}'
	isExist = os.path.exists(path)
	if not isExist:
		os.makedirs(path)

	results =  rsatoolbox.inference.eval_bootstrap_pattern(models,rdm_board, method='corr_cov')
	#results =  rsatoolbox.inference.eval_bootstrap_pattern(models,rdm_board, method='corr_cov')	
	filename_corr = os.path.join(path,f'results_{layer_name}_corr_cov.pkl')
	results.save(filename_corr, file_type='pkl', overwrite=True)

	filename_activity_dict = os.path.join(path,f'data_activity_dict_{layer_name}.pkl')
	with open(filename_activity_dict,'wb') as f:
		pickle.dump(data_activity_dict,f)	
	# filename_cka=os.path.join(path,f'results_{layer_name}_cka.pkl')
	# results.save(filename_cka, file_type='pkl', overwrite=True)
	return results

#model_line = 'tournament_8;mcts100;cpuct5e-01;id-res3-0' #pick a model line, remember to check if color exist
#model_line = 'tournament_13;mcts100;cpuct2;id-res3-0' #,color
#model_line = 'tournament_8;mcts100;cpuct2;id-res3-0' #pick a model line, remember to check if color exist

model_line = 'tournament_16;mcts100;cpuct2;id-res3-0'
#model_line = 'tournament_15;mcts100;cpuct2;id-res3-0'

#names = nnet.get_layer_name()
#layer_name = names[-4] #-4 is flatten2 (p), -3 is 'dense_1'(v), -5 is 'flatten_3'(v), -10 is conv2d_40 (p),-11 is 'conv2d_41'(v)
print('model_line:',model_line)
if __name__ == "__main__":
    main_args = sys.argv[1:]
    # print(main_args)
    get_forward_pass_2(model_line)
    #compute_rsa(model_line)