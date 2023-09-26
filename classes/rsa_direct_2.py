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
# board_num = 2500
# board_idx = random.sample(range(5482), board_num)

DATABASE_LOC='/scratch/xl1005/deep-master/tournaments/all_players_xl.pkl'
all_players = pd.read_pickle('/scratch/xl1005/deep-master/tournaments/all_players_xl.pkl')
tournament_res = pd.read_pickle('/scratch/xl1005/deep-master/tournaments/ai_all_player_repeated_round_robin_base.pkl')
g = Game(4, 9, 4)

###################
#helper function
###################
def preprocess_dfs(df_all_boards_features,df_layer_activity):
    list_all_boards_features = df_all_boards_features.values.tolist()
    data_all_boards_features = [np.array(item) for item in list_all_boards_features]
    list_layer_activity = df_layer_activity.values.tolist()
    data_layer_activity = [np.array(item) for item in list_layer_activity]

    data_all_boards_features = np.array(data_all_boards_features)
    data_layer_activity = np.array(data_layer_activity)
    
    data_boards = rsatoolbox.data.Dataset(data_all_boards_features[board_idx])
    data_activity = rsatoolbox.data.Dataset(data_layer_activity[board_idx])
    return data_boards,data_activity
def get_rdm(data):
    rdm = rsatoolbox.rdm.calc_rdm(data, method='euclidean', descriptor=None, noise=None)
    #rdm_flatten3 = rsatoolbox.rdm.calc_rdm(data_activity, method='euclidean', descriptor=None, noise=None)
    return rdm
def get_activity_models(activity_dict_layer,layer_name):
    data_activity_dict = {}
    for key in activity_dict_layer:
        list_layer_activity = activity_dict_layer[key]
    #     list_layer_activity = df_layer_activity.values.tolist()
        if list_layer_activity[0].ndim ==1:         
            data_layer_activity = [item for item in list_layer_activity]
            data_layer_activity = np.array(data_layer_activity)
        elif list_layer_activity[0].ndim ==2:
            d1=list_layer_activity[0].shape[0]
            d2=list_layer_activity[0].shape[1]
            new_length = d1*d2
            data_layer_activity = [np.reshape(np.squeeze(x),new_length) for x in list_layer_activity]
        else:
            d1=list_layer_activity[0].shape[0]
            d2=list_layer_activity[0].shape[1]
            d3=list_layer_activity[0].shape[2]
            new_length = d1*d2*d3
            data_layer_activity = [np.reshape(np.squeeze(x),new_length) for x in list_layer_activity]
        data_layer_activity = np.array(data_layer_activity)
        data_activity = rsatoolbox.data.Dataset(data_layer_activity[board_idx])
        data_activity_dict[key] = data_activity
    models = []
    for key in data_activity_dict:
        print(key)
        data_activity = data_activity_dict[key]
        rdm_activity = get_rdm(data_activity)
        # print(rdm_activity.dissimilarities)

        model_activity = rsatoolbox.model.ModelFixed(f'{layer_name}_{key}', rdm_activity)
        models.append(model_activity)
    return models
def get_activity_dict_per_layer(model_line,layer_num,last_agent_iter):
    activity_dict={}
    for agent_iter in range(last_agent_iter,0,-2):
        filename=f'/Users/daisy/Desktop/projects/DeepRL-master/rsa/{model_line}/activity_dict_{agent_iter}.pkl'
        with open(filename, 'rb') as fp:
            activity_dict_val = pickle.load(fp)
        val_list = activity_dict_val[layer_num]
        dict_val = [item[0][0] for item in val_list]
        activity_dict[agent_iter] = dict_val
    return activity_dict
 ####################
 #process boards
 #####################
import pandas as pd
#randonly sample 3000 boards
path_boards = f'/scratch/xl1005/deep-master/rsa/boards'
# df_all_boards_features = pd.read_pickle(os.path.join(path_boards,f'df_all_boards_features_extra_8138.pkl'))
df_all_boards_features = pd.read_pickle(os.path.join(path_boards,f'df_all_boards_features.p'))

# df_all_boards_features = df_all_boards_features.iloc[:,0:4]
df_boards_features_central = df_all_boards_features.iloc[:,[0]]
df_boards_features_2con = df_all_boards_features.iloc[:,[1]]
df_boards_features_2uncon = df_all_boards_features.iloc[:,[2]]
df_boards_features_3 = df_all_boards_features.iloc[:,[3]]
# df_boards_features_tri = df_all_boards_features.iloc[:,[5]]
# df_boards_features_dt = df_all_boards_features.iloc[:,[6]]

list_all_boards_features = df_all_boards_features.values.tolist()
list_boards_features_central = df_boards_features_central.values.tolist()
list_boards_features_2con = df_boards_features_2con.values.tolist()
list_boards_features_2uncon = df_boards_features_2uncon.values.tolist()
list_boards_features_3 = df_boards_features_3.values.tolist()
# list_boards_features_tri = df_boards_features_tri.values.tolist()
# list_boards_features_dt = df_boards_features_dt.values.tolist()

##################################################
#get board idx half feature present, half present, 
#feature 3inarow
#1.change feature_name, 2. change board_subset 3.rdm_board below 
#4. change board pickle file above 5.check path_activity 6.check saved path_activity where results are saved
#7. delete or add back df_all_boards_features.iloc[:,[5]] and [6], along with related lists above, and RDMS below
random.seed(3)
feature_name = '3'
board_subset = np.array(list_boards_features_3)
feature_present_idx = np.where(board_subset > 0)[0].tolist()
feature_absent_idx = np.where(board_subset == 0)[0].tolist()
if len(feature_present_idx) <= len(feature_absent_idx):
    board_idx_absent = random.sample(feature_absent_idx, len(feature_present_idx))
else:
    board_idx_absent = feature_absent_idx
    feature_present_idx = random.sample(feature_present_idx, len(board_idx_absent))
#sample size can't be too large for rsa
if len(board_idx_absent) > 1500:
    board_idx_absent = random.sample(feature_absent_idx, 1200)
    feature_present_idx = random.sample(feature_present_idx, 1200)

board_idx = board_idx_absent + feature_present_idx
# board_idx = random.sample(range(5482), 2800)
board_num = len(board_idx)
print('board_num:',board_num)
print('feature:',feature_name)
#rdm for feature 3inarow
data_boards_features_3 = [np.array(item) for item in list_boards_features_3]
data_boards_features_3 = np.array(data_boards_features_3)
data_boards_3 = rsatoolbox.data.Dataset(data_boards_features_3[board_idx])
rdm_board_3 = get_rdm(data_boards_3)
rdm_board_3.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
#rdm for feature 2con
data_boards_features_2con = [np.array(item) for item in list_boards_features_2con]
data_boards_features_2con = np.array(data_boards_features_2con)
data_boards_2con = rsatoolbox.data.Dataset(data_boards_features_2con[board_idx])
rdm_board_2con = get_rdm(data_boards_2con)
rdm_board_2con.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
#rdm for feature 2uncon
data_boards_features_2uncon = [np.array(item) for item in list_boards_features_2uncon]
data_boards_features_2uncon = np.array(data_boards_features_2uncon)
data_boards_2uncon = rsatoolbox.data.Dataset(data_boards_features_2uncon[board_idx])
rdm_board_2uncon = get_rdm(data_boards_2uncon)
rdm_board_2uncon.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
# #rdm for feature tri
# data_boards_features_tri = [np.array(item) for item in list_boards_features_tri]
# data_boards_features_tri = np.array(data_boards_features_tri)
# data_boards_tri = rsatoolbox.data.Dataset(data_boards_features_tri[board_idx])
# rdm_board_tri = get_rdm(data_boards_tri)
# rdm_board_tri.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
# #rdm for feature dt
# data_boards_features_dt = [np.array(item) for item in list_boards_features_dt]
# data_boards_features_dt = np.array(data_boards_features_dt)
# data_boards_dt = rsatoolbox.data.Dataset(data_boards_features_dt[board_idx])
# rdm_board_dt = get_rdm(data_boards_dt)
# rdm_board_dt.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
###################
rdm_board = rdm_board_3
###################
#rdm for all features
# data_all_boards_features = [np.array(item) for item in list_all_boards_features]
# data_all_boards_features = np.array(data_all_boards_features)
# data_boards = rsatoolbox.data.Dataset(data_all_boards_features[board_idx])
# # rdm_board = get_rdm(data_boards)
# rdm_board.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
# #rdms for seperate features 
# data_boards_features_central = [np.array(item) for item in list_boards_features_central]
# data_boards_features_2con = [np.array(item) for item in list_boards_features_2con]
# data_boards_features_2uncon = [np.array(item) for item in list_boards_features_2uncon]
# data_boards_features_3 = [np.array(item) for item in list_boards_features_3]

# data_boards_features_central  = np.array(data_boards_features_central)
# data_boards_features_2con  = np.array(data_boards_features_2con)
# data_boards_features_2uncon  = np.array(data_boards_features_2uncon)
# data_boards_features_3  = np.array(data_boards_features_3)

# data_boards_central = rsatoolbox.data.Dataset(data_boards_features_central[board_idx])
# data_boards_2con = rsatoolbox.data.Dataset(data_boards_features_2con[board_idx])
# data_boards_2uncon = rsatoolbox.data.Dataset(data_boards_features_2uncon[board_idx])
# data_boards_3 = rsatoolbox.data.Dataset(data_boards_features_3[board_idx])

# rdm_board_central = get_rdm(data_boards_central)
# rdm_board_2con = get_rdm(data_boards_2con)
# rdm_board_2uncon = get_rdm(data_boards_2uncon)
# rdm_board_3 = get_rdm(data_boards_3)

# rdm_board_central.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
# rdm_board_2con.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
# rdm_board_2uncon.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}
# rdm_board_3.pattern_descriptors =  {'index':[*range(0, board_num, 1)]}

###################
# model_line = 'tournament_8;mcts100;cpuct5e-01;id-res3-0' #pick a model line
# model_line = 'tournament_8;mcts100;cpuct2;id-res3-0'
model_line = 'tournament_13;mcts100;cpuct2;id-res3-0'
# model_line = 'tournament_15;mcts100;cpuct2;id-res3-0'
# model_line = 'tournament_16;mcts100;cpuct2;id-res3-0'
print(model_line)
#retrieve layers
# path_activity = f'/scratch/xl1005/deep-master/rsa/{model_line}/activity_layer/extra'
path_activity = f'/scratch/xl1005/deep-master/rsa/{model_line}'

names = ['v','p','flattenp','flattenv','relu5','relu3','relu1','convp','convv','dense1','input']
filename_activity_layer = os.path.join(path_activity,f'activity_dict_v.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_v = pickle.load(fp)
    fp.close()
filename_activity_layer = os.path.join(path_activity,f'activity_dict_p.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_p = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_flattenp.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_flattenp = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_flattenv.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_flattenv = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_relu5.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_relu5 = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_relu3.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_relu3 = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_relu1.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_relu1 = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_convv.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_convv = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_convp.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_convp = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_dense1.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_dense1 = pickle.load(fp)
    fp.close()

filename_activity_layer = os.path.join(path_activity,f'activity_dict_input.pkl')
with open(filename_activity_layer, 'rb') as fp:
    activity_dict_input = pickle.load(fp)
    fp.close()

models_v = get_activity_models(activity_dict_v,'v')
models_p = get_activity_models(activity_dict_p,'p')
models_flattenp = get_activity_models(activity_dict_flattenp,'flatten_p')
models_flattenv = get_activity_models(activity_dict_flattenv,'flatten_v')
models_relu5 = get_activity_models(activity_dict_relu5,'relu5')
models_relu3 = get_activity_models(activity_dict_relu3,'relu3')
models_relu1 = get_activity_models(activity_dict_relu1,'relu1')
models_convv = get_activity_models(activity_dict_convv,'convv')
models_convp = get_activity_models(activity_dict_convp,'convp')
models_dense1 = get_activity_models(activity_dict_dense1,'dense1')
models_input = get_activity_models(activity_dict_input,'input')

def get_rsa_results():
    # results_v_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_v,rdm_board, method='corr_cov',N=2)
    # results_p_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_p,rdm_board, method='corr_cov',N=2)
    # results_flatten_p_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_flattenp,rdm_board, method='corr_cov',N=2)
    # results_flatten_v_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_flattenv,rdm_board, method='corr_cov',N=2)
    # results_dense1_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_dense1,rdm_board, method='corr_cov',N=2)
    # results_input_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_input,rdm_board, method='corr_cov',N=2)
    # results_convp_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_convp,rdm_board, method='corr_cov',N=2)
    # results_convv_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_convv,rdm_board, method='corr_cov',N=2)
    # results_relu5_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_relu5,rdm_board, method='corr_cov',N=2)
    # results_relu3_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_relu3,rdm_board, method='corr_cov',N=2)
    # results_relu1_3 =  rsatoolbox.inference.eval_bootstrap_pattern(models_relu1,rdm_board, method='corr_cov',N=2)
    results_v_3 =  rsatoolbox.inference.eval_fixed(models_v,rdm_board, method='corr_cov')
    results_p_3 =  rsatoolbox.inference.eval_fixed(models_p,rdm_board, method='corr_cov')
    results_flatten_p_3 =  rsatoolbox.inference.eval_fixed(models_flattenp,rdm_board, method='corr_cov')
    results_flatten_v_3 =  rsatoolbox.inference.eval_fixed(models_flattenv,rdm_board, method='corr_cov')
    results_dense1_3 =  rsatoolbox.inference.eval_fixed(models_dense1,rdm_board, method='corr_cov')
    results_input_3 =  rsatoolbox.inference.eval_fixed(models_input,rdm_board, method='corr_cov')
    results_convp_3 =  rsatoolbox.inference.eval_fixed(models_convp,rdm_board, method='corr_cov')
    results_convv_3 =  rsatoolbox.inference.eval_fixed(models_convv,rdm_board, method='corr_cov')
    results_relu5_3 =  rsatoolbox.inference.eval_fixed(models_relu5,rdm_board, method='corr_cov')
    results_relu3_3 =  rsatoolbox.inference.eval_fixed(models_relu3,rdm_board, method='corr_cov')
    results_relu1_3 =  rsatoolbox.inference.eval_fixed(models_relu1,rdm_board, method='corr_cov')
    results = [results_v_3,results_p_3,results_flatten_p_3,results_flatten_v_3,results_relu5_3,results_relu3_3,results_relu1_3,results_convp_3,results_convv_3,results_dense1_3,results_input_3]
    names = ['v','p','flattenp','flattenv','relu5','relu3','relu1','convp','convv','dense1','input']
    feature_num = '3'
    path_activity = f'/scratch/xl1005/deep-master/rsa/{model_line}/results/{feature_name}/fixed/5482/seed3'
    isExist = os.path.exists(path_activity)
    if not isExist:
        os.makedirs(path_activity)
    count = 0
    for layer_name in names:
        filename_result = os.path.join(path_activity,f'result_{layer_name}_{feature_num}_cka.pkl')
        with open(filename_result, 'wb') as handle:
            pickle.dump(results[count], handle, protocol=pickle.HIGHEST_PROTOCOL)
        count+=1
        print(count)
#names = nnet.get_layer_name()
#layer_name = names[-4] #-4 is flatten2 (p), -3 is 'dense_1'(v), -5 is 'flatten_3'(v), -10 is conv2d_40 (p),-11 is 'conv2d_41'(v)

if __name__ == "__main__":
    main_args = sys.argv[1:]
    # print(main_args)
    get_rsa_results()
    #compute_rsa(model_line)