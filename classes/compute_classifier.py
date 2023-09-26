import tensorflow as tf
tf.compat.v1.reset_default_graph()
import sys
sys.path.insert(0,'../classes')
sys.path.insert(0,'../analysis')
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
from compute_RSA import *
log = logging.getLogger(__name__)

from keras import backend as K

import tournament
import tournament_new

from importlib import reload
reload(tournament)

participant_iters = tournament.participant_iters

g = Game(4, 9, 4)

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns

#########################################################
#helper
def squeeze_activity(activity_dict_layer):
	data_activity_dict = {}
	for key in activity_dict_layer:
		list_layer_activity = activity_dict_layer[key]
		if list_layer_activity[0].ndim ==1:         
			data_layer_activity = [item for item in list_layer_activity]
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
		data_activity_dict[str(key)] = data_layer_activity
	return data_activity_dict

def get_labels(df_boards_features_3):
    positive_idx_3 = np.where(df_boards_features_3>0)[0]
    positive_3_num = len(positive_idx_3)
    positive_idx_3_list = positive_idx_3.tolist()
    negative_idx_3 = [i for i in range(0,8138) if i not in positive_idx_3_list]
    if positive_3_num <= len(negative_idx_3):
        negative_idx_3_balanced = np.array(random.sample(negative_idx_3,positive_3_num))
        positive_idx_3_balanced = positive_idx_3
        labels_3 = np.array([0] * positive_3_num*2)
    else:
        negative_idx_3_balanced = negative_idx_3
        positive_idx_3_balanced = np.array(random.sample(positive_idx_3_list,len(negative_idx_3_balanced)))
        labels_3 = np.array([0] * len(negative_idx_3_balanced)*2)
    half_idx_3 = int(len(labels_3)/2)
    labels_3[0:half_idx_3]=1
    return labels_3,positive_idx_3_balanced,negative_idx_3_balanced

def get_balanced_activity(positive_idx_3,negative_idx_3_balanced,df_activity_dict):
    positive_3_activity = df_activity_dict.iloc[positive_idx_3, :]
    negative_3_activity = df_activity_dict.iloc[negative_idx_3_balanced, :]
    balanced_activity = positive_3_activity.append(negative_3_activity)
    return balanced_activity

def reshape_activity(X):
    reshaped=[]
    for l in X:
        reshaped.append(l)

    X_new=np.array(reshaped)
    return X_new

#classifier
def train_classifier(agent_iter,feature_label,df_activity_dict_layer):
    X =df_activity_dict_layer[str(agent_iter)]
    X = reshape_activity(X)
    y =feature_label
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
    if X_train.ndim == 1:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    clf = MLPClassifier(
                        solver='lbfgs', 
                        alpha=1e-4,hidden_layer_sizes=(12,2),max_iter = 1000,
                        random_state=1).fit(X_train, y_train)
#     predict_proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    return score
 def get_classifier_score(last_agent,labels_3,activity_list):
	score_list_3_v = []
	score_list_3_p = []
	score_list_3_fp = []
	score_list_3_fv = []
	score_list_3_cv = []
	score_list_3_cp = []
	score_list_3_relu1 = []
	score_list_3_relu3 = []
	score_list_3_relu5 = []
	score_list_3_dense1 = []
	score_list_3_input = []

	for agent_num in range(1,last_agent,2):
		score_3_v = train_classifier(agent_num,labels_3,activity_list[0])
		score_list_3_v.append(score_3_v)
		score_3_p = train_classifier(agent_num,labels_3,activity_list[1])
		score_list_3_p.append(score_3_p)   
		score_3_fp = train_classifier(agent_num,labels_3,activity_list[2])
		score_list_3_fp.append(score_3_fp) 
		score_3_fv = train_classifier(agent_num,labels_3,activity_list[3])
		score_list_3_fv.append(score_3_fv)   
		score_3_cv = train_classifier(agent_num,labels_3,activity_list[4])
		score_list_3_cv.append(score_3_cv)   
		score_3_cp = train_classifier(agent_num,labels_3,activity_list[5])
		score_list_3_cp.append(score_3_cp)   
		score_3_relu1 = train_classifier(agent_num,labels_3,activity_list[6])
		score_list_3_relu1.append(score_3_relu1)  
		score_3_relu3 = train_classifier(agent_num,labels_3,activity_list[7])
		score_list_3_relu3.append(score_3_relu3)  
		score_3_relu5 = train_classifier(agent_num,labels_3,activity_list[8])
		score_list_3_relu5.append(score_3_relu5) 
		score_3_dense1 = train_classifier(agent_num,labels_3,activity_list[9])
		score_list_3_dense1.append(score_3_dense1)  
		score_3_input = train_classifier(agent_num,labels_3,activity_list[10])
		score_list_3_input.append(score_3_input)  
	df_score = pd.DataFrame()
	df_score['v']=score_list_3_v
	df_score['dense1']=score_list_3_dense1
	df_score['flattenv']=score_list_3_fv 
	df_score['convv']=score_list_3_cv
	df_score['p']=score_list_3_p 
	df_score['flattenp']=score_list_3_fp
	df_score['convp']=score_list_3_cp
	df_score['relu5']=score_list_3_relu5 
	df_score['relu3']=score_list_3_relu3
	df_score['relu1']=score_list_3_relu1
	df_score['input']=score_list_3_input

	return df_score  
#########################################################

#get boards
path_boards = f'/scratch/xl1005/deep-master/rsa/boards'
df_all_boards_features = pd.read_pickle(os.path.join(path_boards,f'df_all_boards_features_extra_8138.pkl'))
df_boards_features_central = df_all_boards_features.iloc[:,[0]]
df_boards_features_2con = df_all_boards_features.iloc[:,[1]]
df_boards_features_2uncon = df_all_boards_features.iloc[:,[2]]
df_boards_features_3 = df_all_boards_features.iloc[:,[3]]
df_boards_features_tri = df_all_boards_features.iloc[:,[5]]
df_boards_features_dt = df_all_boards_features.iloc[:,[6]]

list_all_boards_features = df_all_boards_features.values.tolist()
list_boards_features_central = df_boards_features_central.values.tolist()
list_boards_features_2con = df_boards_features_2con.values.tolist()
list_boards_features_2uncon = df_boards_features_2uncon.values.tolist()
list_boards_features_3 = df_boards_features_3.values.tolist()
list_boards_features_tri = df_boards_features_tri.values.tolist()
list_boards_features_dt = df_boards_features_dt.values.tolist()
#rdm for feature 3inarow
data_boards_features_3 = [np.array(item) for item in list_boards_features_3]
data_boards_features_3 = np.array(data_boards_features_3)
data_boards_3 = rsatoolbox.data.Dataset(data_boards_features_3[:])
rdm_board_3 = get_rdm(data_boards_3)
rdm_board_3.pattern_descriptors =  {'index':[*range(0, 3000, 1)]}
#rdm for all features
data_all_boards_features = [np.array(item) for item in list_all_boards_features]
data_all_boards_features = np.array(data_all_boards_features)
data_boards = rsatoolbox.data.Dataset(data_all_boards_features[:])
rdm_board = get_rdm(data_boards)
rdm_board.pattern_descriptors =  {'index':[*range(0, 3000, 1)]}

##############################################################
#retrieve dfs
# model_line = 'tournament_13;mcts100;cpuct2;id-res3-0'
model_line = 'tournament_8;mcts100;cpuct2;id-res3-0'
last_agent = 29
path = f'/scratch/xl1005/deep-master/rsa/{model_line}/activity_layer/df'

df_activity_dict_dense1=pd.read_pickle(os.path.join(path,f'df_activity_dict_dense1.pkl'))
df_activity_dict_convp=pd.read_pickle(os.path.join(path,f'df_activity_dict_convp.pkl'))
df_activity_dict_convv=pd.read_pickle(os.path.join(path,f'df_activity_dict_convv.pkl'))
df_activity_dict_relu1=pd.read_pickle(os.path.join(path,f'df_activity_dict_relu1.pkl'))
df_activity_dict_relu3=pd.read_pickle(os.path.join(path,f'df_activity_dict_relu3.pkl'))
df_activity_dict_relu5=pd.read_pickle(os.path.join(path,f'df_activity_dict_relu5.pkl'))
df_activity_dict_flattenp=pd.read_pickle(os.path.join(path,f'df_activity_dict_flattenp.pkl'))
df_activity_dict_flattenv=pd.read_pickle(os.path.join(path,f'df_activity_dict_flattenv.pkl'))
df_activity_dict_v=pd.read_pickle(os.path.join(path,f'df_activity_dict_v.pkl'))
df_activity_dict_p=pd.read_pickle(os.path.join(path,f'df_activity_dict_p.pkl'))
df_activity_dict_input=pd.read_pickle(os.path.join(path,f'df_activity_dict_input.pkl'))
#get labels and feature present/absent index
labels_2con,positive_idx_2con,negative_idx_2con = get_labels(df_boards_features_2con)
labels_2uncon,positive_idx_2uncon,negative_idx_2uncon = get_labels(df_boards_features_2uncon)
labels_tri,positive_idx_tri,negative_idx_tri = get_labels(df_boards_features_tri)
labels_dt,positive_idx_dt,negative_idx_dt = get_labels(df_boards_features_dt)
labels_3,positive_idx_3,negative_idx_3 = get_labels(df_boards_features_3)

labels_list = [labels_3,labels_dt,labels_tri,labels_2con,labels_2uncon]
positive_idx_list  = [positive_idx_3,positive_idx_dt,positive_idx_tri,positive_idx_2con,positive_idx_2uncon]
negative_idx_list  = [negative_idx_3,negative_idx_dt,negative_idx_tri,negative_idx_2con,negative_idx_2uncon]
feature_name_list = [ '3','dt','tri','2con','2uncon']

import itertools
for positive_idx, negative_idx,labels,feature_name in itertools.izip(positive_idx_list, negative_idx_list,labels_list,feature_name_list):
	print(feature_name)
	balanced_activity_v = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_v)
	balanced_activity_p = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_p)
	balanced_activity_flattenv = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_flattenv)
	balanced_activity_flattenp = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_flattenp)
	balanced_activity_convv = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_convv)
	balanced_activity_convp = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_convp)
	balanced_activity_relu5 = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_relu5)
	balanced_activity_relu3 = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_relu3)
	balanced_activity_relu1 = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_relu1)
	balanced_activity_dense1 = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_dense1)
	balanced_activity_input = get_balanced_activity(positive_idx,negative_idx,df_activity_dict_input)

	activity_list = [balanced_activity_v,balanced_activity_p,balanced_activity_flattenp,balanced_activity_flattenv,
                balanced_activity_convv,balanced_activity_convp,balanced_activity_relu1,balanced_activity_relu3,
                balanced_activity_relu5,balanced_activity_dense1]
	df_score = get_classifier_score(last_agent,labels,activity_list)
	path = f'/scratch/xl1005/deep-master/rsa/{model_line}/results/classifier'
	isExist = os.path.exists(path)
	if not isExist:
		os.makedirs(path)
	df_score.to_pickle(os.path.join(path,f'df_score_{feature_name}.pkl'))

