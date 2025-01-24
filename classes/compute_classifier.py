"""
Neural Network Activity Analysis for Board Game Features

This module analyzes neural network layer activities in relation to board game features,
using classifiers to measure feature representation across different network layers.
It processes board states and their features, trains classifiers, and saves results.
"""

import os
import sys
import random
import logging
import importlib
from importlib import reload
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import rsatoolbox
import coloredlogs
from keras import backend as K
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Add custom module paths
sys.path.insert(0, '../classes')
sys.path.insert(0, '../analysis')

# Custom imports
from arena import Arena
from coach import Coach
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer, NNPolicyPlayer, NNValuePlayer
from mcts import MCTS
from utils import *
from compute_RSA import *
import tournament
import tournament_new

# Reset TensorFlow graph
tf.compat.v1.reset_default_graph()

# Configure logging
log = logging.getLogger(__name__)

# Configure numpy printing
np.set_printoptions(precision=3, suppress=True)

# Reload tournament module
reload(tournament)
participant_iters = tournament.participant_iters

# Initialize game
g = Game(4, 9, 4)

def squeeze_activity(activity_dict_layer: Dict) -> Dict:
    """
    Reshape and squeeze layer activities into flattened arrays.
    
    Args:
        activity_dict_layer: Dictionary containing layer activities
        
    Returns:
        Dictionary with reshaped activities
    """
    data_activity_dict = {}
    for key in activity_dict_layer:
        list_layer_activity = activity_dict_layer[key]
        if list_layer_activity[0].ndim == 1:
            data_layer_activity = [item for item in list_layer_activity]
        elif list_layer_activity[0].ndim == 2:
            d1, d2 = list_layer_activity[0].shape
            new_length = d1 * d2
            data_layer_activity = [np.reshape(np.squeeze(x), new_length) for x in list_layer_activity]
        else:
            d1, d2, d3 = list_layer_activity[0].shape
            new_length = d1 * d2 * d3
            data_layer_activity = [np.reshape(np.squeeze(x), new_length) for x in list_layer_activity]
        data_activity_dict[str(key)] = data_layer_activity
    return data_activity_dict

def get_labels(df_boards_features_3: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate balanced labels and indices for positive/negative samples.
    
    Args:
        df_boards_features_3: DataFrame containing board features
        
    Returns:
        Tuple containing (labels array, positive indices, negative indices)
    """
    positive_idx_3 = np.where(df_boards_features_3 > 0)[0]
    positive_3_num = len(positive_idx_3)
    positive_idx_3_list = positive_idx_3.tolist()
    negative_idx_3 = [i for i in range(0, 8138) if i not in positive_idx_3_list]
    
    if positive_3_num <= len(negative_idx_3):
        negative_idx_3_balanced = np.array(random.sample(negative_idx_3, positive_3_num))
        positive_idx_3_balanced = positive_idx_3
        labels_3 = np.array([0] * positive_3_num * 2)
    else:
        negative_idx_3_balanced = negative_idx_3
        positive_idx_3_balanced = np.array(random.sample(positive_idx_3_list, len(negative_idx_3_balanced)))
        labels_3 = np.array([0] * len(negative_idx_3_balanced) * 2)
    
    half_idx_3 = int(len(labels_3) / 2)
    labels_3[0:half_idx_3] = 1
    return labels_3, positive_idx_3_balanced, negative_idx_3_balanced