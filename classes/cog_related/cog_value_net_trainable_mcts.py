import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import pandas as pd

from keras.models import *
from keras.layers import *
from keras.optimizers import *

sys.path.append('..')
sys.path.append('../beck')
import copy
from neural_net import NeuralNet
from utils import *

from beck.beck_nnet import NNetWrapper
from cog_related import cog_value_net as cvn


N_feats = 5

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
    'N_feats':N_feats
})

class CogValueNet():
    '''
    policy same as beck_nnet
    value, a linear tanh layer on the pre computed features: center, 2_con, 2_uncon, 3, 4
    '''
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y),name='board')    # s: batch_size x board_x x board_y
        self.input_feats = Input(shape=(self.args.N_feats * 2,),name='feat') # self and oppo
        
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
#         self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1
        self.v = Dense(1, activation='tanh',name = 'v', use_bias=False)(self.input_feats)    # batch_size x 1 # use predefined features

#         self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model = Model(inputs=[self.input_boards, self.input_feats], outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        

class NNetWrapper_cog(NNetWrapper):
    def __init__(self, game):
        super().__init__(game)
        self.nnet = CogValueNet(game, args)
        self.inv_dist_to_center = cvn.get_inv_dist_to_center(game)
        

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_feats = np.asarray([cvn.get_all_feat_self_oppo(b, self.inv_dist_to_center) for b in input_boards])
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        # self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        self.nnet.model.fit(x = [input_boards, input_feats], y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()
        input_feats = cvn.get_all_feat_self_oppo(board, self.inv_dist_to_center)
        # preparing input
        board = board[np.newaxis, :, :]
        input_feats = input_feats[np.newaxis, :]

        # run
        pi, v = self.nnet.model.predict([board,input_feats])

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def predict_batch(self,board):
        """
        [SZ]
        different input and output dim comparing to predict
        board: np array num_in_batch x board_x x board_y
        """
        input_feats = np.array([cvn.get_all_feat_self_oppo(b, self.inv_dist_to_center) for b in board])
        # print(input_feats.shape)
        # print(board.shape)
        pi, v = self.nnet.model.predict([board,input_feats])
        return pi, v