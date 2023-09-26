import pandas as pd
import sys,os,copy,pdb,importlib
sys.path.append('../classes')
sys.path.append('../analysis')
import numpy as np
import matplotlib.pyplot as plt

# import tournament_new as tn
import tournament_new_sz as tn

import create_database_sz as cd
importlib.reload(tn)

import beck.beck_game
from importlib import reload
reload(beck.beck_game)
from beck.beck_game import BeckGame as Game
game = Game(4,9,4)
import pickle
from arena import Arena
from mcts import MCTS
importlib.reload(tn)
game = Game(4,9,4)
all_p = pd.read_pickle(cd.DATABASE_LOC)

# res = tn.merge_res_to_base()
#tournament_res = pd.read_pickle('../final_agents/ai_all_player_round_robin_base.pkl')
#tournament_res = pd.read_pickle('/scratch/zz737/fiar/tournaments/ai_all_player_round_robin_base.pkl')
tournament_res = pd.read_pickle('/scratch/xl1005/deep-master/tournaments/ai_all_player_round_robin_base.pkl')
import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

import keras
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import tensorflow as tf
from tensorflow import Tensor

sys.path.append('..')
from neural_net import NeuralNet
from utils import *

reload(beck.beck_nnet)
from beck.beck_nnet import OthelloNNet, NNetWrapper, NNetWrapper_color


from collections import OrderedDict


from pickle import Pickler, Unpickler

ex_tournament = 8#6
ex_id = 'checkpoints_mcts100_cpuct2_id_res3-0'#'checkpoints_mcts100_cpuct2_id_1'#
ex_fn_part = 'checkpoint_61'#'checkpoint_55'#

ex_dir = f'/scratch/zz737/fiar/tournaments/tournament_{ex_tournament}/{ex_id}'
ex_fn = f'{ex_fn_part}.pth.tar'

load_folder_file = None#(ex_dir,'best.pth.tar')
nepochs = 80

def load_data(track_color=False):
    # load and organize data

    # weight_dir = '/scratch/zz737/fiar/tournaments/tournament_1/checkpoints_mcts100_cpuct2'
    # fn = 'checkpoint_39.pth.tar'

    #weight_dir = '/scratch/zz737/fiar/tournaments/tournament_6/checkpoints_mcts100_cpuct2_id_1'
    #fn = #'checkpoint_55.pth.tar'

    # load_folder_file = (weight_dir,fn)
    load_folder_file = (ex_dir, ex_fn)

    modelFile = os.path.join(load_folder_file[0], load_folder_file[1])
    examplesFile = modelFile + ".examples"
    with open(examplesFile, "rb") as f:
        trainExamplesHistory = Unpickler(f).load()


    from random import shuffle
    trainExamples = []
    for e in trainExamplesHistory:
        trainExamples.extend(e)
    shuffle(trainExamples)
    N_trainexs = len(trainExamples)
    if track_color:
        trainExamples = [(*x,int(x[0].sum()==0)) for x in trainExamples] # black: sum to 0, 1; white: not sum to 0, 0

    return trainExamples
    # return trainExamples[:1]


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 200,#80,#200,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,#512,
    'wl2':0.0001,
    'n_res':19
})

class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
 
class OthelloNNet_ks4(): # an extra kernelsize=4 layer, changed the next to also size 4
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv0 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 4, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 4, padding='same', use_bias=False)(h_conv0)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))


class OthelloNNet_wl2():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        
        alpha = args.wl2
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.bias))
        
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

class OthelloNNet_convhead():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-3) x (board_y-3) x num_channels
        
        h_conv_pi = Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, 2, padding='same', use_bias=False)(h_conv4)))        
        h_conv_v = Activation('relu')(BatchNormalization(axis=3)(Conv2D(16, 2, padding='same', use_bias=False)(h_conv4)))        
        
        h_conv_pi_flat = Flatten()(h_conv_pi)
        h_conv_v_flat = Flatten()(h_conv_v)
        


        s_fc_pi1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv_pi_flat))))  # batch_size x 1024
        s_fc_pi2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc_pi1))))  # batch_size x 1024

        s_fc_v1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(h_conv_v_flat))))  # batch_size x 1024
        
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc_pi2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc_v1)                    # batch_size x 1
        
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        
#         alpha = 0.0001
#         for layer in self.model.layers:
#             if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
#                 layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.kernel))
#             if hasattr(layer, 'bias_regularizer') and layer.use_bias:
#                 layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.bias))
        
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))


        
class OthelloNNet_convsame():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv3)))        # batch_size  x (board_x-3) x (board_y-3) x num_channels
        
        h_conv_pi = Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, 2, padding='same', use_bias=False)(h_conv4)))        
        h_conv_v = Activation('relu')(BatchNormalization(axis=3)(Conv2D(16, 2, padding='same', use_bias=False)(h_conv4)))        
        
        h_conv_pi_flat = Flatten()(h_conv_pi)
        h_conv_v_flat = Flatten()(h_conv_v)
        


        s_fc_pi1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv_pi_flat))))  # batch_size x 1024
        s_fc_pi2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc_pi1))))  # batch_size x 1024

        s_fc_v1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(h_conv_v_flat))))  # batch_size x 1024
        
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc_pi2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc_v1)                    # batch_size x 1
        
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        
        
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))



def bn_relu(inputs: Tensor)-> Tensor:
    bn = BatchNormalization(axis=3)(inputs)
    relu = ReLU()(bn)
    return relu

def residual_block(x: Tensor, filters: int, kernel_size: int=3)->Tensor:
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding='same')(x)
    y = bn_relu(y)
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding='same')(y)
    y = BatchNormalization(axis=3)(y)
    out = Add()([x,y])
    out = ReLU()(out)
    return out

class OthelloNNet_resnet(): # has l2 on w
    def __init__(self, game, args, return_compiled=True):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args



        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        final_input = self.input_boards        
        if 'track_color' in args.keys():
            if args.track_color:
                # self.input_color = Input(shape=[])
                # input_color = keras.layers.Lambda(lambda x: tf.fill((self.board_x, self.board_y, 1),x))(self.input_color)
                self.input_color = Input(shape=(self.board_x, self.board_y)) # color should be expanded to match the shape of the board!
                input_color = Reshape((self.board_x, self.board_y, 1))(self.input_color)
                x_image = Concatenate(axis=-1)([x_image, input_color])
                final_input = [self.input_boards, self.input_color]

        t = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, args.res_ks, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        
        for n in range(args.n_res):
            t = residual_block(t,filters=args.num_channels,kernel_size=args.res_ks)
        
        # policy head
        pt = Conv2D(filters=2,kernel_size=1,strides=1,padding='same')(t)
        pt = bn_relu(pt)
        pt = Flatten()(pt)
        self.pi = Dense(self.action_size,activation='softmax',name='pi')(pt)
        
        # value head
        vt = Conv2D(filters=1,kernel_size=1,strides=1,padding='same')(t)
        vt = bn_relu(vt)
        vt = Flatten()(vt)
        vt = Dense(256, activation='relu')(vt)
        self.v = Dense(1,activation='tanh',name='v')(vt)

        # self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model = Model(inputs=final_input, outputs=[self.pi, self.v])

        alpha = args.wl2
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.bias))
        if return_compiled:
            self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        



def get_args(**kwargs):
    args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 200,#80,#200,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,#512,
    'wl2':0.0001,
    'n_res':19,
    'res_ks':3,
    'track_color':False,
    })
    for k,v in kwargs.items():
        args[k] = v
    return args



# othello_dict = OrderedDict(old = OthelloNNet(game,args),
#      wl2 = OthelloNNet_wl2(game,args),
#      convhead = OthelloNNet_convhead(game,args),
#      convsame = OthelloNNet_convsame(game,args),
#      res19 = OthelloNNet_resnet(game,get_args(n_res=19, epochs=80, num_channels=256)),
#      ks4 = OthelloNNet_ks4(game,args),
#      res9 = OthelloNNet_resnet(game,get_args(n_res=9, epochs=1, num_channels=256)),
#     )


othello_dict = OrderedDict(
    # convsame = OthelloNNet_convsame(game,get_args(epochs=nepochs, num_channels=512)),
     # res3 = OthelloNNet_resnet(game,get_args(n_res=3, epochs=nepochs, num_channels=256)),
     # res15 = OthelloNNet_resnet(game,get_args(n_res=15, epochs=nepochs, num_channels=256)),
     # res9 = OthelloNNet_resnet(game,get_args(n_res=9, epochs=nepochs, num_channels=256)),
     # res5 = OthelloNNet_resnet(game,get_args(n_res=5, epochs=nepochs, num_channels=256)),
     # res19 = OthelloNNet_resnet(game,get_args(n_res=19, epochs=nepochs, num_channels=256)),
     # res3_ks4 = OthelloNNet_resnet(game,get_args(n_res=3, epochs=nepochs, num_channels=256, res_ks=4)),
     # res15_ks4 = OthelloNNet_resnet(game,get_args(n_res=3, epochs=nepochs, num_channels=256, res_ks=4)),
    )

nnet_dict = OrderedDict()
for name,onet in othello_dict.items():
    nnet_dict[name] = NNetWrapper(game, args=othello_dict[name].args, nnet=othello_dict[name])

othello_dict['res15_color'] = OthelloNNet_resnet(game, args=get_args(n_res=15, epochs=nepochs, num_channels=256, track_color=True))
nnet_dict['res15_color'] = NNetWrapper_color(game, args=othello_dict['res15_color'].args, nnet=othello_dict['res15_color'])     

def main(i):
    if load_folder_file is not None:
        name = 'res3'
        assert name in load_folder_file[0]
        args = pickle.load(open(os.path.join(load_folder_file[0],'args.p'),'rb'))
        on = OthelloNNet_resnet(game, args)
        nnet = NNetWrapper(game,nnet=on,args=args)
        nnet.load_checkpoint(folder=load_folder_file[0], filename=load_folder_file[1])
        print(f'load from {load_folder_file}')
        nnet.args.epochs = nepochs

    else:
        name = list(nnet_dict.keys())[i]
        print(f'training {name}...')
        nnet = nnet_dict[name]
        
    if 'track_color' in nnet.args.keys():
        tc = nnet.args.track_color
    else:
        tc = False
    

    folder = '/scratch/zz737/fiar/sl/resnet'
    # filename = f'{name}_Ex_tournament_6_mcts100_cpuct2_id_1_iter55.pth.tar'
    sub_folder = f'{name}_Ex_tournament_{ex_tournament}_{ex_id}_{ex_fn_part}'
    # filename = f'{name}_Ex_tournament_{ex_tournament}_{ex_id}_{ex_fn_part}'
    filename = 'final.pth.tar'
    # nnet.nnet.model.save(os.path.join(folder,filename))
    final_folder = os.path.join(folder,sub_folder)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    

    args_fn = os.path.join(final_folder, 'args.p')
    with open(args_fn, "wb") as f:
        Pickler(f).dump(nnet.args)

    trainEx = load_data(track_color=tc)
    nnet.train(trainEx, checkpoint_dir=final_folder)
    nnet.save_checkpoint(folder=final_folder, filename=filename)

    # nnet.save_checkpoint(folder='/scratch/zz737/fiar/sl', filename=f'{name}_Ex_tournament_6_mcts100_cpuct2_id_1_iter55.pth.tar')

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
