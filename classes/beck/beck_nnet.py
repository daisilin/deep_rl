import tensorflow as tf
tf.compat.v1.reset_default_graph()
import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from tensorflow import Tensor
from keras import backend as K

sys.path.append('..')
from neural_net import NeuralNet
from utils import *

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
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

 
class NNetWrapper(NeuralNet):
    def __init__(self, game, args=None, nnet='default'): #[sz] added args in the arguments
        if isinstance(nnet,str) and nnet=='default':
            self.nnet = OthelloNNet(game, args)
        else:
            self.nnet = nnet
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        if args is None and self.nnet is not None:
            self.args = self.nnet.args
        else:
            self.args=args

    def train(self, examples, checkpoint_dir=None):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        if checkpoint_dir is not None:
            
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir,'checkpointSGD{epoch:d}'),
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            callbacks=[model_checkpoint_callback]
        else:
            callbacks = None

        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = self.args.batch_size, epochs = self.args.epochs,callbacks=callbacks)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def predict_batch(self,board):
        """
        [SZ]
        different input and output dim comparing to predict
        board: np array num_in_batch x board_x x board_y
        """
        pi, v = self.nnet.model.predict(board)
        return pi, v


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath):
        #     raise FileNotFoundError("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath).expect_partial()
    def get_all_activity(self,board):
        #https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
        # color_input = np.full((1,*board.shape),color)
        # board_input = board[np.newaxis, :, :]
        # final_input = [board_input,color_input]
        # inp = self.nnet.model.input         # input placeholder
        # outputs = [layer.output for layer in self.nnet.model.layers]          # all layer outputs
        # functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]  # evaluation function
        # # Testing
        # layer_outs = [func([final_input, 1.]) for func in functors]
        #new method
        board_input = board[np.newaxis, :, :]
        final_input = board_input
        for layerIndex, layer in enumerate(self.nnet.model.layers):
            func = K.function([self.nnet.model.get_layer(index=0).input], layer.output)
            layer_outs = func([final_input])  # input_data is a numpy array
            print(layer_outs)
        return layer_outs

    def get_all_activity_no_dropout(self,board):
        # print(self.nnet.model.summary())
        board_input = board[np.newaxis, :, :]
        final_input = board_input
        #board = board[np.newaxis,...]
        inp = self.nnet.model.input 
        outputs = [layer.output for layer in self.nnet.model.layers]
        functors = [K.function([inp], [out]) for out in outputs]   # evaluation functions
        layer_outs = [func([final_input]) for func in functors]
        return layer_outs
    def get_all_activity_no_dropout_batch(self,board):
        #input all boards
        #output nested list of all layer activities for all boards 
        final_input = board

        # for layerIndex, layer in enumerate(self.nnet.model.layers):
        #     func = K.function([self.nnet.model.get_layer(index=0).input], layer.output)
        #     layerOutput = func([final_input[0]])  # input_data is a numpy array
        #board = board[np.newaxis,...]
        inp = self.nnet.model.input 
        outputs = [layer.output for layer in self.nnet.model.layers]
        functors = [K.function([inp], [out]) for out in outputs]   # evaluation functions
        all_outs = []
        for i in range(len(board)):
            layer_outs = [func([final_input[i][np.newaxis, :, :]]) for func in functors]     
            all_outs.append(layer_outs) 
        return all_outs

    def get_layer_name(self):
        print(self.nnet.model.summary())
        names = []
        # print(self.nnet.model.summary())
        for item in self.nnet.model.layers:
            names.append(item.name)
        #print(self.nnet.model.layers)
        return names
    def get_layer_activity(self,board,layer_name):
        # print(self.nnet.model.summary())
        #given a layer name, output the activity
        board_input = board[np.newaxis, :, :]
        final_input = board_input
        intermediate_layer_model = Model(inputs=self.nnet.model.input,
                                 outputs=self.nnet.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(final_input)
        return intermediate_output
    # def get_all_activity_batch(self,board):
    #     # print(self.nnet.model.summary())
    #     #given a layer name, output the activity
    #     # board_input = board[np.newaxis,...]
    #     final_input = board
    #     names = self.get_layer_name()
    #     all_activity_batch_dict = {}
    #     for layer_name in names:
    #         intermediate_layer_model = Model(inputs=self.nnet.model.input,
    #                                  outputs=self.nnet.model.get_layer(layer_name).output)
    #         intermediate_output = intermediate_layer_model.predict(final_input[0][np.newaxis,...])
    #     all_activity_batch_dict[layer_name] = intermediate_output
    #     return all_activity_batch_dict
class NNetWrapper_color(NNetWrapper):
    def __init__(self, game, args=None, nnet=None):
        super().__init__(game, args=args,nnet=nnet)


    def train(self, examples, checkpoint_dir=None):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs, colors = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        color_input = np.full((self.board_x,self.board_y,len(colors)),colors).swapaxes(0,-1).swapaxes(1,2) # turn color_batch into b x 4 x 9

        if checkpoint_dir is not None:
            
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir,'checkpointSGD{epoch:d}'),
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            callbacks=[model_checkpoint_callback]
        else:
            callbacks = None
        self.nnet.model.fit(x = [input_boards,color_input], y = [target_pis, target_vs], batch_size = self.args.batch_size, epochs = self.args.epochs, callbacks=callbacks) 

    def predict(self, board, color):
        '''
        color: scalar, need to be turned into 1 x 4 x 9
        '''
        
        color_input = np.full((1,*board.shape),color)
        board_input = board[np.newaxis, :, :]
        # run
        pi, v = self.nnet.model.predict([board_input,color_input])

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def predict_batch(self, board_batch, color_batch):
        '''
        color_batch: (b,)
        '''
        color_batch = np.full((self.board_x,self.board_y,len(color_batch)),color_batch).swapaxes(0,-1).swapaxes(1,2) # turn color_batch into b x 4 x 9
        pi, v = self.nnet.model.predict([board_batch, color_batch])
        return pi, v

    def get_all_activity(self,board,color):
        #https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
        # color_input = np.full((1,*board.shape),color)
        # board_input = board[np.newaxis, :, :]
        # final_input = [board_input,color_input]
        # inp = self.nnet.model.input         # input placeholder
        # outputs = [layer.output for layer in self.nnet.model.layers]          # all layer outputs
        # functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]  # evaluation function
        # # Testing
        # layer_outs = [func([final_input, 1.]) for func in functors]
        #new method
        color_input = np.full((1,*board.shape),color)
        board_input = board[np.newaxis, :, :]
        final_input = [board_input,color_input]
        for layerIndex, layer in enumerate(self.nnet.model.layers):
            func = K.function([self.nnet.model.get_layer(index=0).input], layer.output)
            layer_outs = func([final_input])  # input_data is a numpy array
            print(layer_outs)
        return layer_outs

    def get_all_activity_no_dropout(self,board,color):
        print(self.nnet.model.summary())
        color_input = np.full((1,*board.shape),color)
        board_input = board[np.newaxis, :, :]
        final_input = [board_input,color_input]
        #board = board[np.newaxis,...]
        inp = self.nnet.model.input 
        outputs = [layer.output for layer in self.nnet.model.layers]
        functors = [K.function([inp], [out]) for out in outputs]   # evaluation functions
        layer_outs = [func([final_input]) for func in functors]
        return layer_outs
    def get_all_activity_no_dropout_batch(self,board,color):
        #input all boards
        #output nested list of all layer activities for all boards 
        # color_input = np.full((len(board),*board.shape),color)
        board_input = board

        inp = self.nnet.model.input 
        outputs = [layer.output for layer in self.nnet.model.layers]
        functors = [K.function([inp], [out]) for out in outputs]   # evaluation functions
        all_outs = []
        for i in range(len(board)):
            final_input = [board_input[i][np.newaxis, :, :],np.full((1,*board[i].shape),color)]
            layer_outs = [func(final_input) for func in functors]     
            all_outs.append(layer_outs) 
        return all_outs

    def get_layer_name(self):
        names = []
        # print(self.nnet.model.summary())
        for item in self.nnet.model.layers:
            names.append(item.name)
        return names
    def get_layer_activity(self,board,color,layer_name):
        #given a layer name, output the activity
        color_input = np.full((1,*board.shape),color)
        board_input = board[np.newaxis, :, :]
        final_input = [board_input,color_input]
        intermediate_layer_model = Model(inputs=self.nnet.model.input,
                                 outputs=self.nnet.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(final_input)
        return intermediate_output