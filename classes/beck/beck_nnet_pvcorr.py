import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

import keras
import tensorflow as tf
import tensorflow.math as tfm
import keras.backend as K 
from keras.models import *
from keras.layers import *
from keras.optimizers import *

sys.path.append('..')
sys.path.append('../beck')
from importlib import reload
from neural_net import NeuralNet
from utils import *

EPS = 1e-5#0#1e-5
import beck.beck_nnet
reload(beck.beck_nnet)
from beck.beck_nnet import OthelloNNet, NNetWrapper
import supervised_learning

reload(supervised_learning)
from supervised_learning import OthelloNNet_resnet, get_args

# args = dotdict({
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 10,
#     'batch_size': 64,
#     'cuda': False,
#     'num_channels': 512,
#     'sc':1,
# })
ARGS = get_args(n_res=9,num_channels=256,track_color=True,sc=1)

def masked_mean(x,mask, **kwargs):
    mask = K.cast(mask, K.floatx())
    return K.sum(x * mask, **kwargs) / K.sum(mask, **kwargs)
    
def masked_std(x, mask, **kwargs):
    mask = K.cast(mask, K.floatx())
    masked_x = x * mask
    x_mean = masked_mean(x,mask,**kwargs)
    var = masked_mean( (x - x_mean)**2, mask, **kwargs)
    std = tf.math.sqrt(var)
    return std

def masked_standardize(x, mask, **kwargs):
    return (x - masked_mean(x, mask, keepdims=True,**kwargs)) / (masked_std(x, mask, keepdims=True,**kwargs)+EPS)

class Correlation(keras.layers.Layer):
    def __init__(self):
        super(Correlation,self).__init__()
    
    def call(self, policy, val_reshaped, valids):
        '''

        policy: b x action_size
        val_reshaped: b x action_size
        
        valids: b x 36; for masking out the occupied indices in correlation computation; valids = game.getValidMovesBatch(input, 1)
        '''

        p_stddz = masked_standardize(policy, valids, axis=1) 
        val_stddz = masked_standardize(val_reshaped, valids, axis=1)
        corr = masked_mean(p_stddz * val_stddz, valids, axis=1, keepdims=False)
        
        # mask = ~tf.math.is_nan(corr)
        # return corr[mask][:,None] # if one var is constant, ignore this data

        return corr[:,None]

import tensorflow as tf
import tensorflow.math as tfm

class OthelloNNet_sub():
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


class OthelloNNet_with_PVcorr(Model):
    def __init__(self,game,args=None,sub_model='default',**kwargs):
        super(OthelloNNet_with_PVcorr, self).__init__()
        self.game=game
        self.action_size = game.getActionSize()
        self.args = args
        
        if isinstance(sub_model,str) and sub_model=='default':
            if args is None:
                print('args is None, use default')
                print(ARGS)
                self.args = ARGS
            self.args['track_color'] = False
                # self.sub_model = OthelloNNet_sub(game,self.args).model
            self.sub_model = OthelloNNet_resnet(game,args=self.args,return_compiled=False).model # now using resnet
        else:
            self.sub_model = sub_model

        self.corr = Correlation()

        # lr = 1e-3
        # if lr in kwargs.keys():
        #     lr = kwargs['lr']
        # optim = keras.optimizers.Adam(learning_rate=lr)

        # sc = 1
        # if sc in kwargs.keys():
        #     sc = kwargs['sc']
        # self.compile(optimizer=optim, loss=['categorical_crossentropy','mean_squared_error',Scaled_MSE_with_mask(sc)])
        
    def call(self,inputs,training=True):
        '''
        input: b x game.m x game.n
        input_expanded: (b x action_size) x game.m x game.n
        valids: b x 36; for masking out the occupied indices in correlation computation; valids = game.getValidMovesBatch(input, 1)
        '''
        B = inputs.shape[0]
        # B = tf.shape(inputs)[0]
        # self.B = B
        inputs_expanded,valids = self.get_next_step_boards(self.game, inputs)

        pi,v_train  = self.sub_model(inputs,training=training) # b x action_size
        _,v_expanded = self.sub_model(inputs_expanded, training=training) #  (b x action_size) x 1
        v_expanded_reshaped = K.reshape(v_expanded,(B, self.action_size)) # b x action_size
        corr_mask = self.corr(pi, v_expanded_reshaped, valids)
        return pi, v_train, corr_mask
    
    # @staticmethod
    @tf.function
    def get_next_step_boards(self, game,board_batch):
        '''
        [SZ]
        board_batcch: b x game.m x game.n
        input_expanded: (b x action_size) x game.m x game.n
        valids: b x action_size
        '''
        board_batch = tf.cast(board_batch,tf.int32)
        # B = self.B

        valids = tf.cast(K.reshape((board_batch==0),(board_batch.shape[0],-1)),tf.float32) # this would be problamatic if batch size does not divide total number of examples
        # valids = tf.cast(K.reshape((board_batch==0),(tf.shape(board_batch)[0],-1)),tf.float32) # this would be problamatic if batch size does not divide total number of examples
        # valids = tf.cast(K.reshape((board_batch==0),(B,-1)),tf.float32) # this would be problamatic if batch size does not divide total number of examples
        action_size = game.getActionSize()

        # B_expanded = B*action_size

        input_expanded = K.repeat_elements(board_batch,action_size, axis=0) # repeat for : 1 2 -> 1 1 2 2; in tf2.7, K.repeat is good
        
        inds_to_be_placed = K.tile(K.arange(action_size),board_batch.shape[0]) # tile for: 1 2 -> 1 2 1 2
        # inds_to_be_placed = K.tile(K.arange(action_size),[tf.shape(board_batch)[0]]) # tile for: 1 2 -> 1 2 1 2
        # inds_to_be_placed = K.tile(K.arange(action_size),[B]) # tile for: 1 2 -> 1 2 1 2
        
        xs, ys = inds_to_be_placed // game.n, inds_to_be_placed % game.n
        
        expanded_batch_inds = K.arange(input_expanded.shape[0])
        # expanded_batch_inds = K.arange(tf.shape(input_expanded)[0])
        # expanded_batch_inds = K.arange(B_expanded)

        inds_tensor = tf.stack([expanded_batch_inds,xs,ys],axis=1)

        input_expanded_updated = tf.tensor_scatter_nd_update(input_expanded, inds_tensor,tf.ones(input_expanded.shape[0],dtype=tf.int32))    
        # input_expanded_updated = tf.tensor_scatter_nd_update(input_expanded, inds_tensor,tf.ones(tf.shape(input_expanded)[0],dtype=tf.int32))
        # input_expanded_updated = tf.tensor_scatter_nd_update(input_expanded, inds_tensor,tf.ones(B_expanded,dtype=tf.int32))

        return tf.stop_gradient(input_expanded_updated), tf.stop_gradient(valids)


class OthelloNNet_with_PVcorr_color(OthelloNNet_with_PVcorr):
    def __init__(self,game,args=None,sub_model='default'):
        super().__init__(game,args=args,sub_model=None)
        if isinstance(sub_model,str) and sub_model=='default':
            if args is None:
                print('args is None, use default')
                print(ARGS)
                self.args = ARGS
            self.args['track_color'] = True
                # self.sub_model = OthelloNNet_sub(game,self.args).model
            self.sub_model = OthelloNNet_resnet(game,args=self.args,return_compiled=False).model # now using resnet
        else:
            self.sub_model = sub_model
        assert self.args.track_color

    def call(self,inputs_,training=True):
        '''
        input: b x game.m x game.n
        input_expanded: (b x action_size) x game.m x game.n
        valids: b x 36; for masking out the occupied indices in correlation computation; valids = game.getValidMovesBatch(input, 1)
        '''
        inputs = inputs_[0]
        colors  = inputs_[1]
        B = inputs.shape[0]
        inputs_expanded,valids = self.get_next_step_boards(self.game, inputs)
        colors = tf.cast(colors, tf.int32) #important!
        # oppo_colors = tf.abs(1-colors)
        colors_expanded = K.repeat_elements(colors,self.game.getActionSize(), axis=0) # repeat for : 1 2 -> 1 1 2 2; in tf2.7, K.repeat is good
        # need to flip the colors for the expanded boards!!! NO.. confused
        # colors_expanded = K.repeat_elements(oppo_colors,self.game.getActionSize(), axis=0) # repeat for : 1 2 -> 1 1 2 2; in tf2.7, K.repeat is good

        pi,v_train  = self.sub_model((inputs,colors),training=training) # b x action_size
        _,v_expanded = self.sub_model((inputs_expanded,colors_expanded), training=training) #  (b x action_size) x 1
        v_expanded_reshaped = K.reshape(v_expanded,(B, self.action_size)) # b x action_size
        corr_mask = self.corr(pi, v_expanded_reshaped, valids)
        return pi, v_train, corr_mask

class Scaled_MSE_with_mask(keras.losses.Loss):
    def __init__(self, scale):
        super().__init__()
        if scale is None:
            self.scale = 1
        else:
            self.scale = scale
    def call(self, y_true, y_pred):
        '''
        y_true: b, not actually used. Fixed at 1.
        y_pred: corr, masked; cannot put tuple here! 
        One option is to set EPS to be small, then there won't be nan in y_pred
        The other is to mask the nan in y_pred
        '''
        # mask = tf.cast(tf.math.equal(y_pred,0.0),dtype=tf.float32) # if strictly 0, means even outputs
        res = self.scale * keras.losses.mean_squared_error(1, y_pred)
        mask = tf.math.logical_not(tf.math.is_nan(res))
        # res_masked = tf.boolean_mask(res,mask)
        # return res[mask] 
        return res
        # res_masked = res*mask
        # return res_masked
        
        # return tf.cond(tf.math.equal(y_pred.shape[0],0),true_fn=lambda:tf.zeros(1),false_fn=lambda:res) # seems tf.where involving nan will cause nan in gradients; try if y_pred is empty return 0;
        
        # return tf.where(tf.math.is_nan(res),tf.zeros_like(res, dtype=tf.float32),res) # if nan, ie mask all False, then return 0? specify the dtype is important, otherwise would screw up everything else

class NNetWrapper_pvcorr(NNetWrapper):
    def __init__(self, game, args=None,nnet='default'):
        super().__init__(game, args=args, nnet=None)
        if isinstance(nnet,str) and nnet == 'default':
            self.nnet = OthelloNNet_with_PVcorr(game, args=args, sub_model=None)
        else:
            self.nnet=nnet
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        if self.nnet is not None:
            self.args = self.nnet.args

    def train(self, examples, checkpoint_dir=None):
        """
        examples: list of examples, each example is of form (board, pi, v)

        NB! Batch_size has to divide num of examples!!!
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        self.nnet.compile(loss=['categorical_crossentropy','mean_squared_error',Scaled_MSE_with_mask(self.args.sc)], optimizer=Adam(self.args.lr))
        if checkpoint_dir is not None:
            
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir,'checkpoint{epoch:02d}'),
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            callbacks = [model_checkpoint_callback]
        else:
            callbacks = None
        self.nnet.fit(x = input_boards, y = [target_pis, target_vs, target_pis], batch_size = self.args.batch_size, epochs = self.args.epochs, callbacks=callbacks)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.sub_model.predict(board) # predict only work for the sub_model, i.e. the othello_net, because the reshape in get_next_step_boards does not work with predict

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def predict_batch(self,board):
        """
        [SZ]
        different input and output dim comparing to predict
        board: np array num_in_batch x board_x x board_y
        """
        pi, v = self.nnet.sub_model.predict(board)
        return pi, v


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.sub_model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath):
        #     raise FileNotFoundError("No model in path {}".format(filepath))
        self.nnet.sub_model.load_weights(filepath).expect_partial()


class NNetWrapper_pvcorr_color(NNetWrapper_pvcorr):
    def __init__(self, game, args=None,nnet='default'):
        super().__init__(game,args=args,nnet=None)
        if isinstance(nnet,str) and nnet == 'default':
            self.nnet = OthelloNNet_with_PVcorr_color(game, args=args, sub_model='default')
        else:
            self.nnet = nnet
        if self.nnet is not None:
            self.args = self.nnet.args
        assert self.args.track_color

    def train(self, examples,checkpoint_dir=None):
        """
        examples: list of examples, each example is of form (board, pi, v)

        NB! Batch_size has to divide num of examples!!!
        """
        if len(examples)%self.args.batch_size!=0: # make sure batch_size always divides len of examples
            len_ex = int((len(examples) // self.args.batch_size) * self.args.batch_size)
            examples = examples[:len_ex]
        input_boards, target_pis, target_vs, colors = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        color_input = np.full((self.board_x,self.board_y,len(colors)),colors).swapaxes(0,-1).swapaxes(1,2) # turn color_batch into b x 4 x 9

        self.nnet.compile(loss=['categorical_crossentropy','mean_squared_error',Scaled_MSE_with_mask(self.args.sc)], optimizer=Adam(self.args.lr))

        if checkpoint_dir is not None:
            
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir,'checkpoint{epoch:d}'),
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            callbacks=[model_checkpoint_callback]
        else:
            callbacks = None

        self.nnet.fit(x = [input_boards,color_input], y = [target_pis, target_vs, target_pis], batch_size = self.args.batch_size, epochs = self.args.epochs, callbacks=callbacks)

    def predict(self, board, color):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]
        color_input = np.full((1,*board.shape),color)

        # run
        # pi, v = self.nnet.sub_model.predict(board) # predict only work for the sub_model, i.e. the othello_net, because the reshape in get_next_step_boards does not work with predict
        # pi, v = self.nnet.sub_model.predict((board,color_input)) # predict only work for the sub_model, i.e. the othello_net, because the reshape in get_next_step_boards does not work with predict
        pi, v = self.nnet.sub_model.predict((board,color_input)) # predict only work for the sub_model, i.e. the othello_net, because the reshape in get_next_step_boards does not work with predict

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def predict_batch(self,board_batch,color_batch):
        """
        [SZ]
        different input and output dim comparing to predict
        board: np array num_in_batch x board_x x board_y
        """
        color_batch = np.full((self.board_x,self.board_y,len(color_batch)),color_batch).swapaxes(0,-1).swapaxes(1,2) # turn color_batch into b x 4 x 9
        pi, v = self.nnet.model.predict([board_batch, color_batch])
        
        return pi, v


