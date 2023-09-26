import argparse
import os,pdb
import shutil
import time
import random
import numpy as np 
import scipy
from scipy.ndimage import convolve1d
import jax
import jax.numpy as jnp
from jax import jit, vmap

import math
import sys
import pandas as pd

from keras.models import *
from keras.layers import *
from keras.optimizers import *

sys.path.append('..')
import copy
from neural_net import NeuralNet
from utils import *

'''
This file contains the functions for doing cog model value and a lazy network: using the policy from a trained nnet
canonicalBoard here aways means a batch of boards! b x 4(nr) x 9(nc)
'''

class NNetWrapper(NeuralNet):
    def __init__(self, game, nnet, args):
        self.nnet = nnet
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.inv_dist_to_center = get_inv_dist_to_center(game)
        self.w = args[0]
        self.C = args[1]

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board_for_nnet = board[np.newaxis, :, :]

        # run
        if self.nnet is not None:
            pi,_ = self.nnet.predict(board) 
        else:
            pi = np.ones(self.action_size) / self.action_size

        # need get_value
        # v = get_value(board, self.C, self.w, self.inv_dist_to_center)
        v = get_value(board_for_nnet, self.C, self.w, self.inv_dist_to_center)


        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi, v

    def predict_batch(self, board):
        if self.nnet is not None:
            pi,_ = self.nnet.predict_batch(board) 
        else:
            pi = np.ones((board.shape[0],self.action_size)) / self.action_size # actualy never used in predict_batch, since pi is used for mcts; predict_batch used in bfs
        
        v = get_value(board, self.C, self.w, self.inv_dist_to_center)
        return pi, v

def get_value(canonicalBoard, C, w, inv_dist_to_center,**kwargs):
    
    # assume w ordered in the same way as the return of get_all_feat
    feat_vals_self, feat_names_self = get_all_feat(canonicalBoard,inv_dist_to_center,**kwargs)
    feat_vals_opp, feat_names_opp = get_all_feat(-canonicalBoard,inv_dist_to_center,**kwargs)
    # if self_playing ==1:
    #     Cself = 1
    #     Copp = C
    # else:
    #     Cself = C
    #     Copp = 1
    # V = 1 * np.dot(feat_vals_self,w) - C * np.dot(feat_vals_opp,w)
    V = 1 * np.einsum('bf,f->b',feat_vals_self,w) - C * np.einsum('bf,f->b',feat_vals_opp,w)
    return np.tanh(V) # normalize it to -1 and 1

# all feat self other
def get_all_feat_self_oppo(canonicalBoard,inv_dist_to_center,diag_seperate=False):
    self_feat_vals, feat_names = get_all_feat(canonicalBoard,inv_dist_to_center,diag_seperate=diag_seperate)
    oppo_feat_vals, feat_names = get_all_feat(-canonicalBoard,inv_dist_to_center,diag_seperate=diag_seperate)
    return np.concatenate([self_feat_vals,oppo_feat_vals],axis=1) # axis 0 is batch
def get_all_feat_self_oppoself_added(canonicalBoard,inv_dist_to_center,diag_seperate=False):
    self_feat_vals, feat_names = get_all_feat(canonicalBoard,inv_dist_to_center,diag_seperate=diag_seperate)
    oppo_feat_vals, feat_names = get_all_feat(-canonicalBoard,inv_dist_to_center,diag_seperate=diag_seperate)
    return self_feat_vals+oppo_feat_vals,feat_names # axis 0 is batch

# all feat
def get_all_feat(canonicalBoard,inv_dist_to_center,diag_seperate=False):
    b,_,_ = canonicalBoard.shape 
    f_center = get_center_feat(canonicalBoard,inv_dist_to_center)
    
    feat_names = ['center']
    f_inarow = filter_all_ortho_diag(canonicalBoard, filters_dict=None)
    if diag_seperate==False: # diag features not counted seperately
        f_inarow = pd.DataFrame(f_inarow).sum(axis=1) # get series, index: 2_con, 2_uncon, etc.; values, counts
        inarow_names = list(f_inarow.index)
        inarow_vals = f_inarow.values

    if b==1: # batchsize =1 need seperate treatment. pd df sum values would not retain the array structure of each element, if batch=1, so then f_center would be an array whereas the rest of the values not
        feat_vals = [f_center[0]]
        feat_vals.extend(inarow_vals)
        feat_vals = np.array(feat_vals)[None,:] # create the batch dimension for consistency
    else:
        inarow_vals = np.stack(inarow_vals,axis=1) #b x feat
        feat_vals = np.concatenate([f_center[:,None],inarow_vals],axis=1)
    
    feat_names.extend(inarow_names)

    # return np.array(feat_vals), np.array(feat_names)
    return feat_vals, feat_names

# center
def get_center_feat(canonicalBoard,inv_dist_to_center):
    mask = canonicalBoard == 1
    # val = (inv_dist_to_center * mask).sum()
    val = np.einsum('ij,bij->b',inv_dist_to_center,mask)
    return val
def get_inv_dist_to_center(g):
    test_board = g.getInitBoard()
    nr,nc = test_board.shape
    center = ((nr-1)/2, (nc-1)/2)
    row,col = np.indices((nr,nc))
    d_mat = ((row - center[0])**2 + (col - center[1])**2)**(1/2)
    # compute inv dist
    inv_dist = 1 / d_mat
    return inv_dist

# connected in a row
def create_filters():
    filters_dict = {'2_con':[[1,1,-1,-1],[-1,1,1,-1],[-1,-1,1,1]],'2_uncon':[[1,-1,-1,1],[1,-1,1,-1],[-1,1,-1,1]],'3':[[1,-1,1,1],[1,1,1,-1],[-1,1,1,1],[1,1,-1,1]],'4':[[1,1,1,1]]}
    return filters_dict


def get_filtered_ortho(filt,canonicalBoard,n_in_row=2,dim='row'):
    # non diagonal; diagonal need to be tweaked first into orthogonal
    # mask = canonicalBoard == 1
    mask = canonicalBoard 
    # if dim=='row':
    #     axis=1
        # axis = 0 #for jax; vmap axis=0, each row as argument
    # else: #dim=='col':
    #     axis=0
    #     # axis = 1 #for jax; vmap axis=1, each col as argument

    filt_with0 = np.ones_like(filt)
    # jax.ops.index_update(filt_with0,filt==-1, 2*np.arange(np.sum(filt==-1)))# jax version # filter with the unwanted slots filled with unequal integers, the comparison between the two filtered results would yield spots that are truly x-in-a-row, without opponent's occupation
    filt_with0[filt==-1] = 2*np.arange(np.sum(filt==-1)) # filter with the unwanted slots filled with unequal integers, the comparison between the two filtered results would yield spots that are truly x-in-a-row, without opponent's occupation
    # def _get_filtered_ortho(filt,filt_with0,canonicalBoard, n_in_row): # to avoid named arguments; argument can't be in control flow either; 
        
    #     # filt_with0[filt==-1] = 2*np.arange(np.sum(filt==-1)) # filter with the unwanted slots filled with unequal integers, the comparison between the two filtered results would yield spots that are truly x-in-a-row, without opponent's occupation
    #     # jax.ops.index_update(filt_with0,filt==-1, 2*np.arange(np.sum(filt==-1)))# jax version # filter with the unwanted slots filled with unequal integers, the comparison between the two filtered results would yield spots that are truly x-in-a-row, without opponent's occupation
    #     # import pdb
    #     # pdb.set_trace()

    # old way
    # convolved_board_1p = np.apply_along_axis(lambda m:np.convolve(m,filt,mode='valid'),axis=axis,arr=mask)
    # convolved_board_1p_with0 = np.apply_along_axis(lambda m:np.convolve(m,filt_with0,mode='valid'),axis=axis,arr=mask)

    # with scipy.ndimage.convolve1d
    # only work with 4 x 9 board so far! If board shape change, the indices below (for selecting the valid part of the convolution results) should change!
    if dim=='row':
        axis=1 + 1 # plus one for the batch dim
        convolved_board_1p = convolve1d(mask, filt, axis=axis)[:,:,1:7] #empirically tested against np.convolve. For Scipy convolve1d the filter starts in the center (origin=0), and always pad such that the result is the same len
        convolved_board_1p_with0 = convolve1d(mask, filt_with0, axis=axis)[:,:,1:7] #
    else: #dim=='col':
        axis=0 + 1
        convolved_board_1p = convolve1d(mask, filt, axis=axis)[:,[1],:] #empirically tested against np.convolve. For Scipy convolve1d the filter starts in the center (origin=0), and always pad such that the result is the same len
        convolved_board_1p_with0 = convolve1d(mask, filt_with0, axis=axis)[:,[1],:] #

    # with jax
    # convolve_wrapper = lambda m:jnp.convolve(m,filt,mode='valid')
    # convolve_wrapper_vec = jit(vmap(convolve_wrapper,in_axes=axis)) #jit here is better than create a sub function and jit that 

    # convolve_wrapper_with0 = lambda m:jnp.convolve(m,filt_with0,mode='valid')
    # convolve_wrapper_with0_vec = jit(vmap(convolve_wrapper_with0,in_axes=axis))

    # convolved_board_1p = convolve_wrapper_with0_vec(mask)
    # convolved_board_1p_with0 = convolve_wrapper_with0_vec(mask)

    selected_inds = (convolved_board_1p == convolved_board_1p_with0).astype(int)
    # convolved_board_1p = convolved_board_1p[selected_inds]
    # f = (convolved_board_1p == n_in_row).sum()
    f=np.einsum('bij,bij->b',convolved_board_1p == n_in_row,selected_inds)
    # f = np.dot((convolved_board_1p == n_in_row).flatten(),selected_inds.flatten())
    #     return f
    # f = _get_filtered_ortho(filt,filt_with0,canonicalBoard, n_in_row)
    return f

    
def get_diag(filt,canonicalBoard,n_in_row=2):
    # patching the board by 4 x 3 squares of 0 on each side, then shift and subselect each row to make the diagonals vertical
    # the shift has to be done in both directions for diagonals of both directions
    # @jit
    # def _get_patched_board_diag():
    # nr,nc = canonicalBoard.shape
    b,nr,nc = canonicalBoard.shape # add in batch
    # patch = np.zeros((nr,4-1))
    patch = np.zeros((b,nr,4-1))
    patched_board = np.concatenate([patch,canonicalBoard,patch],axis=-1)
    # nc_tot = patched_board.shape[1]
    nc_tot = patched_board.shape[2]
    nc_new = nc - (4-1) #contingent upon the facts: 4 rows, 4-i-a-r wins, which means diagonals with total slots fewer than 4 are meaningless
    # patched_board_diag_1 = np.zeros((nr,nc_new)) # 2 diag directions
    patched_board_diag_1 = np.zeros((b,nr,nc_new)) # 2 diag directions
    # patched_board_diag_2 = np.zeros((nr,nc_new))
    patched_board_diag_2 = np.zeros((b,nr,nc_new))
    for i in range(nr):
        # patched_board_diag_1[i,:] = patched_board[i,i+4-1:i+4-1+nc_new]
        patched_board_diag_1[:,i,:] = patched_board[:,i,i+4-1:i+4-1+nc_new]
        # patched_board_diag_1.at[i,:].set(patched_board[i,i+4-1:i+4-1+nc_new])
        # patched_board_diag_2[i,:] = patched_board[i,-(nc_new+(4-1))-i:-(4-1)-i]
        patched_board_diag_2[:,i,:] = patched_board[:,i,-(nc_new+(4-1))-i:-(4-1)-i]
        # patched_board_diag_2.at[i,:].set(patched_board[i,-(nc_new+(4-1))-i:-(4-1)-i])
        # return patched_board_diag_1, patched_board_diag_2
    patched_board_diag_1 = np.concatenate([patched_board_diag_1,patch],axis=-1) #recover the shape of the original board, such that the get_filtered_ortho could work properly
    patched_board_diag_2 = np.concatenate([patch,patched_board_diag_2],axis=-1)
    # import pdb
    # pdb.set_trace()
    f1 = get_filtered_ortho(filt,patched_board_diag_1,n_in_row=n_in_row,dim='col')
    f2 = get_filtered_ortho(filt,patched_board_diag_2,n_in_row=n_in_row,dim='col')
    f = f1 + f2
    return f
def filter_all_ortho_diag(canonicalBoard, filters_dict=None):
    f_count_dict_all = {}
    if filters_dict is None:
        filters_dict = create_filters()
    for direc in ['diag','ortho']:
        f_count_dict = {}
        for k,v in filters_dict.items():
            n_in_row = int(k[0])
            f = 0
            for filt in v:
                filt = np.array(filt)
                if direc =='ortho':
                    for dim in ['row','col']:
                        f +=get_filtered_ortho(filt,canonicalBoard, n_in_row=n_in_row, dim=dim)
                else:
                    # import pdb
                    # pdb.set_trace()
                    f += get_diag(filt, canonicalBoard, n_in_row=n_in_row)
            f_count_dict[k] = f
        f_count_dict_all[direc] = f_count_dict
    return f_count_dict_all









