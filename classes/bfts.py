import logging
import math

import numpy as np
from utils import dotdict

EPS = 1e-8

log = logging.getLogger(__name__)

DEFAULT_args = dotdict({'numBFSsims':5,'PruningThresh':0.5})


class BFTS():
    '''
    best first search
    '''
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.tree = {} # {board_str: node}
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        # self.Nsa = {}  # stores #times edge s,a was visited
        # self.Ns = {}  # stores #times canonicalBoard s was visited
        # self.Ps = {}  # stores initial policy (returned by neural net)
        # self.EXPs = {} # store whether a state has been expanded
        self.Qs = {}
        self.Es = {}  # stores game.getGameEnded ended for canonicalBoard s
        self.Vs = {}  # stores game.getValidMoves for canonicalBoard s
    
    @staticmethod
    def prune(v_batch, PruningThresh):
        vmax = np.max(v_batch)
        to_keep = np.nonzero(np.abs(v_batch - vmax) < PruningThresh)[0]
        return to_keep

    def refresh(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=0, refresh_first=False):
        '''
        This function performs numBFSsims simulations of BFS starting from
        canonicalBoard.
        Returns:

        '''
        if refresh_first:
            self.refresh()

        for _ in range(self.args.numBFSsims):
            self.search(canonicalBoard)

        # s = self.game.stringRepresentation(canonicalBoard)
        s = np.array_str(canonicalBoard)
        values = [self.Qsa[(s, a)] if (s, a) in self.Qsa else -10000 for a in range(self.game.getActionSize())]
        # print(self.Qsa)

        # deterministic:
        # if temp == 0:
        # temp doesn't work, always deterministic
        bestAs = np.array(np.argwhere(np.array(values) == np.max(values))).flatten()
        # print(np.max(values))
        # print(values)
        # print(values == np.max(values))
        bestA = np.random.choice(bestAs)
        probs = [0] * len(values)
        probs[bestA] = 1
        return probs


    def search(self, canonicalBoard):
        '''
        One iter of best first search
        '''
        s = np.array_str(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1) * 100 # a large number such that winning trumps other values
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # the popping-up situation: if state not visited before, 
        # evaluate, return
        # expansion - backpropagation
        # s = np.array_str(canonicalBoard)
        if s not in self.Vs.keys(): # not having checked for valids, i.e. leaf node
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Vs[s] = valids
            a_batch = np.nonzero(valids)[0] # the indices of valid actions
            x_l = a_batch // self.game.n
            y_l = a_batch % self.game.n
            n_valids = len(x_l)
            ind_l = np.arange(n_valids)
            # board_batch = np.tile(-canonicalBoard,(n_valids,1,1)) # vectorized way of getting a batch for evaluation; -canonicalBoard, from opponent's perspective
            # board_batch[ind_l, x_l, y_l] = -1 # from opponent's perspective
            # _,v_batch = self.nnet.predict_batch(board_batch) # from opponent's perspective
            # for ii,b in enumerate(board_batch):
            #     s_next = np.array_str(b)
            #     self.Es[s_next] = self.game.getGameEnded(b, 1)
            #     if self.Es[s_next]!=0:
            #         v_batch[ii] = self.Es[s_next] # update the value to game ending state

            # prune
            # to_keep = self.prune(-v_batch,self.args.PruningThresh) # -v_batch, from self perspective

            # best_val = -float('inf')
            # for b in to_keep:
            #     u = -v_batch[b]
            #     # u = v_batch[b]
            #     self.Qsa[(s, a_batch[b])] = u
            #     if u > best_val:
            #         best_val = u


            board_batch = np.tile(canonicalBoard,(n_valids,1,1)) # vectorized way of getting a batch for evaluation; canonicalBoard, from self's perspective
            board_batch[ind_l, x_l, y_l] = 1 # from self perspective
            _,v_batch = self.nnet.predict_batch(board_batch) # from self perspective
            for ii,b in enumerate(board_batch):
                s_next = np.array_str(b)
                self.Es[s_next] = self.game.getGameEnded(b, 1) * 100 # a large number such that winning trumps other values
                if self.Es[s_next]!=0:
                    v_batch[ii] = self.Es[s_next]

            # prune
            
            to_keep = self.prune(v_batch,self.args.PruningThresh) # v_batch, from self perspective



            best_val = -float('inf')
            for b in to_keep:
                u = v_batch[b]
                self.Qsa[(s, a_batch[b])] = u
                if u > best_val:
                    best_val = u
            


            return -best_val # return the max value among children; - sign because the backprop step involves a switch in the player

        # selection - expansion
        cur_best = -float('inf')
        best_act = -1
        val_dict = {}
        valids = self.Vs[s] 
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)]
                    val_dict[a] = u
                # else: a is pruned
                    if u > cur_best:
                        cur_best = u
                        best_act = a
                        # val_dict[a] = u # this caused a huge bug!!!!! should be outside of the if; otw lots of actions not put in val_dict for comparison!!!

        
        

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, best_act)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # self.game.display(canonicalBoard)
        # self.game.display(next_s)
        # print(valids)
        # print(val_dict)
        # print('\n')

        v_backpropped = self.search(next_s)
        
        # if np.abs(v_backpropped)==100:
        #     self.game.display(canonicalBoard)
        #     self.game.display(next_s)
        #     print(v_backpropped)
        #     print('\n')
            

        self.Qsa[(s, best_act)] = v_backpropped # update the expanded node value
        # reevaluate among all a at one level and back propagate
        val_dict[best_act]= v_backpropped 


        # self.game.display(canonicalBoard)
        # self.game.display(next_s)
        # print(val_dict)
        # print('\n')

        # return -np.min(list(val_dict.values())) #from the perspective of the target of the backprop, ??
        return -np.max(list(val_dict.values()))


    def search_two_step():
        '''
        every time the tree expands, it expands two node --i.e. simulate one step of opponent and then 
        '''
        pass

# ==========util functions for traversing the tree===========

def get_children(board_arr,tree,verbose=True):
    children = []
    for (bstr,action),val in tree.Qsa.items():
        b_array = g.str_rep_to_array(bstr)
        if (b_array==board_arr).sum()==(board_arr.shape[0]*board_arr.shape[1]):
            new_b, new_p = game.getNextState(b_array,1,action)
            children.append((new_b,action,val))
        #     print(b_array)
            if verbose:
                print('new board')
                game.display(new_b)

    #             print(new_b)
                print(f'action {action}')
                print(f'value {val}\n')
            
    return children


import copy
def get_parent(board_arr,action):
    x,y = action //9,action%9
    parent = copy.copy(board_arr)
    parent[x,y] = 0
    return -parent

def get_board_size(tree):
    Ssa = {}
    for (bstr,action),val in tree.Qsa.items():
        b_array = tree.game.str_rep_to_array(bstr)
        size=np.sum(b_array!=0)
        Ssa[(bstr,action)] = size
    return Ssa

def get_largest_board(tree,offset=0):
    Ssa = get_board_size(tree)
    maxsize=np.max(list(Ssa.values())) # list here important
    board_q_l = []
    for (bstr,action), size in Ssa.items():
        if size==maxsize - offset:
            val = tree.Qsa[(bstr,action)]
            b_array = tree.game.str_rep_to_array(bstr)
            new_b, new_p = tree.game.getNextState(b_array,1,action)
            board_q_l.append((new_b,val))
    return board_q_l

def traverse_tree_principal_variation(board_arr,tree):
    children = get_children(board_arr,tree,verbose=False)
    board_sequence = []
    while len(children) > 0:
        bestval = np.max([val for _,_,val in children])
        best_board_a_val = [(b,a,val) for b,a,val in children if val==bestval][0]
        board_sequence.append(best_board_a_val)
        children = get_children(-best_board_a_val[0],tree,verbose=False)
    return board_sequence


