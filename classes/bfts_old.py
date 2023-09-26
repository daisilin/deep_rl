'''
Best first search with a tree structure, very inefficient, crash with more than 2 searches
'''

import logging
import math

import numpy as np
from utils import dotdict

EPS = 1e-8

log = logging.getLogger(__name__)

DEFAULT_args = dotdict({'numBFSsims':5,'PruningThresh':0.5})

class Node_with_Value():
    def __init__(self,canonicalBoard,player,value,children={},parents={}):
        
        self.canonicalBoard = canonicalBoard 
        self.value = value
        self.children = children
        self.parents = parents
        self.isroot = False
        # self.isremoved = False
        self.player = player #1 or -1
        if self.children is None:
            self.isleaf = True
        else:
            self.isleaf = False
    def get_children_minmax(self):
        curr_max = -10000
        curr_best_move = -1
        if self.children is None:
            print('no children')
            return None
        elif self.children=={}:
            print('children not yet assigned')
            return {}
        for k,node in self.children.items():
            if (node.value * self.player) > curr_max: # if self.player -1, this gets the min
                curr_best_move = k
        try:
            return self.children[curr_best_move]
        except:
            print(f'curr_best_move is {curr_best_move}, something is wrong')

    def remove(self):
        '''
        del the node from parents' dict of children, 
        del self
        keep the key in the tree dict
        '''
        for k, prt in self.parents.items():
            del prt.children[k]
        del self # not removing it in the BFTS tree, just to keep a record?

class BFTS():
    '''
    best first tree search
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

        self.Es = {}  # stores game.getGameEnded ended for canonicalBoard s
        self.Vs = {}  # stores game.getValidMoves for canonicalBoard s
        self.root = None

    def MakeMove(self, canonicalBoard):
        if self.root is not None:
            self.root.isroot = False # preping to change root
        s_curr = np.array_str(canonicalBoard)
        if s_curr not in self.tree.keys():
            _, v = self.nnet.predict(canonicalBoard) # perhaps unecessary for root to comptue value?
            self.tree[s_curr] = Node_with_Value(canonicalBoard, 1, v, parents=None, children={}) #None for no parent, {} for not yet assigned
        self.root = self.tree[s_curr]
        self.root.isroot = True

        for _ in range(self.args.numBFSsims):
            root = self.root
            # s = self.game.stringRepresentation(root)
            s = np.array_str(root.canonicalBoard) # array_str is more intuitive and save more space than tostring!
            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(root.canonicalBoard, 1)
            if self.Es[s] != 0:
                # terminal node
                return 
            n = self.SelectNode()
            self.ExpandNode(n)
            self.BackPropagate(n)

        # select action to max value of next state
        val_l = []
        a_l = []
        for a,child in self.tree[s_curr].children.items():            
            val_l.append(child.value)
            a_l.append(a)
        a = a_l[np.argmax(val_l)]
        return a

    def SelectNode(self):   
        n = self.root
        while n.children is not None and n.children != {}:
            n = n.get_children_minmax()
        return n

    def ExpandNode(self, n):
        canonicalBoard = n.canonicalBoard
        s = np.array_str(canonicalBoard)
        if s not in self.Vs.keys():
            valids = self.game.getValidMoves(canonicalBoard,1)
            self.Vs[s] = valids 
        else:
            valids = self.Vs[s]

        vmax = -1000
        for a in range(self.game.getActionSize()):
            if valids[a]:
                next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
                next_s = self.game.getCanonicalForm(next_s, next_player)
                _, v = self.nnet.predict(next_s) # (boardsize,) scalar
                v = -v # it should be, from current player's pespective, evaluate next board; because the board is flipped so the perspective is from the next player, to translate it back need -. The idea is consistent with the bfs in the paper, but I think the way that more strictly follows the bfs in paper is to evaluate the non-flipped board. This way of flipping value is consistent with the MCTS though and thus with how it was trained.
                
                next_s_str = np.array_str(next_s)
                if next_s_str in self.tree.keys(): 
                    leaf_node = self.tree[next_s_str]
                    leaf_node.parents[a] = n # update the parents of an existing node
                    n.children[a] = leaf_node
                else:
                    leaf_node = Node_with_Value(next_s, next_player, v, parents={a:n}) #when propagating value back, flip the sign; problem is the function is not linear; so it's different to propagate -v or evaluate the -board
                    n.children[a] = leaf_node
                    self.tree[next_s_str] = leaf_node
                if (v * n.player) > vmax: # if player -1, this gets the min
                    vmax = v * n.player

        # pruning 
        # print(f'vmax {vmax}')
        # print(f'n.player {n.player}')
        for a, leaf_node in list(n.children.items()):  # list() to avoid a runtime error
            if np.abs(leaf_node.value * n.player - vmax) > self.args.PruningThresh:
                # leaf_node.isremoved = True # pruned but still kept in the tree, for bookkeeping in cases of multiple parents same children
                # print(f'a {a}')
                # print(f'value {leaf_node.value}')
                leaf_node.remove() 



    def BackPropagate(self, n):
        vmax = -1000
        for a,child in n.children.items():
            if (n.player * child.value) > vmax:
                vmax = n.player * child.value
        if not n.isroot:
            for a, prt in n.parents.items():
                print(prt)
                self.BackPropagate(prt)

        


