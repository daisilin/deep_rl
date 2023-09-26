import logging
import math

import numpy as np
import scipy
import scipy.special as ss
from utils import *

EPS = 1e-8

log = logging.getLogger(__name__)

MCTS_ARGS = dotdict({
        'numMCTSSims': 100,
        'cpuct': 2,
        'track_color':False,
        })

# class MCTS_select_value(MCTS):
#     """
#     This class handles the MCTS tree.
#     """

#     def __init__(self, game, nnet, args):
#         super().__init__(game,nnet,args)
#     def getActionProb(self, canonicalBoard, temp=1):
#         for _ in range(self.args.numMCTSSims):
#             self.search(canonicalBoard)

#         s = self.game.stringRepresentation(canonicalBoard)
#         vals = np.array([self.Qsa[(s, a)] if (s, a) in self.Qsa else -1000 for a in range(self.game.getActionSize())]).astype(np.float32)
#         counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

#         if 'w_count' not in self.args.keys():
#             w_count = 0. # weight assigned to result from count
#         else:
#             w_count = self.args.w_count

#         if temp == 0:
#             bestvalA = (vals == np.max(vals)).astype(np.float32)
#             bestcountA = (counts == np.max(counts)).astype(np.float32)
#             bestcombinaed = bestvalA + bestcountA * w_count
#             bestAs = np.array(np.argwhere(bestcombinaed == np.max(bestcombinaed))).flatten()
#             # bestAs = np.array(np.argwhere(vals == np.max(vals))).flatten()
#             bestA = np.random.choice(bestAs)
#             probs = [0] * len(counts)
#             probs[bestA] = 1
#             return probs
#         vals = vals * (1./temp)
#         counts = np.array([x ** (1. / temp) for x in counts])
#         probs_val = ss.softmax(vals).astype('float64')
#         probs_val[probs_val < 1e-3] = 0
#         probs_val = probs_val / probs_val.sum()
#         # counts_sum = float(sum(counts))
#         probs_count = counts / counts.sum()
#         # probs = [x / counts_sum for x in counts]
#         probs = (1-w_count)*probs_val + w_count * probs_count
#         print(probs.reshape(4,9))
#         print(probs.reshape(4,9).sum())
#         return probs


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    #[SZ] clear the tree
    def refresh(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    #[SZ], combine value and visit counts, default all visit counts
    # include dirichlet noise
    def getActionProb(self, canonicalBoard, temp=1, dir_alpha=0,epsilon=0):
        s = self.game.stringRepresentation(canonicalBoard)
        if s in self.Ps.keys():
            if dir_alpha > 0:
                self.Ps[s] = self.Ps[s] * (1-epsilon) + epsilon * np.random.dirichlet(dir_alpha * np.ones(self.game.getActionSize()))
            search_count = self.args.numMCTSSims
        else:              
            self.search(canonicalBoard)
            search_count = self.args.numMCTSSims - 1
            if dir_alpha > 0:
                try:
                    self.Ps[s] = self.Ps[s] * (1-epsilon) + epsilon * np.random.dirichlet(dir_alpha * np.ones(self.game.getActionSize()))
                    
                except:
                    print('no P for s?')
        for _ in range(search_count):
            self.search(canonicalBoard)
        
        vals = np.array([self.Qsa[(s, a)] if (s, a) in self.Qsa else -1000 for a in range(self.game.getActionSize())]).astype(np.float32)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if 'w_count' not in self.args.keys():
            w_count = 1. # weight assigned to result from count
        else:
            w_count = self.args.w_count

        if temp == 0:
            bestvalA = (vals == np.max(vals)).astype(np.float32)
            bestcountA = (counts == np.max(counts)).astype(np.float32)
            bestcombinaed = bestvalA + bestcountA * w_count
            bestAs = np.array(np.argwhere(bestcombinaed == np.max(bestcombinaed))).flatten()
            # bestAs = np.array(np.argwhere(vals == np.max(vals))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        vals = vals * (1./temp)
        counts = np.array([x ** (1. / temp) for x in counts])
        probs_val = ss.softmax(vals).flatten().astype('float64') # if no flatten, shape messed up
        
        probs_val[probs_val < 1e-3] = 0
        
        probs_val = probs_val / probs_val.sum()
        
        # counts_sum = float(sum(counts))
        probs_count = counts / counts.sum()
        # probs = [x / counts_sum for x in counts]
        probs = (1-w_count)*probs_val + w_count * probs_count
        
        return probs


    # def getActionProb(self, canonicalBoard, temp=1):
    #     """
    #     This function performs numMCTSSims simulations of MCTS starting from
    #     canonicalBoard.
    #     Returns:
    #         probs: a policy vector where the probability of the ith action is
    #                proportional to Nsa[(s,a)]**(1./temp)
    #     """
    #     for _ in range(self.args.numMCTSSims):
    #         self.search(canonicalBoard)

    #     s = self.game.stringRepresentation(canonicalBoard)
    #     counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

    #     if temp == 0:
    #         bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
    #         bestA = np.random.choice(bestAs)
    #         probs = [0] * len(counts)
    #         probs[bestA] = 1
    #         return probs
        
    #     counts = [x ** (1. / temp) for x in counts]
    #     counts_sum = float(sum(counts))
    #     probs = [x / counts_sum for x in counts]
    #     return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)
        # s = np.array_str(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1) 
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard) # (boardsize,) scalar
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v 

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


class MCTS_color(MCTS):
    def __init__(self, game, nnet, args):
        super().__init__(game,nnet,args)

    def getActionProb(self, canonicalBoard, color, temp=1, dir_alpha=0,epsilon=0):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonicalBoard)
        if s in self.Ps.keys():
            if dir_alpha > 0:
                self.Ps[s] = self.Ps[s] * (1-epsilon) + epsilon * np.random.dirichlet(dir_alpha * np.ones(self.game.getActionSize()))
            search_count = self.args.numMCTSSims
        else:              
            self.search(canonicalBoard, color)
            search_count = self.args.numMCTSSims - 1
            if dir_alpha > 0:
                try:
                    self.Ps[s] = self.Ps[s] * (1-epsilon) + epsilon * np.random.dirichlet(dir_alpha * np.ones(self.game.getActionSize()))
                    
                except:
                    print('no P for s?')
        for _ in range(search_count):
            self.search(canonicalBoard, color)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return np.array(probs)

    def search(self, canonicalBoard, color):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)
        # s = np.array_str(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1) 
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard, color) # (boardsize,) scalar
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v 

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        next_color = np.abs(color - 1) #black - 1, white - 0, 
        v = self.search(next_s, next_color)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
