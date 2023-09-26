# Global imports
import math
import numpy as np
from queue import Queue
from threading import Thread

class ModelWorker(Thread):
    def __init__(self, queue, model, parallel_threads):
        Thread.__init__(self)
        self.queue = queue
        self.model = model
        self.parallel_threads = parallel_threads
    
    def run(self):
        while True:
            pass

class MCTS():
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.actions_size = self.game.get_actions_size()

        self.validate_mcts_args(args)
        self.args = args

        self.iterate = self.straight_iterate if args['parallel_threads'] == 1 else self.parallel_iterate

        self.reset()

    def reset(self):
        self.Qsa = {}
        self.Wsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Ts = {}
        self.As = {}
        self.Vs = {}

    def validate_mcts_args(self, args):
        assert 'parallel_threads' in args, 'MCTS args must specify parallel_threads'
        assert 'cpuct' in args, 'MCTS args must specify cpuct'
        assert 'mcts_iterations' in args, 'MCTS args must specify mcts_iterations'

    def get_probs(self, canonical_state, temperature=0):
        canonical_state = canonical_state.copy()
        self.straight_iterate(canonical_state)

        for _ in range(self.args['mcts_iterations'] - 1):
            self.iterate(canonical_state)
        
        s = self.game.get_hash_of_state(canonical_state)
        counts = np.array([self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.actions_size)])

        # TODO Dirichlet noise!
        if temperature == 0:
            best_action = np.argmax(counts)
            probs = np.zeros(self.actions_size)
            probs[best_action] = 1
            return probs
        else:
            probs = np.power(counts, 1./temperature)
            probs /= probs.sum()
            return probs

    def parallel_iterate(self, canonical_state):
        pass

    def straight_iterate(self, canonical_state):
        s = self.game.get_hash_of_state(canonical_state)

        if s not in self.Ts:
            # Then we haven't visited this node before: check if it's terminal
            self.Ts[s] = self.game.get_result(canonical_state, 1)

        if self.Ts[s] is not None:
            # Then s is a terminal leaf node: return -value
            return -self.Ts[s]

        if s not in self.Ps:
            # Then s is a non-terminal leaf node:
            self.Ps[s], v = self.model.predict(canonical_state)
            allowed_actions = self.game.get_allowed_actions(canonical_state, 1)
            self.Ps[s] *= allowed_actions

            sum_Ps_s = self.Ps[s].sum()
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                self.Ps[s] += allowed_actions
                self.Ps[s] /= self.Ps[s].sum()
                print('Something bad happened')
            
            self.As[s] = allowed_actions
            self.Ns[s] = 0
            return -v
        
        allowed_actions = self.As[s]
        best_value = -float('inf')
        best_action = -1

        for a in range(self.actions_size):
            if allowed_actions[a]:
                # if epsilon == 0:
                #     probs = self.Ps[s][a]
                # else:
                #     dirichlet_noise = np.random.dirichlet([config.ALPHA] * len(current_node.children))
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
                else:
                    u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s])
                
                if u > best_value:
                    best_value = u
                    best_action = a
        
        next_state, next_player = self.game.get_next_state(canonical_state, 1, best_action)
        next_state = self.game.get_canonical_form(next_state, next_player)

        v = self.straight_iterate(next_state)

        a = best_action
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
        
        self.Ns[s] += 1
        return -v


