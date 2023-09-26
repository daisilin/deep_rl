import sys
sys.path.append('..')

# Global imports
import numpy as np
import os
import pickle as pkl
import tensorflow as tf

# Local imports
from model import Model
from beck.nnet_construction import build_residual_cnn
from utils import printl

class RandomBeckModel(Model):
    def __init__(self, game):
        self.game = game

    def train(self, examples):
        pass

    def predict(self, state):
        p = np.random.random(36)
        p[~self.game.get_allowed_actions(state)] = 0
        return p / p.sum(), np.random.random()

    def predict_batch(self, state):
        pass

    def save_checkpoint(self, folder, filename):
        pass

    def load_checkpoint(self, folder, filename):
        pass


def sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))

class NnetBeckModel(Model):
    def __init__(self, game, nnet_args):
        self.game = game
        self.nnet = build_residual_cnn(nnet_args)

    def train(self, examples, batch_size, epochs):
        states, target_probs, target_values = list(zip(*examples))
        nnet_input = self.states_to_nnet_input(states)
        target_probs = np.asarray(target_probs)
        target_values = np.asarray(target_values)
        self.nnet.fit(x = nnet_input, y = [target_values, target_probs], batch_size = batch_size, epochs = epochs)

    def predict(self, state):
        # printl('Pre-prediction!')
        nnet_input = self.state_to_nnet_input(state)
        nnet_output = self.nnet.predict(nnet_input)
        value = nnet_output[0][0]
        policy = self.mask_probs_by_state(nnet_output[1][0], state)
        # printl('Post-prediction!')
        return policy, value[0]
    
    def get_policy(self, state):
        return self.predict(state)[0]
    
    def get_value(self, state):
        return self.predict(state)[1]

    def predict_batch(self, states):
        nnet_input = self.states_to_nnet_input(states)
        nnet_output = self.nnet.predict(nnet_input)
        value = nnet_output[0,:,0]
        policy = nnet_output[1,:]
        for i in range(states.shape[0]):
            policy[i] = self.mask_probs_by_state(policy[i], states[i])
        return policy, value

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            printl(f'Checkpoint dir does not exist - making directory {folder}')
            os.mkdir(folder)
        else:
            printl('Checkpoint dir exists')
        # pkl.dump(self.nnet, open(filepath, 'w+'))
        self.nnet.save(filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise(f'No model in path {filepath}')
        # self.nnet = pkl.load(open(filepath, 'r+'))
        self.nnet = tf.keras.models.load_model(filepath)

    def set_weights(self, weights):
        self.nnet.set_weights(weights)

    def get_weights(self):
        return self.nnet.get_weights()

    @staticmethod
    def state_to_nnet_input(state):
        nnet_input = np.zeros((1, 3, *state.shape))
        nnet_input[0][0] = (state == 1)
        nnet_input[0][1] = (state == 2)
        is_player_ones_turn = 1 - np.sum(state != 0)
        nnet_input[0][2] = np.full(state.shape, is_player_ones_turn)
        return nnet_input

    @staticmethod
    def states_to_nnet_input(states):
        nnet_input = np.zeros((len(states), 3, *states[0].shape))
        for i in range(len(states)):
            state = states[i]
            nnet_input[i][0] = (state == 1)
            nnet_input[i][1] = (state == 2)
            is_player_ones_turn = 1 - np.sum(state != 0)
            nnet_input[i][2] = np.full(state.shape, is_player_ones_turn)
        return nnet_input
    
    def mask_probs_by_state(self, probs, state):
        # probs = sigmoid(logits)
        probs[~self.game.get_allowed_actions(state)] = 0
        probs = probs / np.sum(probs)
        return probs

ExportedModel = NnetBeckModel