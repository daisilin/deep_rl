import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

import ray



from os.path import dirname, abspath
# d = dirname(dirname(abspath(__file__))) # get parent dir in a robust way
# print(d)
# sys.path.append(d)

# d = dirname(abspath(__file__)) # get curr dir in a robust way
# print(d)
# sys.path.append(d)


# from utils import *




# ray.init(ignore_reinit_error=True)


class OthelloNNet():
    def __init__(self, game, args):
        # game params
        from utils import dotdict
        import keras.models as km
        import keras.layers as kl
        import keras.optimizers as ko
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = km.Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = kl.Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = kl.Flatten()(h_conv4)       
        s_fc1 = kl.Dropout(args.dropout)(kl.Activation('relu')(kl.BatchNormalization(axis=1)(kl.Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = kl.Dropout(args.dropout)(kl.Activation('relu')(kl.BatchNormalization(axis=1)(kl.Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = kl.Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = kl.Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = km.Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=ko.Adam(args.lr))

def create_OrthelloNNet_model(game,args):
    from utils import dotdict
    import keras.models as km
    import keras.layers as kl
    import keras.optimizers as ko
    board_x, board_y = game.getBoardSize()
    action_size = game.getActionSize()
    args = args

    # Neural Net
    input_boards = km.Input(shape=(board_x, board_y))    # s: batch_size x board_x x board_y

    x_image = kl.Reshape((board_x, board_y, 1))(input_boards)                # batch_size  x board_x x board_y x 1
    h_conv1 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
    h_conv2 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
    h_conv3 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
    h_conv4 = kl.Activation('relu')(kl.BatchNormalization(axis=3)(kl.Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
    h_conv4_flat = kl.Flatten()(h_conv4)       
    s_fc1 = kl.Dropout(args.dropout)(kl.Activation('relu')(kl.BatchNormalization(axis=1)(kl.Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
    s_fc2 = kl.Dropout(args.dropout)(kl.Activation('relu')(kl.BatchNormalization(axis=1)(kl.Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
    pi = kl.Dense(action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x action_size
    v = kl.Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

    model = km.Model(inputs=input_boards, outputs=[pi, v])
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=ko.Adam(args.lr))
    return model

@ray.remote
class NNetWrapper():
    def __init__(self, game):
        from utils import dotdict
        args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 64,
            'cuda': False,
            'num_channels': 512,
        })
        self.nnet = OthelloNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)


    @ray.method(num_returns=2)
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


if __name__ == '__main__':
    from beck_game import BeckGame as Game
    # d = dirname(dirname(abspath(__file__))) # get parent dir in a robust way
    # print(d)
    # sys.path.append(d)
    ray.init(ignore_reinit_error=True)
    game = Game(4,9,4)
    nn = NNetWrapper.remote(game)
    b = ray.put(game.getInitBoard())

    print(ray.get([nn.predict.remote(bb) for bb in [b,b]]))