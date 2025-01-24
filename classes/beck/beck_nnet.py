"""
Neural network implementations for AlphaZero-style agent.

This module provides neural network architectures and wrappers for training
an AlphaZero-style AI to play board games. It includes both a basic convolutional
network and a ResNet variant, with optional color tracking functionality.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
import keras
from keras.models import Model
from keras.layers import (Input, Dense, Conv2D, BatchNormalization, Activation,
                         Flatten, Reshape, Dropout, ReLU, Add, Concatenate)
from keras.optimizers import Adam
from keras import backend as K

sys.path.append('..')
from neural_net import NeuralNet
from utils import dotdict

# Default hyperparameters
DEFAULT_ARGS = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

def bn_relu(inputs: Tensor) -> Tensor:
    """Applies batch normalization followed by ReLU activation.
    
    Args:
        inputs: Input tensor
        
    Returns:
        Tensor after BatchNorm and ReLU
    """
    bn = BatchNormalization(axis=3)(inputs)
    relu = ReLU()(bn)
    return relu

def residual_block(x: Tensor, filters: int, kernel_size: int = 3) -> Tensor:
    """Creates a residual block with two convolutional layers.
    
    Args:
        x: Input tensor
        filters: Number of filters in conv layers
        kernel_size: Size of conv kernels
        
    Returns:
        Output tensor after residual connection
    """
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding='same')(x)
    y = bn_relu(y)
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding='same')(y)
    y = BatchNormalization(axis=3)(y)
    out = Add()([x, y])
    out = ReLU()(out)
    return out

class OthelloNNet:
    """Basic convolutional neural network architecture for board games."""
    
    def __init__(self, game, args):
        """Initialize the neural network.
        
        Args:
            game: Game instance providing board dimensions and action space
            args: Arguments containing network hyperparameters
        """
        # Game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Network architecture
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)
        
        # Convolutional layers
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 2, padding='valid', use_bias=False)(h_conv3)))
        
        # Fully connected layers
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(
            Dense(1024, use_bias=False)(h_conv4_flat))))
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(
            Dense(512, use_bias=False)(s_fc1))))
        
        # Policy and value heads
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                         optimizer=Adam(args.lr))

class OthelloNNet_resnet:
    """ResNet-based neural network architecture with L2 regularization."""
    
    def __init__(self, game, args, return_compiled=True):
        """Initialize the ResNet architecture.
        
        Args:
            game: Game instance providing board dimensions and action space
            args: Arguments containing network hyperparameters
            return_compiled: Whether to compile the model after initialization
        """
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Input layers
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)
        final_input = self.input_boards

        # Optional color tracking
        if 'track_color' in args.keys() and args.track_color:
            self.input_color = Input(shape=(self.board_x, self.board_y))
            input_color = Reshape((self.board_x, self.board_y, 1))(self.input_color)
            x_image = Concatenate(axis=-1)([x_image, input_color])
            final_input = [self.input_boards, self.input_color]

        # Initial convolution
        t = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, args.res_ks, padding='same', use_bias=False)(x_image)))
        
        # Residual blocks
        for _ in range(args.n_res):
            t = residual_block(t, filters=args.num_channels, kernel_size=args.res_ks)
        
        # Policy head
        pt = Conv2D(filters=2, kernel_size=1, strides=1, padding='same')(t)
        pt = bn_relu(pt)
        pt = Flatten()(pt)
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(pt)
        
        # Value head
        vt = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(t)
        vt = bn_relu(vt)
        vt = Flatten()(vt)
        vt = Dense(256, activation='relu')(vt)
        self.v = Dense(1, activation='tanh', name='v')(vt)

        self.model = Model(inputs=final_input, outputs=[self.pi, self.v])

        # Add L2 regularization
        alpha = args.wl2
        for layer in self.model.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
                layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda layer=layer: keras.regularizers.l2(alpha)(layer.bias))

        if return_compiled:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                             optimizer=Adam(args.lr))

def get_args(**kwargs):
    """Get default arguments with optional overrides.
    
    Args:
        **kwargs: Keyword arguments to override defaults
        
    Returns:
        dotdict containing all arguments
    """
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 200,
        'batch_size': 64,
        'cuda': False,
        'num_channels': 512,
        'wl2': 0.0001,
        'n_res': 19,
        'res_ks': 3,
        'track_color': False,
    })
    for k, v in kwargs.items():
        args[k] = v
    return args

class NNetWrapper(NeuralNet):
    """Wrapper class for neural network implementations."""
    
    def __init__(self, game, args=None, nnet='default'):
        """Initialize the wrapper.
        
        Args:
            game: Game instance
            args: Network arguments (optional)
            nnet: Neural network instance or 'default'
        """
        if isinstance(nnet, str) and nnet == 'default':
            self.nnet = OthelloNNet(game, args)
        else:
            self.nnet = nnet
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args if args is not None else self.nnet.args

    def train(self, examples, checkpoint_dir=None):
        """Train the network on examples.
        
        Args:
            examples: List of (board, pi, v) tuples
            checkpoint_dir: Directory to save checkpoints (optional)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        callbacks = None
        if checkpoint_dir is not None:
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'checkpointSGD{epoch:d}'),
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)]

        self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            callbacks=callbacks)

    def predict(self, board):
        """Predict policy and value for a board position.
        
        Args:
            board: Board state as numpy array
            
        Returns:
            Tuple of (policy, value) predictions
        """
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]

    def predict_batch(self, board):
        """Predict policy and value for a batch of boards.
        
        Args:
            board: Batch of board states as numpy array
            
        Returns:
            Tuple of (policies, values) predictions
        """
        pi, v = self.nnet.model.predict(board)
        return pi, v

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """Save model weights to file.
        
        Args:
            folder: Checkpoint directory
            filename: Name of checkpoint file
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """Load model weights from file.
        
        Args:
            folder: Checkpoint directory
            filename: Name of checkpoint file
        """
        filepath = os.path.join(folder, filename)
        self.nnet.model.load_weights(filepath).expect_partial()

    def get_layer_activity(self, board, layer_name):
        """Get activations for a specific layer.
        
        Args:
            board: Input board state
            layer_name: Name of the layer
            
        Returns:
            Layer activations
        """
        board_input = board[np.newaxis, :, :]
        intermediate_layer_model = Model(
            inputs=self.nnet.model.input,
            outputs=self.nnet.model.get_layer(layer_name).output)
        return intermediate_layer_model.predict(board_input)

    def get_layer_name(self):
        """Get names of all layers in the model.
        
        Returns:
            List of layer names
        """
        return [layer.name for layer in self.nnet.model.layers]

class NNetWrapper_color(NNetWrapper):
    """Wrapper class for neural networks with color tracking."""
    
    def train(self, examples, checkpoint_dir=None):
        """Train the network on examples with color information.
        
        Args:
            examples: List of (board, pi, v, color) tuples
            checkpoint_dir: Directory to save checkpoints (optional)
        """
        input_boards, target_pis, target_vs, colors = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        color_input = np.full(
            (self.board_x, self.board_y, len(colors)), colors).swapaxes(0, -1).swapaxes(1, 2)

        callbacks = None
        if checkpoint_dir is not None:
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'checkpointSGD{epoch:d}'),
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)]

        self.nnet.model.fit(
            x=[input_boards, color_input],
            y=[target_pis, target_vs],
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            callbacks=callbacks)

    def predict(self, board, color):
        """Predict policy and value for a board position with color.
        
        Args:
            board: Board state as numpy array
            color: Player color
            
        Returns:
            Tuple of (policy, value) predictions
        """
        color_input = np.full((1, *board.shape), color)
        board_input = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict([board_input, color_input])
        return pi[0], v[0]

    def predict_batch(self, board_batch, color_batch):
        """Predict policy and value for a batch of boards with colors.
        
        Args:
            board_batch: Batch of board states
            color_batch: Batch of player colors
            
        Returns:
            Tuple of (policies, values) predictions
        """
        color_batch = np.full(
            (self.board_x, self.board_y, len(color_batch)), 
            color_batch).swapaxes(0, -1).swapaxes(1, 2)
        pi, v = self.nnet.model.predict([board_batch, color_batch])
        return pi, v

    def get_layer_activity(self, board, color, layer_name):
        """Get activations for a specific layer with color information.
        
        Args:
            board: Board state as numpy array
            color: Player color
            layer_name: Name of the layer
            
        Returns:
            Layer activations
        """
        color_input = np.full((1, *board.shape), color)
        board_input = board[np.newaxis, :, :]
        final_input = [board_input, color_input]
        intermediate_layer_model = Model(
            inputs=self.nnet.model.input,
            outputs=self.nnet.model.get_layer(layer_name).output)
        return intermediate_layer_model.predict(final_input)

    def get_all_activity_no_dropout(self, board, color):
        """Get activations for all layers without dropout.
        
        Args:
            board: Board state as numpy array
            color: Player color
            
        Returns:
            List of activations for all layers
        """
        color_input = np.full((1, *board.shape), color)
        board_input = board[np.newaxis, :, :]
        final_input = [board_input, color_input]
        
        inp = self.nnet.model.input
        outputs = [layer.output for layer in self.nnet.model.layers]
        functors = [K.function([inp], [out]) for out in outputs]
        return [func([final_input]) for func in functors]

    def get_all_activity_no_dropout_batch(self, board, color):
        """Get activations for all layers for a batch of inputs without dropout.
        
        Args:
            board: Batch of board states
            color: Player color
            
        Returns:
            List of activations for all layers for each input
        """
        board_input = board
        inp = self.nnet.model.input
        outputs = [layer.output for layer in self.nnet.model.layers]
        functors = [K.function([inp], [out]) for out in outputs]
        
        all_outs = []
        for i in range(len(board)):
            final_input = [
                board_input[i][np.newaxis, :, :],
                np.full((1, *board[i].shape), color)
            ]
            layer_outs = [func(final_input) for func in functors]
            all_outs.append(layer_outs)
        return all_outs