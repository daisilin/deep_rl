"""
Distributed neural network implementation for board games using Ray.

This module provides both class-based and functional implementations of neural networks
that can be distributed across multiple processes using Ray framework.
"""

import os
import numpy as np
import ray
from typing import Tuple, List, Optional, Dict, Any

import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
from utils import dotdict

# Default configuration
DEFAULT_ARGS = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class OthelloNNet:
    """Neural network implementation for Othello-like board games."""
    
    def __init__(self, game, args: dotdict):
        """
        Initialize neural network architecture.
        
        Args:
            game: Game instance providing board dimensions and action space
            args: Configuration arguments for network architecture
        """
        # Game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Build network
        self.model = self._build_model()
        
    def _build_model(self) -> km.Model:
        """
        Build the neural network architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        input_boards = km.Input(shape=(self.board_x, self.board_y))
        x = kl.Reshape((self.board_x, self.board_y, 1))(input_boards)
        
        # Convolutional layers
        conv_block_params = {
            'num_channels': self.args.num_channels,
            'use_bias': False,
            'axis': 3
        }
        
        # First two conv blocks with same padding
        for _ in range(2):
            x = self._conv_block(x, kernel_size=3, padding='same', **conv_block_params)
            
        # Last two conv blocks with valid padding
        x = self._conv_block(x, kernel_size=3, padding='valid', **conv_block_params)
        x = self._conv_block(x, kernel_size=2, padding='valid', **conv_block_params)
        
        # Flatten and dense layers
        x = kl.Flatten()(x)
        x = self._dense_block(x, units=1024, dropout=self.args.dropout)
        x = self._dense_block(x, units=512, dropout=self.args.dropout)
        
        # Output heads
        pi = kl.Dense(self.action_size, activation='softmax', name='pi')(x)
        v = kl.Dense(1, activation='tanh', name='v')(x)
        
        # Create and compile model
        model = km.Model(inputs=input_boards, outputs=[pi, v])
        model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=ko.Adam(self.args.lr)
        )
        return model
        
    @staticmethod
    def _conv_block(x, num_channels: int, kernel_size: int, padding: str,
                   use_bias: bool, axis: int) -> kl.Layer:
        """Create a convolutional block with batch normalization and activation."""
        x = kl.Conv2D(
            filters=num_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=use_bias
        )(x)
        x = kl.BatchNormalization(axis=axis)(x)
        x = kl.Activation('relu')(x)
        return x
        
    @staticmethod
    def _dense_block(x, units: int, dropout: float) -> kl.Layer:
        """Create a dense block with batch normalization, activation and dropout."""
        x = kl.Dense(units, use_bias=False)(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(dropout)(x)
        return x

def create_OthelloNNet_model(game, args: dotdict) -> km.Model:
    """
    Functional implementation to create neural network model.
    
    Args:
        game: Game instance providing board dimensions and action space
        args: Configuration arguments
        
    Returns:
        Compiled Keras model
    """
    board_x, board_y = game.getBoardSize()
    action_size = game.getActionSize()
    
    # Create model using the same architecture as OthelloNNet
    nnet = OthelloNNet(game, args)
    return nnet.model

@ray.remote
class NNetWrapper:
    """Distributed neural network wrapper using Ray."""
    
    def __init__(self, game):
        """
        Initialize distributed neural network.
        
        Args:
            game: Game instance providing board dimensions and action space
        """
        self.nnet = OthelloNNet(game, DEFAULT_ARGS)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """
        Train the network on examples.
        
        Args:
            examples: List of (board, policy, value) training examples
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        self.nnet.model.fit(
            x=np.asarray(input_boards),
            y=[np.asarray(target_pis), np.asarray(target_vs)],
            batch_size=DEFAULT_ARGS.batch_size,
            epochs=DEFAULT_ARGS.epochs
        )

    @ray.method(num_returns=2)
    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Make prediction for a single board position.
        
        Args:
            board: Board state as numpy array
            
        Returns:
            tuple: (policy distribution, value prediction)
        """
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]

    def predict_batch(self, board: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for a batch of board positions.
        
        Args:
            board: Batch of board states [batch_size x board_x x board_y]
            
        Returns:
            tuple: (policy distributions, value predictions)
        """
        return self.nnet.model.predict(board)

    def save_checkpoint(self, folder: str = 'checkpoint',
                       filename: str = 'checkpoint.pth.tar') -> None:
        """
        Save model weights to file.
        
        Args:
            folder: Directory to save checkpoint
            filename: Name of checkpoint file
        """
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder: str = 'checkpoint',
                       filename: str = 'checkpoint.pth.tar') -> None:
        """
        Load model weights from file.
        
        Args:
            folder: Directory containing checkpoint
            filename: Name of checkpoint file
        """
        filepath = os.path.join(folder, filename)
        self.nnet.model.load_weights(filepath).expect_partial()

def main():
    """Example usage of distributed neural network."""
    from beck_game import BeckGame as Game
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Create game and network instances
    game = Game(4, 9, 4)
    nn = NNetWrapper.remote(game)
    
    # Example prediction
    board = ray.put(game.getInitBoard())
    predictions = ray.get([nn.predict.remote(b) for b in [board, board]])
    print(f"Predictions for two identical boards: {predictions}")

if __name__ == '__main__':
    main()