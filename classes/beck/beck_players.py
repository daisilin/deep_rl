"""
Player implementations for Four-in-a-Row game.

This module provides different types of players:
- Random player: Makes random valid moves
- Human player: Accepts mouse input via Pygame interface
- Greedy player: Makes moves that maximize immediate score
- Neural network players: Uses trained networks for decision making
"""

import numpy as np
import math
import pygame
import sys
from typing import Tuple, List, Optional

class RandomPlayer:
    """Player that makes random valid moves."""
    
    def __init__(self, game):
        """
        Initialize random player.
        
        Args:
            game: Game instance providing game mechanics
        """
        self.game = game

    def play(self, board: np.ndarray) -> int:
        """
        Select random valid move.
        
        Args:
            board: Current game board state
            
        Returns:
            int: Selected move index
        """
        valids = self.game.getValidMoves(board, 1)
        valid_moves = np.where(valids == 1)[0]
        if len(valid_moves) == 0:
            raise ValueError("No valid moves available")
        return np.random.choice(valid_moves)

class HumanBeckPlayer:
    """Human player with Pygame mouse interface."""
    
    def __init__(self, game):
        """
        Initialize human player.
        
        Args:
            game: Game instance providing game mechanics
        """
        self.game = game
        self.SQUARE_SIZE = 100  # Pixel size of each board square

    def play(self, board: np.ndarray) -> int:
        """
        Get move from human via mouse input.
        
        Args:
            board: Current game board state
            
        Returns:
            int: Selected move index
            
        Raises:
            SystemExit: If pygame window is closed
        """
        valid_moves = self.game.getValidMoves(board, 1)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Convert mouse position to board coordinates
                    pos_x = event.pos[0]
                    pos_y = event.pos[1]
                    col = int(pos_x // self.SQUARE_SIZE)
                    row = int(pos_y // self.SQUARE_SIZE)
                    
                    # Validate move
                    if self._is_valid_position(row, col):
                        move = self.game.n * row + col
                        if valid_moves[move]:
                            return move
                            
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if board position is valid."""
        return (0 <= row < self.game.m) and (0 <= col < self.game.n)

class GreedyBeckPlayer:
    """Player that makes locally optimal moves."""
    
    def __init__(self, game):
        """
        Initialize greedy player.
        
        Args:
            game: Game instance providing game mechanics
        """
        self.game = game

    def play(self, board: np.ndarray) -> int:
        """
        Select move with highest immediate score.
        
        Args:
            board: Current game board state
            
        Returns:
            int: Selected move index
            
        Raises:
            ValueError: If no valid moves available
        """
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        
        for action in range(self.game.getActionSize()):
            if not valids[action]:
                continue
                
            next_board, _ = self.game.getNextState(board, 1, action)
            score = self.game.getGameEnded(next_board, 1)
            candidates.append((-score, action))
            
        if not candidates:
            raise ValueError("No valid moves available")
            
        candidates.sort()
        return candidates[0][1]

class NNPolicyPlayer:
    """Neural network player using policy network for decisions."""
    
    def __init__(self, game, nnet):
        """
        Initialize policy network player.
        
        Args:
            game: Game instance providing game mechanics
            nnet: Neural network instance for predictions
        """
        self.game = game
        self.nnet = nnet

    def play(self, board: np.ndarray) -> int:
        """
        Select move based on policy network predictions.
        
        Args:
            board: Current game board state
            
        Returns:
            int: Selected move index
            
        Raises:
            ValueError: If no valid moves available
        """
        valids = self.game.getValidMoves(board, 1)
        if not np.any(valids):
            raise ValueError("No valid moves available")
            
        policy, _ = self.nnet.predict(board)
        policy *= valids  # Mask invalid moves
        
        # Select randomly among moves with highest probability
        max_prob_moves = np.where(policy == np.max(policy))[0]
        return np.random.choice(max_prob_moves)

class NNValuePlayer:
    """Neural network player using value network for decisions."""
    
    def __init__(self, game, nnet):
        """
        Initialize value network player.
        
        Args:
            game: Game instance providing game mechanics
            nnet: Neural network instance for predictions
        """
        self.game = game
        self.nnet = nnet

    def play(self, board: np.ndarray, cur_player: int) -> Tuple[int, List[float]]:
        """
        Select move based on value network predictions.
        
        Args:
            board: Current game board state
            cur_player: Current player (1 or -1)
            
        Returns:
            tuple: (selected_move, value_predictions)
                - selected_move: Index of selected move
                - value_predictions: List of predicted values for each move
                
        Raises:
            ValueError: If no valid moves available
        """
        valids = self.game.getValidMoves(board, 1)
        if not np.any(valids):
            raise ValueError("No valid moves available")
            
        value_predictions = []
        LARGE_VALUE = 1000  # Value for invalid moves
        
        # Evaluate each valid move
        for action, is_valid in enumerate(valids):
            if is_valid:
                next_board, next_player = self.game.getNextState(board, cur_player, action)
                # Get canonical board for prediction
                canonical_board = next_board * next_player
                _, value = self.nnet.predict(canonical_board)
                value_predictions.append(float(value))
            else:
                value_predictions.append(LARGE_VALUE)

        # Select randomly among moves with minimum value
        min_value_moves = np.where(value_predictions == np.min(value_predictions))[0]
        selected_move = np.random.choice(min_value_moves)
        
        return selected_move, value_predictions