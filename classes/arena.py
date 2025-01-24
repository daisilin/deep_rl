"""
Game Arena for AI Agent Competition.

This module implements an Arena where two AI agents can compete against each other
in a game. It supports various game types and can track/save game moves and results.
The Arena is particularly designed for AlphaZero-style game playing agents.
"""

import logging
import sys
from copy import copy
from typing import List, Tuple, Optional, Union, Any

import numpy as np
import pygame
from tqdm import tqdm

from utils import *

# Configure logging
log = logging.getLogger(__name__)

def refresh(tree1: Optional[Any], tree2: Optional[Any]) -> None:
    """Refresh the MCTS trees for both players.
    
    Args:
        tree1: MCTS tree for player 1
        tree2: MCTS tree for player 2
    """
    for tree in [tree1, tree2]:
        if tree is not None:
            tree.refresh()

class Arena:
    """An Arena class where two agents can compete against each other."""

    def __init__(self, 
                 player1: callable, 
                 player2: callable, 
                 game: Any,
                 tree1: Optional[Any] = None,
                 tree2: Optional[Any] = None,
                 display: Optional[callable] = None,
                 track_color: List[bool] = [False, False],
                 flip_color: bool = False,
                 game_num: Optional[int] = None):
        """Initialize the Arena.
        
        Args:
            player1: First player's action function (takes board as input, returns action)
            player2: Second player's action function (takes board as input, returns action)
            game: Game object that implements the game rules and mechanics
            tree1: MCTS tree for player 1 (optional)
            tree2: MCTS tree for player 2 (optional)
            display: Display function for rendering the game state (optional)
            track_color: Whether to track colors for each player [p1_track, p2_track]
            flip_color: Whether to flip the color tracking
            game_num: Game number identifier (optional)
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.tree1 = tree1
        self.tree2 = tree2
        self.track_color = track_color
        self.flip_color = flip_color
        self.game_num = game_num

    def playGameSave(self,
                    verbose: bool = False,
                    nnet: Optional[Any] = None,
                    subjectID_l: List[str] = ['default1', 'default2'],
                    fd: str = '../models/moves/') -> int:
        """Play and save a single game.
        
        Args:
            verbose: Whether to display game progress
            nnet: Neural network for value prediction (optional)
            subjectID_l: List of subject IDs for saving moves
            fd: Directory path for saving moves
            
        Returns:
            Game result (1 for player1 win, -1 for player2 win, 0 for draw)
        """
        win_result, moves_result = self.playGame(verbose=verbose, nnet=nnet, is_save_moves=True)
        for i in range(2):
            save_moves(moves_result, 
                      tosave=i,
                      subjectID=subjectID_l[i],
                      model_class=None,
                      model_instance=None,
                      temp=None,
                      fd=fd)
        return win_result

    def playGame(self,
                verbose: bool = False,
                nnet: Optional[Any] = None,
                is_save_moves: bool = False,
                initboard: Optional[np.ndarray] = None,
                game_num: Optional[int] = None) -> Union[int, Tuple[int, List]]:
        """Execute one episode of a game.
        
        Args:
            verbose: Whether to display game progress
            nnet: Neural network for value prediction (optional)
            is_save_moves: Whether to save game moves
            initboard: Initial board state (optional)
            game_num: Game number identifier (optional)
            
        Returns:
            If is_save_moves=False:
                Game result (1 for player1 win, -1 for player2 win, 0 for draw)
            If is_save_moves=True:
                Tuple of (game_result, moves_data)
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard(initboard=initboard)
        it = 0

        # Initialize move tracking
        moves_list = [[], []]  # For p1(black), p2(white)
        pieces_list = [[], []]  # History of piece positions
        pieces = [0, 0]  # Current piece positions
        value_list = [[], []]  # Value predictions if using neural network

        # Main game loop
        while self.game.getGameEnded(board, curPlayer) == 0:
            curP_ind = int(0.5 - curPlayer/2)
            it += 1

            # Display game state if requested
            if verbose:
                assert self.display
                self.display(board, it, curPlayer, self.game_num, gameover=False)
            
            # Get player action
            if self.track_color[curP_ind]:
                curP_color = np.abs(1 - curP_ind)  # 1 for black
                if self.flip_color:
                    curP_color = np.abs(1 - curP_color)  # 0 for black
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer), curP_color)
            else:
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            # Validate move
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Record move
            encoded_move = 2**action
            moves_list[curP_ind].append(encoded_move)
            pieces_list[curP_ind].append(copy(pieces))
            pieces[curP_ind] += encoded_move

            # Update game state
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            print(board)

        # Handle game end
        if verbose:
            assert self.display
            game_result_txt = self.game.getGameEnded(board, 1)
            print(game_result_txt)
            self.display(board, it, game_result_txt, self.game_num, gameover=True)
            pygame.time.wait(3000)

        win_result = curPlayer * self.game.getGameEnded(board, curPlayer)

        if not is_save_moves:
            return win_result

        # Prepare move history if saving is requested
        moves_result = []
        for player in range(2):  # 0 and 1, corresponding to 1 and -1, black and white
            pieces_list_1p = np.array(pieces_list[player]).reshape(-1, 1)
            nmoves = pieces_list_1p.shape[0]
            color_list = np.ones((nmoves, 1)) * player
            moves_list_1p = np.array(moves_list[player]).reshape(-1, 1)
            value_list_1p = np.array(value_list[player]).reshape(-1, 1)
            
            moves_result_1p = np.hstack([
                pieces_list_1p,
                color_list,
                moves_list_1p,
                value_list_1p
            ])
            moves_result.append(moves_result_1p)

        return win_result, moves_result

    def playGames(self,
                 num: int,
                 verbose: bool = False,
                 nnet: Optional[Any] = None,
                 is_save_moves: bool = False) -> Union[Tuple[int, int, int], 
                                                     Tuple[int, int, int, List]]:
        """Play multiple games, alternating who plays first.
        
        Args:
            num: Number of total games to play (will be divided by 2)
            verbose: Whether to display game progress
            nnet: Neural network for value prediction (optional)
            is_save_moves: Whether to save game moves
            
        Returns:
            If is_save_moves=False:
                Tuple of (player1_wins, player2_wins, draws)
            If is_save_moves=True:
                Tuple of (player1_wins, player2_wins, draws, moves_data)
        """
        num = int(num / 2)
        oneWon = twoWon = draws = 0
        moves_result_multigame = [[], []]  # For p1 and p2 moves across games

        # First half of games with player1 starting
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            if is_save_moves:
                gameResult, moves_result = self.playGame(
                    verbose=verbose, 
                    nnet=nnet, 
                    is_save_moves=True
                )
                moves_result_multigame[0].append(moves_result[0])
                moves_result_multigame[1].append(moves_result[1])
            else:
                gameResult = self.playGame(verbose=verbose, is_save_moves=False)

            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

            refresh(self.tree1, self.tree2)

        # Swap players for second half
        self.player1, self.player2 = self.player2, self.player1
        self.track_color = [self.track_color[1], self.track_color[0]]

        # Second half of games with player2 (originally) starting
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            if is_save_moves:
                gameResult, moves_result = self.playGame(
                    verbose=verbose,
                    nnet=nnet,
                    is_save_moves=True
                )
                moves_result_multigame[0].append(moves_result[1])  # Colors reversed
                moves_result_multigame[1].append(moves_result[0])
            else:
                gameResult = self.playGame(verbose=verbose, is_save_moves=False)

            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

            refresh(self.tree1, self.tree2)

        if not is_save_moves:
            return oneWon, twoWon, draws

        # Stack all game results if saving moves
        moves_result_multigame[0] = np.vstack(moves_result_multigame[0])
        moves_result_multigame[1] = np.vstack(moves_result_multigame[1])
        return oneWon, twoWon, draws, moves_result_multigame