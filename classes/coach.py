"""
AlphaZero Training Coach Implementation.

This module implements the training loop for AlphaZero-style learning through self-play.
It handles the process of generating training examples through self-play, training the
neural network, and evaluating new network versions against previous ones.
"""

import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from typing import List, Tuple, Optional, Union, Any, Deque

import numpy as np
import pandas as pd
from tqdm import tqdm

from arena import Arena
from mcts import MCTS, MCTS_color

log = logging.getLogger(__name__)

def save_new_vs_old_record(old_i: int, 
                          new_i: int, 
                          savedir: str, 
                          nwins: int, 
                          pwins: int, 
                          draws: int, 
                          naccepted: bool) -> pd.DataFrame:
    """Save the results of pitting new network version against old version.
    
    Args:
        old_i: Index of old model version
        new_i: Index of new model version
        savedir: Directory to save records
        nwins: Number of wins by new version
        pwins: Number of wins by old version
        draws: Number of draws
        naccepted: Whether new version was accepted
        
    Returns:
        DataFrame containing all historical records
    """
    record_fn = os.path.join(savedir, 'old_vs_new_record.csv')
    try:
        record_df = pd.read_csv(record_fn)
    except FileNotFoundError:
        record_df = pd.DataFrame(columns=['prev', 'new', 'nwins', 'pwins', 'draws', 'naccepted'])
    
    one_record = pd.DataFrame([[old_i, new_i, nwins, pwins, draws, naccepted]], 
                             columns=['prev', 'new', 'nwins', 'pwins', 'draws', 'naccepted'])
    record_df = pd.concat([record_df, one_record], axis=0, ignore_index=True)
    record_df.to_csv(record_fn, index=False)
    return record_df

class Coach:
    """Implements the training process for AlphaZero learning through self-play."""

    def __init__(self, game: Any, nnet: Any, args: Any):
        """Initialize the Coach.
        
        Args:
            game: Game implementation providing game rules and mechanics
            nnet: Neural network wrapper providing policy and value predictions
            args: Configuration arguments for training process
        """
        self.game = game
        self.nnet = nnet
        # Initialize competitor network with same architecture
        self.pnet = self.nnet.__class__(
            self.game, 
            nnet=self.nnet.nnet.__class__(self.game, self.nnet.nnet.args)
        )
        self.args = args
        
        # Setup color tracking if enabled
        self.track_color = False
        if 'track_color' in self.nnet.args.keys() and self.nnet.args.track_color:
            self.track_color = True
            
        # Initialize MCTS based on color tracking setting
        self.mcts = MCTS_color(self.game, self.nnet, self.args) if self.track_color else MCTS(self.game, self.nnet, self.args)
        
        # Setup color flipping
        self.flip_color = self.args.flip_color if 'flip_color' in self.args.keys() else False
        
        # Training history and state
        self.trainExamplesHistory: List[Deque] = []
        self.skipFirstSelfPlay = False
        
        # Setup Dirichlet noise parameters
        self.args['dir_alpha'] = self.args.get('dir_alpha', 0.03)
        self.args['epsilon'] = self.args.get('epsilon', 0.25)
        
        self.best_i = 0

    def executeEpisode(self) -> List[Tuple]:
        """Execute one episode of self-play and generate training examples.
        
        Returns:
            List of training examples of the form (board, policy, value, [color])
            where color is only included if color tracking is enabled.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            color = int(episodeStep % 2 == 1)
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold) * 2

            # Get move probabilities from MCTS
            if self.track_color:
                pi = self.mcts.getActionProb(
                    canonicalBoard, 
                    color, 
                    temp=temp,
                    dir_alpha=self.args.dir_alpha,
                    epsilon=self.args.epsilon
                )
            else:
                pi = self.mcts.getActionProb(
                    canonicalBoard,
                    temp=temp, 
                    dir_alpha=self.args.dir_alpha,
                    epsilon=self.args.epsilon
                )

            # Add symmetrical positions to training examples
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                example = [b, self.curPlayer, p, None, color] if self.track_color else [b, self.curPlayer, p, None]
                trainExamples.append(example)

            # Make move
            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            # Check if game is ended
            gameResult = self.game.getGameEnded(board, self.curPlayer)
            if gameResult != 0:
                # Prepare final training examples with correct values
                if self.track_color:
                    return [(x[0], x[2], gameResult * ((-1) ** (x[1] != self.curPlayer)), x[-1]) 
                            for x in trainExamples]
                else:
                    return [(x[0], x[2], gameResult * ((-1) ** (x[1] != self.curPlayer))) 
                            for x in trainExamples]

    def learn(self):
        """Execute the main training loop.
        
        Performs multiple iterations of self-play, network training, and evaluation.
        In each iteration:
        1. Generates new games through self-play
        2. Trains neural network on accumulated examples
        3. Evaluates new network against previous version
        4. Accepts or rejects new network based on performance
        """
        n_available_actions = self.game.getActionSize()

        # Save network architecture arguments
        args_fn = os.path.join(self.args.checkpoint, 'args.p')
        with open(args_fn, "wb") as f:
            Pickler(f).dump(self.nnet.args)

        for i in range(1, self.args.numIters + 1):
            log.info(f'Starting Iter #{i} ...')

            # Generate new training examples through self-play
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    # Reset MCTS for each episode
                    if self.track_color:
                        self.mcts = MCTS_color(self.game, self.nnet, self.args)
                    else:
                        self.mcts = MCTS(self.game, self.nnet, self.args)
                    iterationTrainExamples += self.executeEpisode()

                # Update training history
                self.trainExamplesHistory.append(iterationTrainExamples)

            # Manage training history size
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. "
                    f"len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
                )
                self.trainExamplesHistory.pop(0)

            # Save current training examples
            self.saveTrainExamples(i - 1)

            # Prepare training data
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # Save current network as competitor
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Initialize MCTS for both networks
            if self.track_color:
                pmcts = MCTS_color(self.game, self.pnet, self.args)
                nmcts = MCTS_color(self.game, self.nnet, self.args)
            else:
                pmcts = MCTS(self.game, self.pnet, self.args)
                nmcts = MCTS(self.game, self.nnet, self.args)

            # Train new network version
            self.nnet.train(trainExamples)

            # Setup action functions for arena
            if self.track_color:
                ai_p = lambda x, c: np.random.choice(
                    np.arange(n_available_actions),
                    p=pmcts.getActionProb(x, c, temp=1/10)
                )
                ai_n = lambda x, c: np.random.choice(
                    np.arange(n_available_actions),
                    p=nmcts.getActionProb(x, c, temp=1/10)
                )
                arena = Arena(
                    ai_p, ai_n, self.game,
                    tree1=pmcts, tree2=nmcts,
                    track_color=[True, True],
                    flip_color=self.flip_color
                )
            else:
                ai_p = lambda x: np.random.choice(
                    np.arange(n_available_actions),
                    p=pmcts.getActionProb(x, temp=1/10)
                )
                ai_n = lambda x: np.random.choice(
                    np.arange(n_available_actions),
                    p=nmcts.getActionProb(x, temp=1/10)
                )
                arena = Arena(ai_p, ai_n, self.game, tree1=pmcts, tree2=nmcts)

            # Evaluate new network
            log.info('PITTING AGAINST PREVIOUS VERSION')
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, is_save_moves=True)
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

            # Accept or reject new network
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                naccepted = False
                prev_best_i = self.best_i
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint,
                    filename=self.getCheckpointFile(i)
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint,
                    filename='best.pth.tar'
                )
                naccepted = True
                prev_best_i = self.best_i
                self.best_i = i

            # Save evaluation results
            record_df = save_new_vs_old_record(
                prev_best_i, i, self.args.checkpoint,
                nwins, pwins, draws, naccepted
            )

    def getCheckpointFile(self, iteration: int) -> str:
        """Get filename for saving network checkpoint.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Filename for checkpoint
        """
        if self.args.loaded_iter is not None:
            iteration = iteration + self.args.loaded_iter
        return f'checkpoint_{iteration}.pth.tar'

    def saveTrainExamples(self, iteration: int):
        """Save training examples to file.
        
        Args:
            iteration: Current training iteration
        """
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        """Load training examples from file.
        
        Prompts for confirmation if file is not found.
        """
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')
            self.skipFirstSelfPlay = True  # Skip first self-play as examples already exist