"""
Main training script for AlphaZero implementation.

This script configures and launches the training process for an AlphaZero-style
AI to play the game of Four-in-a-Row (also known as Connect Four).
"""

import logging
import os
import sys
from typing import Optional

import coloredlogs

from coach import Coach
import coach_no_reject as cnr
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_nnet import NNetWrapper_color as nnc
import supervised_learning as sl
from utils import dotdict

# Configure logging
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change to DEBUG for more info

# Training configuration
N_RES_BLOCKS = 3
TRACK_COLOR = False
CONTINUOUS_TRAINING = True
TEMP_SWITCH = True
C_PUCT = 2
C_PUCT_STR = f'{C_PUCT:.0e}' if C_PUCT < 1 else str(C_PUCT)

# Checkpoint configuration
LOADED_ITER: Optional[int] = None
CHECKPOINT_PATH = f'/scratch/xl1005/deep-master/tournaments/tournament_21/checkpoints_mcts100_cpuct{C_PUCT_STR}_id_res{N_RES_BLOCKS}-1/'

# Training arguments
TRAINING_ARGS = dotdict({
    'numIters': 1000,                    # Total number of training iterations
    'numEps': 100,                       # Number of self-play games per iteration
    'tempThreshold': 15 if TEMP_SWITCH else 40,  # Temperature threshold for move selection
    'updateThreshold': 0.51,             # Win ratio threshold for accepting new network
    'maxlenOfQueue': 200000,             # Max number of game examples to store
    'numMCTSSims': 100,                  # Number of MCTS simulations per move
    'arenaCompare': 30,                  # Number of evaluation games
    'cpuct': C_PUCT,                     # Exploration constant in MCTS
    'checkpoint': CHECKPOINT_PATH,        # Path to save/load models
    'load_model': False,                 # Whether to load existing model
    'load_folder_file': (CHECKPOINT_PATH, 'best.pth.tar'),  # Model to load
    'numItersForTrainExamplesHistory': 20,  # Number of iterations of history to keep
    'loaded_iter': LOADED_ITER,          # Starting iteration number if loading model
    'w_count': 1,                        # Use visit count (1) or value (0)
    'flip_color': False,                 # Whether to flip colors
    'dir_alpha': 0.3,                    # Dirichlet noise alpha parameter
    'epsilon': 0.25,                     # Epsilon for noise in MCTS
})

def main(test_mode: int = 0):
    """Initialize and run the training process.
    
    Args:
        test_mode: If 1, runs with reduced parameters for testing
    """
    # Initialize game
    log.info('Loading %s...', Game.__name__)
    game = Game(4, 9, 4)  # 4x9 board with 4-in-a-row win condition

    log.info('Loading %s...', nn.__name__)

    # Create checkpoint directory if needed
    if not os.path.exists(TRAINING_ARGS.checkpoint):
        os.makedirs(TRAINING_ARGS.checkpoint)
        log.info('Making directory %s', TRAINING_ARGS.checkpoint)
    
    # Initialize neural network
    nnet_args = sl.get_args(
        n_res=N_RES_BLOCKS,
        epochs=10,
        num_channels=256,
        track_color=TRACK_COLOR
    )
    base_network = sl.OthelloNNet_resnet(game, nnet_args)
    
    # Select appropriate wrapper based on color tracking
    if 'track_color' in base_network.args.keys() and base_network.args.track_color:
        nnet = nnc(game, nnet=base_network, args=base_network.args)
    else:
        nnet = nn(game, nnet=base_network, args=base_network.args)

    # Adjust parameters for test mode
    if test_mode:
        nnet.args.epochs = 5  # Reduced training epochs
        TRAINING_ARGS.numEps = 3
        TRAINING_ARGS.numIters = 5
        TRAINING_ARGS.arenaCompare = 2
        TRAINING_ARGS.numMCTSSims = 2

    # Load existing model if specified
    if TRAINING_ARGS.load_model:
        log.info('Loading checkpoint "%s/%s"...',
                 TRAINING_ARGS.load_folder_file[0],
                 TRAINING_ARGS.load_folder_file[1])
        nnet.load_checkpoint(*TRAINING_ARGS.load_folder_file)
    else:
        log.warning('Not loading a checkpoint!')

    # Initialize appropriate coach
    log.info('Loading the Coach...')
    coach_class = cnr.Coach if CONTINUOUS_TRAINING else Coach
    coach = coach_class(game, nnet, TRAINING_ARGS)

    # Load training examples if continuing from checkpoint
    if TRAINING_ARGS.load_model:
        log.info("Loading 'trainExamples' from file...")
        coach.loadTrainExamples()

    # Start training process
    log.info('Starting the learning process 🎉')
    coach.learn()

if __name__ == "__main__":
    test_mode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(test_mode)