#!/usr/bin/env python3
"""
Game Tournament Manager

This module manages AI and human player tournaments for a board game, supporting:
- AI vs AI round-robin tournaments
- Human vs AI games
- Game state tracking and analysis

The module handles different player types including neural networks, random players,
greedy players, and human players.
"""

import os
import datetime
import argparse
import multiprocessing as mp
from typing import Dict, List, Tuple, Callable, Optional, Union
from random import shuffle
from tqdm import tqdm

import numpy as np
import pandas as pd
from keras import backend as K

# Custom imports
from arena import Arena
from mcts import MCTS
from utils import *
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer, RandomPlayer, GreedyBeckPlayer
from cog_related import cog_value_net as cvn

# Configuration
CONFIG = {
    'participants_dir': '/scratch/zz737/fiar/tournaments/tournament_1',
    'moves_dir': '/scratch/zz737/fiar/tournaments/tournament_3/moves/raw/'
}

def get_participants() -> Dict[str, List[int]]:
    """
    Get sorted list of participants and their training iterations.
    
    Returns:
        Dictionary mapping participant names to their available iterations
    """
    participants = sorted([x[12:] for x in os.listdir(CONFIG['participants_dir'])])
    
    return {p: sorted(list(set([
        int(x.split('.')[0].split('_')[1])
        for x in os.listdir(os.path.join(CONFIG['participants_dir'], 'checkpoints_' + p))
        if x.startswith('checkpoint') and x != 'checkpoint' and not x.endswith('examples')
    ]))) for p in participants}

def get_args(participants_dir: str, participant: str, ix: Union[str, int], 
             mcts_sims: int, cpuct: float) -> dotdict:
    """
    Generate arguments for player initialization.
    
    Args:
        participants_dir: Directory containing player checkpoints
        participant: Player identifier
        ix: Iteration index or special value ('best', 'temp')
        mcts_sims: Number of Monte Carlo Tree Search simulations
        cpuct: Exploration constant for MCTS
        
    Returns:
        dotdict containing player configuration
    """
    direct_participants_dir = 'checkpoints' in participants_dir
    load_folder = participants_dir if direct_participants_dir else \
                 os.path.join(participants_dir, 'checkpoints_' + participant)
    
    fn = f'checkpoint_{ix}.pth.tar' if str(ix).isnumeric() else f'{ix}.pth.tar'
    
    return dotdict({
        'tempThreshold': 15,
        'numMCTSSims': mcts_sims,
        'cpuct': cpuct,
        'load_model': True,
        'load_folder_file': (load_folder, fn),
    })

def get_player(game: Game, participants_dir: str, participant_iter: str, 
               is_return_mcts: bool = False) -> Tuple[Callable, Optional[nn], Optional[MCTS]]:
    """
    Initialize a player based on the specified type and parameters.
    
    Args:
        game: Game instance
        participants_dir: Directory containing player checkpoints
        participant_iter: Player iteration identifier
        is_return_mcts: Whether to return the MCTS instance
        
    Returns:
        Tuple of (player function, neural network, optional MCTS instance)
    """
    # Handle special player types
    special_players = {
        'human': (HumanBeckPlayer(game), None),
        'random': (RandomPlayer(game), None),
        'greedy': (GreedyBeckPlayer(game), None)
    }
    
    if participant_iter in special_players:
        player, val_func = special_players[participant_iter]
        return lambda x: player.play(x), val_func
    
    # Handle AI players
    name_split = participant_iter.split(';')
    participant, ix = name_split[:2]
    mcts_sims = int(participant.split('_')[0][4:])
    cpuct = int(participant.split('_')[1][5:])
    
    args = get_args(participants_dir, participant, ix, mcts_sims, cpuct)
    nnet = NNetWrapper(game)
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    
    nmcts = MCTS(game, nnet, args)
    
    # Handle cognitive model
    if len(name_split) > 2 and name_split[2] == 'cog':
        w = np.array([0.1, 1/3, 1/3, 5, 10]) * 10
        C = 0.1
        cv_net = cvn.NNetWrapper(game, nnet, [w, C])
        cvnmcts = MCTS(game, cv_net, args)
        return lambda x: np.argmax(cvnmcts.getActionProb(x, temp=0)), cv_net
    
    return (lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), nnet, nmcts) if is_return_mcts else \
           (lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), nnet)