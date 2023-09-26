import logging
import numpy as np

import coloredlogs

from arena import Arena
from coach import Coach
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer, NNPolicyPlayer, NNValuePlayer
from mcts import MCTS
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

numMCTSSims = 100
cpuct = 2
model_class_name = f'checkpoints_mcts{numMCTSSims}_cpuct{cpuct}'
model_instance_name = 'checkpoint_70'
temp = 0
# model_dir = '/scratch/zz737/fiar/tournaments/tournament_1/'
model_dir = '/scratch/zz737/fiar/tournaments/tournament_4/checkpoints_mcts100_cpuct2_id-37549660/'

args = dotdict({
    'display':True,
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'temp':temp,
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': numMCTSSims,#100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': cpuct,

    'checkpoint': './temp/',
    'load_model': True,
    # 'load_folder_file': (f'../models/{model_class_name}',f'{model_instance_name}.pth.tar'),
    # 'load_folder_file': (f'{model_dir}/{model_class_name}',f'{model_instance_name}.pth.tar'),
    'load_folder_file': (f'{model_dir}',f'{model_instance_name}.pth.tar'),

    'numItersForTrainExamplesHistory': 20,

    # for saving moves
    'is_save_moves': False,
    'save_moves_folder': f'../moves/{model_class_name}',
    'save_moves_file': f'{model_instance_name}_temp{temp}.csv', 
    'overwrite': True, # whether overwrite the existing csv file or add to it

    #
    'numGames':2,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(4, 9, 4)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    nnet2 = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        nnet2.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        raise ValueError('Need a checkpoint!')


    nmcts = MCTS(g, nnet, args)

    nmcts2 = MCTS(g, nnet2, args)

    allActions = np.arange(g.getActionSize())

    ai = lambda x: np.argmax(nmcts.getActionProb(x, temp=0)) #temp=0
    # ai = lambda x: np.random.choice(allActions, p=nmcts.getActionProb(x, temp=args.temp))

    # ai2 = lambda x: np.argmax(nmcts2.getActionProb(x, temp=0)) #temp=0
    # ai2 = lambda x: np.random.choice(allActions, p=nmcts.getActionProb(x, temp=args.temp))

    human_player = HumanBeckPlayer(g)
    human = lambda x: human_player.play(x)

    # ai_nonmcts_player = NNPolicyPlayer(g, nnet2)
    # ai_nonmcts = lambda x:ai_nonmcts_player.play(x)
    ai_nonmcts_player = NNValuePlayer(g,nnet)
    ai_nonmcts = lambda x:ai_nonmcts_player.play(x,-1)[0] #return (action, list of values)

    # arena = Arena(ai, human, g, display=g.display)
    arena = Arena(human, ai, g, display=g.display)
    # arena = Arena(human, ai_nonmcts, g, display=g.display)

    # arena = Arena(ai, ai2, g, display=g.display)
    # arena = Arena(ai, ai2, g, display=args.display)

    log.info('Starting the game 🎉')
    if args.is_save_moves:
        gameResult, moves_result = arena.playGame(verbose=args.display, nnet=nnet, is_save_moves=True)
        # oneWon, twoWon, draws, moves_result_multigame = arena.playGames(args.numGames,verbose=args.display, nnet=[nnet,nnet2], is_save_moves=True)
        moves_result_multigame = moves_result
        save_moves(moves_result_multigame,tosave='both',subjectID=None,model_class=model_class_name, model_instance=model_instance_name, temp=temp,
                fd = '../models/moves/')
    else:
        arena.playGame(verbose=args.display, nnet=nnet, is_save_moves=False)
    

    
    



if __name__ == "__main__":
    main()
