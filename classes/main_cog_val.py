'''
train cog model value + mcts
'''
import logging

import coloredlogs

from coach import Coach
from beck.beck_game import BeckGame as Game
# from beck.beck_nnet import NNetWrapper as nn
from cog_related.cog_value_net_trainable_mcts import NNetWrapper_cog as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

checkpoint = '/scratch/zz737/fiar/models/cog_model_value_mcts/copy1/'
loaded_iter = 0

args = dotdict({
    'numIters': 1000,#1000,
    'numEps': 100,#100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,#0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,#100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 60,#60,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 2,

    # 'checkpoint': './scratch/zz737/fiar/tournaments/tournament_4/checkpoints_mcts100_cpuct2_id10/',
    'checkpoint': checkpoint,
    'load_model': False,#False,
    'load_folder_file': (checkpoint,'best.pth.tar'),
    # 'load_folder_file': (checkpoint,f'checkpoint_{loaded_iter}.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'loaded_iter': loaded_iter, #[SZ] by default -1 (0 ? ), if loading a checkpoint, use that iter 

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(4, 9, 4)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()