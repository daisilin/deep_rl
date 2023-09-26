import logging

import coloredlogs

from coach import Coach
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from utils import *



log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

import argparse
my_parser = argparse.ArgumentParser(description='optional args to training the NN')

my_parser.add_argument('-ch',
                       '--checkpoint',
                       type=str,
                       help='folder to load model')

my_parser.add_argument('-li',
                       '--loaded_iter',
                       type=int,
                       help='iter of the loaded model', default=-1)

my_parser.add_argument('-lm',
                       '--load_model',
                       type=bool,
                       help='whether to load checkpoint', default=False)


def default_args(loaded_iter=None, checkpoint=None, **kwargs):
    if checkpoint is None:
        checkpoint = '/scratch/zz737/fiar/run-3752918/scratch/zz737/fiar/tournaments/tournament_4/checkpoints_mcts100_cpuct2_id3/'
    if loaded_iter is None:
        loaded_iter = 49
    args = dotdict({
        'numIters': 1000,
        'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
        'arenaCompare': 60,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 2,

        # 'checkpoint': './scratch/zz737/fiar/tournaments/tournament_4/checkpoints_mcts100_cpuct2_id10/',
        'checkpoint': checkpoint,
        'load_model': False,
        # 'load_folder_file': (checkpoint,'best.pth.tar'),
        'load_folder_file': (checkpoint,f'checkpoint_{loaded_iter}.pth.tar'), # might cause issue in coach.loadTrainExamples, but that one is not used anyway.
        'numItersForTrainExamplesHistory': 20,

        'loaded_iter': loaded_iter, #[SZ] by default -1, if loading a checkpoint, use that iter 

    })
    return args


def main(args):
    print('args.load_folder_file')
    print(args.load_folder_file)
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

    args_in = vars(my_parser.parse_args())
    print('args_in\n')
    print(args_in)
    args = default_args(**args_in)
    args.update(args_in)
    # print(args.checkpoint)
    main(args)