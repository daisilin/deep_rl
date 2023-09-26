import logging

import coloredlogs

from coach import Coach
import coach_no_reject as cnr
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_nnet import NNetWrapper_color as nnc
from utils import *
import sys
import supervised_learning as sl

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# checkpoint = '/scratch/zz737/fiar/run-3752918/scratch/zz737/fiar/tournaments/tournament_4/checkpoints_mcts100_cpuct2_id3/'
loaded_iter = None#-1
# checkpoint = '/scratch/zz737/fiar/tournaments/tournament_5/checkpoints_mcts100_cpuct2_id_1/'
# checkpoint = '/scratch/zz737/fiar/tournaments/tournament_6/checkpoints_mcts100_cpuct2_id_1/'
# checkpoint = '/scratch/xl1005/fiar/tournaments/tournament_17/checkpoints_mcts100_cpuct2_id_1/'

n_res=3
TRACK_COLOR = False#True
continuous_training =True #False
tempswitch = True
c_puct = 2
c_puct_str = f'{c_puct:.0e}' if c_puct < 1 else str(c_puct)
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_8/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_9/checkpoints_mcts100_cpuct2_id_res{n_res}-1/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_11/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_12/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_13/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_10/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = '/scratch/zz737/fiar/tournaments/tournament_7/checkpoints_mcts100_cpuct2_id_1/'
# checkpoint = 'E:\\Sam_data\\fiar\\tournaments\\tournament_1_win\\checkpoints_mcts100_cpuct2_id_0\\'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_14/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_15/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_16/checkpoints_mcts100_cpuct2_id_res{n_res}-0/'
# checkpoint = f'/scratch/zz737/fiar/tournaments/tournament_8/checkpoints_mcts100_cpuct{c_puct_str}_id_res{n_res}-1/'
# checkpoint = f'/scratch/xl1005/deep-master/tournaments/tournament_17/checkpoints_mcts100_cpuct2_id_res9-1/'
# checkpoint = f'/scratch/xl1005/deep-master/tournaments/tournament_18/checkpoints_mcts100_cpuct{c_puct_str}_id_res{n_res}-1/'
# checkpoint = f'/scratch/xl1005/deep-master/tournaments/tournament_19/checkpoints_mcts100_cpuct{c_puct_str}_id_res{n_res}-6/'
checkpoint = f'/scratch/xl1005/deep-master/tournaments/tournament_21/checkpoints_mcts100_cpuct{c_puct_str}_id_res{n_res}-1/'


args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15 if tempswitch else 40,#15
    'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 30,#2,#60,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': c_puct,#2,

    # 'checkpoint': './scratch/zz737/fiar/tournaments/tournament_4/checkpoints_mcts100_cpuct2_id10/',
    'checkpoint': checkpoint,
    'load_model': False,#True,#False,
    'load_folder_file': (checkpoint,'best.pth.tar'),
    # 'load_folder_file': (checkpoint,f'checkpoint_{loaded_iter}.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'loaded_iter': loaded_iter, #[SZ] by default -1, if loading a checkpoint, use that iter 
    'w_count':1, # by default 1, using visit count; if 0, use value
    'flip_color':False, # by default False,
    'dir_alpha':0.3,#0.3,#0.03,
    'epsilon':0.25,#0.25,#0.25,
})


def main(testmode=0):
    '''
    if testmode==1, then do testing
    '''
    log.info('Loading %s...', Game.__name__)
    g = Game(4, 9, 4)

    log.info('Loading %s...', nn.__name__)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
        log.info('making dir %s', args.checkpoint)
    
    # using resnet
    # onet = sl.OthelloNNet_resnet(g,sl.get_args(n_res=n_res, epochs=10, num_channels=256))
    # onet = sl.OthelloNNet_resnet(g,sl.get_args(n_res=n_res, epochs=10, num_channels=256, track_color=True))
    onet = sl.OthelloNNet_resnet(g,sl.get_args(n_res=n_res, epochs=10, num_channels=256, track_color=TRACK_COLOR))
    if 'track_color' in onet.args.keys() and onet.args.track_color: #[SZ] for tracking color
        nnet = nnc(g,nnet=onet,args=onet.args)
    else:
        nnet = nn(g,nnet=onet,args=onet.args)
    # nnet = nn(g)

    if testmode:
        nnet.args.epochs=5 # just train once

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    if testmode:
        args.numEps = 3
        args.numIters = 5
        args.arenaCompare = 2
        args.numMCTSSims = 2
    
    if continuous_training:
        c = cnr.Coach(g, nnet, args)
    else:
        c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main_args = sys.argv[1:]
    print(main_args)
    main(int(main_args[0]))