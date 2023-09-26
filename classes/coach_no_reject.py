import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import pandas as pd

import numpy as np
from tqdm import tqdm

from arena import Arena
from mcts import MCTS, MCTS_color

log = logging.getLogger(__name__)


def save_new_vs_old_record(old_i, new_i, savedir, nwins, pwins, draws, naccepted):
    record_fn = os.path.join(savedir,'old_vs_new_record.csv')
    try:
        record_df = pd.read_csv(record_fn)
    except:
        record_df=pd.DataFrame(columns=['prev','new','nwins','pwins','draws','naccepted'])
    one_record = pd.DataFrame([[old_i,new_i,nwins,pwins,draws, naccepted]],columns=['prev','new','nwins','pwins','draws','naccepted'])
    record_df = pd.concat([record_df, one_record],axis=0,ignore_index=True)
    record_df.to_csv(record_fn,index=False)
    return record_df


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet # the net being trained
        # the evaluator net, kept to be current best
        self.pnet = self.nnet.__class__(self.game, nnet = self.nnet.nnet.__class__(self.game,self.nnet.nnet.args))  # the competitor network; to incoporate other nnet types like resnet; first nnet->NNetWrapper, second: othellonet 

        
        self.best_i = 0

        self.args = args
        if 'track_color' in self.nnet.args.keys() and self.nnet.args.track_color:
            self.track_color = True
        else:
            self.track_color = False

        if self.track_color:
            self.mcts = MCTS_color(self.game, self.pnet, self.args)
        else:
            self.mcts = MCTS(self.game, self.pnet, self.args)

        if 'flip_color' in self.args.keys(): # 0 for black, 1 for white
            self.flip_color = self.args.flip_color
        else:
            self.flip_color = False
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        # both below for dirichlet noise during self play
        if 'dir_alpha' not in self.args.keys():
            self.args['dir_alpha'] = 0.03
        if 'epsilon' not in self.args.keys():
            self.args['epsilon'] = 0.25

        # make sure the two networks start with the same params
        self.pnet.save_checkpoint(folder=self.args.checkpoint, filename='init.pth.tar')
        self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='init.pth.tar')
        assert np.sum(self.pnet.nnet.model.weights[0].numpy()!=self.nnet.nnet.model.weights[0].numpy())==0

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        

        while True:
            episodeStep += 1
            color = int(episodeStep%2==1)
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold) *2 
            # if temp==0: #[SZ] train with a small but non-zero temperature
            #     temp=1/20

            if self.track_color:
                pi = self.mcts.getActionProb(canonicalBoard, color, temp=temp, dir_alpha=self.args.dir_alpha, epsilon=self.args.epsilon)
            else:
                pi = self.mcts.getActionProb(canonicalBoard, temp=temp, dir_alpha=self.args.dir_alpha, epsilon=self.args.epsilon)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            if self.track_color:
                for b, p in sym:
                    trainExamples.append([b, self.curPlayer, p, None, color])

            else:
                for b, p in sym:
                    trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                #[SZ] possibility of tracking color
                if self.track_color:
                    return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)), x[-1]) for x in trainExamples] #black: episodestep odd; white even
                else:
                    return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        n_available_actions = self.game.getActionSize()

        # save args of the network; with args, a resnet can be reconstructed
        args_fn = os.path.join(self.args.checkpoint, 'args.p')
        with open(args_fn, "wb") as f:
            Pickler(f).dump(self.nnet.args)

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    if self.track_color:
                        self.mcts = MCTS_color(self.game, self.pnet, self.args)
                    else:
                        self.mcts = MCTS(self.game, self.pnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples) # self.trainExamplesHistory is a list of deques, each deque is examples (all the self-plays) from one iteration

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            if self.track_color:
                pmcts = MCTS_color(self.game, self.pnet, self.args)
            else:
                pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            if self.track_color:
                nmcts = MCTS_color(self.game, self.nnet, self.args)
            else:
                nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)

            if self.track_color: #t9
                ai_p = lambda x,c: np.random.choice(np.arange(n_available_actions),p=pmcts.getActionProb(x,c, temp=1/10)) # tournament 6
                ai_n = lambda x,c: np.random.choice(np.arange(n_available_actions),p=nmcts.getActionProb(x,c, temp=1/10)) 
                arena = Arena(ai_p, ai_n, self.game, tree1=pmcts, tree2=nmcts, track_color=[True,True], flip_color=self.flip_color)
            else:
                ai_p = lambda x: np.random.choice(np.arange(n_available_actions),p=pmcts.getActionProb(x, temp=1/10)) # tournament 6
                ai_n = lambda x: np.random.choice(np.arange(n_available_actions),p=nmcts.getActionProb(x, temp=1/10)) 
                # ai_p = lambda x: np.random.choice(np.arange(n_available_actions),p=pmcts.getActionProb(x, temp=0)) # tournament 7
                # ai_n = lambda x: np.random.choice(np.arange(n_available_actions),p=nmcts.getActionProb(x, temp=0))
                arena = Arena(ai_p, ai_n, self.game, tree1=pmcts, tree2=nmcts)
            

            pwins, nwins, draws = arena.playGames(self.args.arenaCompare,is_save_moves=True)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                # [SZ] save all models!
                # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                log.info('REJECTING NEW MODEL')
                # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

                naccepted = False
                prev_best_i = self.best_i

            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

                naccepted = True
                prev_best_i = self.best_i
                self.best_i = i

            record_df = save_new_vs_old_record(prev_best_i, i, self.args.checkpoint, nwins, pwins, draws, naccepted)

    def getCheckpointFile(self, iteration):
        if self.args.loaded_iter is not None:
            iteration = iteration + self.args.loaded_iter # [SZ] self.args.loaded_iter, the iter in the name of the loaded model; 
        
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
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

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True