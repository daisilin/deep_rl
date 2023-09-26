import datetime
import multiprocessing as mp
import numpy as np
import os
import random

from lazy_loader import LazyLoader
from memory import Memory
# from stage import execute_episode
from utils import printl, os_reseed
from mcts import MCTS

class TrainingProcess(mp.Process):
    def __init__(self, name, gpu_id, game, lazy_model_module, model_args, example_ingress, weight_egresses, kill_ingress, training_args, mcts_args, selfplay_args):
        mp.Process.__init__(self)
        self.name = name
        self.gpu_id = gpu_id
        self.game = game
        self.lazy_model_module = lazy_model_module
        self.model_args = model_args
        self.example_ingress = example_ingress
        self.weight_egresses = weight_egresses
        self.kill_ingress = kill_ingress
        self.training_args = training_args
        self.examples = Memory(self.training_args['MAX_MEMORY_SIZE'])
        self.mcts_args = mcts_args
        self.selfplay_args = selfplay_args

    def receive_examples(self):
        received_count = 0
        while (received_count < self.training_args['MAX_EXAMPLES_PER_RECEIVE']):
        # while not self.example_ingress.empty() and (received_count < self.training_args['MAX_EXAMPLES_PER_RECEIVE']):
            memories = self.example_ingress.get()
            self.examples.store_memories(memories)
            received_count += 1
        memory_usage, memory_capacity = self.examples.get_memory_usage()
        printl(f'{self.name}: Received examples. Memory usage {memory_usage}/{memory_capacity}.')

    def output_weights(self):
        for egress in self.weight_egresses:
            egress.put(self.model.get_weights())
        printl(f'{self.name}: Outputted weights to selfplay processes')

    def train(self):
        training_sample = self.examples.sample_memories(self.training_args['TRAINING_SAMPLE_SIZE'])
        self.model.train(training_sample,
            self.training_args['BATCH_SIZE'],
            self.training_args['EPOCHS'])
        self.generation += 1
        printl(f'{self.name}: Training for generation {self.generation} done')

    def execute_episode(self, players):
        searches = [MCTS(self.game, player, self.mcts_args) for player in players]
        state = self.game.get_initial_state()
        current_player = 1
        turn_count = 0

        while not self.game.get_is_terminal_state(state):
            canonical_state = self.game.get_canonical_form(state, current_player)
            canonical_state = self.game.get_canonical_form(state, current_player)
            if turn_count >= self.selfplay_args['DETERMINISTIC_PLAY']:
                probs = searches[current_player - 1].get_probs(canonical_state, temperature=1)
                action = np.argmax(probs)
            else:
                probs = searches[current_player - 1].get_probs(canonical_state, temperature=1)
                action = np.random.choice(range(len(probs)), p = probs)
                assert self.game.get_allowed_actions(state)[action], f'Must choose an allowed action!\nChose {action} in state\n{canonical_state}'
            state, current_player = self.game.get_next_state(state, current_player, action)

        result = self.game.get_result(state, current_player)
        return result

    def eval(self):
        games_with_rookie_first = int(self.training_args['EVAL_GAMES'] / 2)
        games_with_champ_first = self.training_args['EVAL_GAMES'] - games_with_rookie_first
        wins, draws, losses = 0, 0, 0
        for _ in range(games_with_rookie_first):
            result = self.execute_episode([self.rookie, self.champ])
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        for _ in range(games_with_champ_first):
            result = self.execute_episode([self.champ, self.rookie])
            if result == -1:
                wins += 1
            elif result == 1:
                losses += 1
            else:
                draws += 1
        
        if wins + losses == 0 or wins / (wins + losses) < self.training_args['PROMOTION_THRESHOLD']:
            printl(f'{self.name}: Rookie not promoted, get back to training')
            return False
        else:
            printl(f'{self.name}: Rookie promoted!')
            self.model.save_checkpoint(
                folder = self.training_args['CHECKPOINT_DIR'],
                filename = self.training_args['CHECKPOINT_PREFIX'] + str(self.generation),
            )
            return True

    def run(self):
        printl(f'{self.name}: Training process started')
        # random.seed()
        # os_reseed()
        if self.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id) # '{}'.format()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        printl(f'{self.name}: CUDA_VISIBLE_DEVICES set to {self.gpu_id}')
        self.lazy_model_module = LazyLoader(
            self.lazy_model_module,
            globals(),
            self.lazy_model_module
            )
        ModelClass = self.lazy_model_module.ExportedModel
        self.model = ModelClass(*self.model_args)
        time = int(datetime.datetime.now().timestamp() * 1e6) % 2**32
        printl(f'{self.name}: Setting seed to {time}')
        np.random.seed(time)

        prev_checkpoint = self.training_args.get('PREVIOUS_CHECKPOINT', None)
        if prev_checkpoint is not None:
            self.model.load_checkpoint(*prev_checkpoint)

        self.champ = self.model
        self.rookie = self.model

        can_train = False
        self.generation = 0

        printl(f'{self.name}: Outputting inital weights')
        self.output_weights()

        while True:
            if not self.kill_ingress.empty():
                break
            for _ in range(self.training_args['TRAINING_ROUNDS_PER_EVAL']):
                if not self.kill_ingress.empty():
                    break
                self.receive_examples()
                if can_train:
                    self.train()
                elif self.examples.get_memory_usage()[0] / self.examples.get_memory_usage()[1] > self.training_args['START_TRAINING_THRESHOLD']:
                    can_train = True
                    printl(f'{self.name}: Memory usage acceptable, can start training')
                    self.train()
            if not self.kill_ingress.empty():
                break
            if can_train:
                improved = self.eval()
                if improved:
                    printl(f'{self.name}: We now have generation {self.generation} of our model')
                    self.output_weights()


        printl(f'{self.name}: Training process killed')
        self.model.save_checkpoint(
            folder = self.training_args['CHECKPOINT_DIR'],
            filename = self.training_args['CHECKPOINT_PREFIX'] + 'final',
            # filename = self.training_args['CHECKPOINT_PREFIX'] + 'final.pkl',
        )
        printl(f'{self.name}: Final checkpoint saved - process ending')