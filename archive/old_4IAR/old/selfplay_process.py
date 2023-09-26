import datetime
import multiprocessing as mp
import numpy as np
import os
import random
import time

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL) 

from ai_player import AIPlayer
from lazy_loader import LazyLoader
from mcts import MCTS
from utils import printl, os_reseed

class SelfplayProcess(mp.Process):
    def __init__(self, name, gpu_id, game, lazy_model_module, model_args, mcts_args, selfplay_args, weight_ingress, example_egress, kill_ingress):
        mp.Process.__init__(self)
        self.name = name
        self.gpu_id = gpu_id
        self.game = game
        self.lazy_model_module = lazy_model_module
        self.model_args = model_args
        self.mcts_args = mcts_args
        self.selfplay_args = selfplay_args
        self.weight_ingress = weight_ingress
        self.example_egress = example_egress
        self.kill_ingress = kill_ingress
        self.episode_count = 0
        self.submission_count = 0

    def update_model_weights(self, weights):
        self.model.set_weights(weights)
        self.searches = [MCTS(self.game, self.model, self.mcts_args) for _ in range(len(self.game.players))]
        printl(f'{self.name}: Updated model weights')

    def execute_episode_with_example_storing(self):
        state = self.game.get_initial_state()
        current_player = 1
        turn_count = 0
        training_examples = []
        start_time = time.time()
        actions = []

        while not self.game.get_is_terminal_state(state):
            canonical_state = self.game.get_canonical_form(state, current_player)
            if turn_count >= self.selfplay_args['DETERMINISTIC_PLAY']:
                probs = self.searches[current_player - 1].get_probs(canonical_state, temperature=1)
                action = np.argmax(probs)
            else:
                probs = self.searches[current_player - 1].get_probs(canonical_state, temperature=1)
                # printl(f'{self.name}: {np.random.random()}')
                # printl(f'{self.name}: {self.game.get_allowed_actions(state)}')
                # action = np.random.choice(self.game.get_allowed_actions(state), p = probs)
                action = np.random.choice(range(len(probs)), p = probs)
                assert self.game.get_allowed_actions(state)[action], f'Must choose an allowed action!\nChose {action} in state\n{canonical_state}'
            # printl(f'{self.name}: action on turn {turn_count} is {action}')
            symmetries = self.game.get_symmetries(canonical_state, probs)
            for s, p in symmetries:
                training_examples.append([s, p, current_player])
            # action = np.argmax(probs)
            previous_player = current_player
            state, current_player = self.game.get_next_state(state, current_player, action)
            turn_count += 1
            actions.append(str(action))
        
        self.episode_count += 1
        result = self.game.get_result(state, current_player)
        if result == 0:
            printed_result = 'draw'
        else:
            printed_result = f'P{3 - current_player} wins'
        printl(f'{self.name}: Episode {self.episode_count} completed in {round(time.time() - start_time, 2)}s with \
{turn_count} actions and generating {len(training_examples)} examples.\
\nActions taken were {", ".join(actions)}.\nResult was {printed_result}\n{state}')
        training_examples = [(x[0], x[1], result*((-1)**(x[2] != previous_player))) for x in training_examples]
        for search in self.searches:
            search.reset()

        return training_examples

    def generate_training_examples(self):
        training_examples = []
        for _ in range(self.selfplay_args['GAMES_PER_SUBMISSION']):
            training_examples += self.execute_episode_with_example_storing()
        printl(f'{self.name}: Finished episode generation {self.submission_count + 1} with {len(training_examples)} examples - looking to submit examples')
        return training_examples

    def run(self):
        printl(f'{self.name}: Selfplay process started')
        # np.random.seed(int(datetime.datetime.now().timestamp() * 1e6) % 2**32)
        # np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
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
        time = (int(datetime.datetime.now().timestamp() * 1e6) ** 10) % 2**32
        printl(f'{self.name}: Setting seed to {time}')
        np.random.seed(time)
        self.model = ModelClass(*self.model_args)
        self.searches = [MCTS(self.game, self.model, self.mcts_args) for _ in range(len(self.game.players))]
        printl(f'{self.name}: Set up model and searches')

        printl(f'Awaiting initial weights')
        weights = self.weight_ingress.get()
        printl(f'{self.name}: Found initial weights')
        self.update_model_weights(weights)
        printl(f'{self.name}: Initial weights loaded')

        while self.kill_ingress.empty():
            if not self.weight_ingress.empty():
                weights = self.weight_ingress.get()
                printl(f'{self.name}: Found weights update')
                self.update_model_weights(weights)
                printl(f'{self.name}: Weights updated')
            else:
                printl(f'{self.name}: No weights update, continuing')
            printl(f'{self.name}: Starting example submission {self.submission_count + 1}')
            self.example_egress.put(self.generate_training_examples())
            self.submission_count += 1
            printl(f'{self.name}: Finished example submission {self.submission_count}')
        
        printl(f'{self.name}: Selfplay process killed - process ending')


