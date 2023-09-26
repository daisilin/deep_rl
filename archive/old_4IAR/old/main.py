# Global imports
import tensorflow as tf
#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Local imports
from ai_player import AIPlayer
from beck.beck_display import BeckDisplay
from beck.beck_game import BeckGame
from beck.beck_model import RandomBeckModel, NnetBeckModel
from beck.config import MCTS_ARGS, NNET_ARGS
from mcts import MCTS
from stage import Stage

import time

game = BeckGame(m=4, n=9, k=4)

start_time = time.time()
model1, model2 = NnetBeckModel(game, NNET_ARGS), NnetBeckModel(game, NNET_ARGS)
# model1, model2 = RandomBeckModel(game), RandomBeckModel(game)
trials = 100
for _ in range(trials):
    mcts1, mcts2 = MCTS(game, model1, MCTS_ARGS), MCTS(game, model1, MCTS_ARGS)
    players = [AIPlayer(mcts1, True), AIPlayer(mcts2, True)]

    #display = BeckDisplay(game, ['Rebecca', 'Rebecca'])
    stage = Stage(players, game, None)

    stage.execute()

print('Time: ', (time.time() - start_time)/trials)

import time
time.sleep(5)