MCTS_ARGS = {
    'parallel_threads': 1,
    'cpuct': 1,
    'mcts_iterations': 60
}

SELFPLAY_ARGS = {
    'DETERMINISTIC_PLAY': 8,
    'GAMES_PER_SUBMISSION': 3,
}

TRAINING_ARGS = {
    'MAX_MEMORY_SIZE': int(1e6),
    'MAX_EXAMPLES_PER_RECEIVE': 10,
    'START_TRAINING_THRESHOLD': 4096 * 3 / 1e6,
    'TRAINING_SAMPLE_SIZE': 4096,
    'BATCH_SIZE': 128,
    'EPOCHS': 2,
    'EVAL_GAMES': 40,
    'PROMOTION_THRESHOLD': 0.55,
    'PREVIOUS_CHECKPOINT': None,
    'TRAINING_ROUNDS_PER_EVAL': 10,
    'CHECKPOINT_DIR': '../checkpoints',
    'CHECKPOINT_PREFIX': 'model_ex1_',
}

MAIN_ARGS = {
    # 'NUM_OF_GPUS': 4,
    # 'USE_GPUS': True,
    'NUM_OF_GPUS': 0,
    'USE_GPUS': False,
    'NUM_OF_SELFPLAY_PROCESSES': 3,
    'RUNNING_TIME': 252000
}




#################################
##  Beck data (game_state.py)  ##
#################################
m = 4
n = 9
k = 4


#####################################################
##  MCTS constants (parallel_mcts_nothreading.py)  ##
#####################################################
# MCTS_ARGS = {
#     'parallel_threads': 1,
#     'cpuct': 1,
#     'mcts_iterations': 40
# }

ALPHA = 0.3
C_PUCT = 3.0
EPSILON = 0.25
TAU = 1

NUMBER_OF_PASSES = 400
NUMBER_OF_THREADS = 8

POINT_OF_DETERMINISM = 10

IS_TWO_PLAYER_GAME = True

PASSES = 800

##################################
## Memory constants (memory.py) ##
##################################
STARTING_MEMORY_SIZE = 40000


####################################
## Network constants (network.py) ##
####################################
NNET_ARGS = {
    'REG_CONST': 0.0001,
    'LEARNING_RATE': 0.2,
    'MOMENTUM': 0.9,
    'INPUT_DIM': (3,4,9),
    'OUTPUT_DIM': (4,9),
    'NUM_OF_RESIDUAL_LAYERS': 4,

    'CONV_FILTERS': 128,
    'CONV_KERNEL_SIZE': (4,4),
    'RES_FILTERS': 128,
    'RES_KERNEL_SIZE': (4,4),
    'POLICY_HEAD_FILTERS': 32,
    'POLICY_HEAD_KERNEL_SIZE': (1,1),
    'VALUE_HEAD_FILTERS': 32,
    'VALUE_HEAD_KERNEL_SIZE': (1,1),
    'VALUE_HEAD_DENSE_NEURONS': 20,
}


MAX_GENERATIONS = 40000
LEARNING_RATES_TO_TRY = [0.02, 0.002, 0.0002]

TRAINING_LOOPS = 10
BATCH_SIZE = 2048
MINIBATCH_SIZE = 32
EPOCHS = 1


INITIAL_NNET_WEIGHTS_FILENAME = 'classes/initial_nnet_weights.pkl'