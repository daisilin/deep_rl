import multiprocessing as mp
from multiprocessing import Queue
import time


# lazy_model_module = LazyLoader(
#     ,
#     globals(),
#     "beck.beck_model"
#     )

LAZY_MODEL_NAME = "beck.beck_model"

from beck.beck_game import BeckGame as Game
from beck.config import MAIN_ARGS, MCTS_ARGS, NNET_ARGS, SELFPLAY_ARGS, TRAINING_ARGS, m, n, k

from selfplay_process import SelfplayProcess
from training_process import TrainingProcess

from utils import printl

def set_up_processes(name, main_args, mcts_args, nnet_args, selfplay_args, training_args, m, n, k):
    printl(f'{name}: Setting up processes')
    game = Game(m, n, k)
    kill_ingress = Queue(1)
    example_ingress = Queue()
    weight_egresses = [Queue() for _ in range(main_args['NUM_OF_SELFPLAY_PROCESSES'])]
    
    training_process = TrainingProcess(
        name = 'TP0',
        gpu_id = '0' if main_args['USE_GPUS'] and main_args['NUM_OF_GPUS'] else None,
        game = game,
        lazy_model_module = LAZY_MODEL_NAME,
        model_args = (game, nnet_args),
        example_ingress = example_ingress,
        weight_egresses = weight_egresses,
        kill_ingress = kill_ingress,
        training_args = training_args,
        mcts_args = mcts_args,
        selfplay_args = selfplay_args,
    )

    printl(f'{name}: Training process set up')

    if main_args['USE_GPUS'] and main_args['NUM_OF_GPUS']:
        selfplay_gpu_ids = [str(i) for i in range(1, main_args['NUM_OF_GPUS'])]
        while len(selfplay_gpu_ids) < MAIN_ARGS['NUM_OF_SELFPLAY_PROCESSES']:
            selfplay_gpu_ids.append(None)
    else:
        selfplay_gpu_ids = [None for _ in range(main_args['NUM_OF_SELFPLAY_PROCESSES'])]
    
    selfplay_processes = []
    for i, gpu_id, weight_egress in zip(range(main_args['NUM_OF_SELFPLAY_PROCESSES']), selfplay_gpu_ids, weight_egresses):
        time.sleep(1)
        selfplay_processes.append(SelfplayProcess(
            name = f'SP{i}',
            gpu_id = gpu_id,
            game = game,
            lazy_model_module = LAZY_MODEL_NAME,
            model_args = (game, nnet_args),
            mcts_args = mcts_args,
            selfplay_args = selfplay_args,
            weight_ingress = weight_egress,
            example_egress = example_ingress,
            kill_ingress = kill_ingress,
        ))

    printl(f'{name}: Selfplay processes set up')

    return training_process, selfplay_processes, kill_ingress


if __name__ == '__main__':
    printl('Main: Program beginning')
    mp.set_start_method('spawn', force=True)
    training_process, selfplay_processes, kill_ingress = set_up_processes(
        'Main',
        MAIN_ARGS,
        MCTS_ARGS,
        NNET_ARGS,
        SELFPLAY_ARGS,
        TRAINING_ARGS,
        m, n, k)
    printl('Main: Processes created')

    printl('Main: Starting training process')
    training_process.start()
    printl('Main: Starting selfplay processes')
    for proc in selfplay_processes:
        proc.start()
    
    printl('Main: Main process sleeping until kill time')
    time.sleep(MAIN_ARGS['RUNNING_TIME'])

    printl('Main: Main process awake - killing training and selfplay processes')
    kill_ingress.put('kill')

    training_process.join()
    printl(f'Main: Training process {training_process.name} joined')
    for proc in selfplay_processes:
        proc.join()
        printl(f'Main: Selfplay process {proc.name} joined')

    printl('Program exiting')
