import os,sys,subprocess
import pandas as pd


participants_dir = '/scratch/zz737/fiar/tournaments/tournament_4' 
results_dir = '/scratch/zz737/fiar/tournaments/results/tournament_4' 
moves_dir = '/scratch/zz737/fiar/tournaments/tournament_4/moves/raw/'


script_name_list = []
results_name = 'round_robin'
ngames = 10

###### repeated in tournament_parallel_within_modelclass.py; for convenience, otw need to deal with singularity ####
def join_dirs(model_copy_name, participants_dir, results_dir, moves_dir):
    participants_dir = os.path.join(participants_dir, model_copy_name)
    results_dir = os.path.join(results_dir, model_copy_name)
    moves_dir = os.path.join(moves_dir, model_copy_name)
    return participants_dir, results_dir, moves_dir
def get_participant_iters(model_copy_name, participants_dir, results_dir, moves_dir):
    '''
    return participant_iters, len(participant_iters)
    list of participants/models in a tournament 
    '''
    participants_dir, results_dir, moves_dir = join_dirs(model_copy_name, participants_dir, results_dir, moves_dir)

    model_class = model_copy_name.split('-')[0] # model copy name: eg checkpoints_mcts100_cpuct2_id-3751934
    model_class = model_class.split('_')[1:3] #skip checkpoints, id
    model_class = '_'.join(model_class) # eg mcts100_cpuct2

    all_iters = os.listdir(participants_dir)
    all_iters_int = []
    for iter in all_iters: # eg. checkpoint_0.pth.tar.examples
        if iter.startswith('checkpoint_') and not iter.endswith('examples'):
            iter_int = int(iter.split('_')[1].split('.')[0])
            all_iters_int.append(iter_int)

    all_iters_int = set(all_iters_int)

    participant_iters = [f'{model_class};{x}' for x in all_iters_int] # eg 'mcts80_cpuct3;57'
    print(participant_iters)
    return participant_iters, len(participant_iters)
###### ================== ####

def main():

    model_copy_list  = os.listdir(participants_dir) # get ['checkpoints_mcts100_cpuct2_id-3751934', ...]
    model_copy_list = [x for x in model_copy_list if x.startswith('checkpoints')][1:2] # only select checkpoints # [[testing!!!]] only select :1 to see if it runs

    # create a folder for the scripts
    script_dir_name = 'script_tournament_parallel_within_modelclass'
    if not os.path.exists(script_dir_name):
        os.makedirs(script_dir_name, exist_ok=True) # makedirs recursively

    # create the new tournament_x folder for results and moves
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(moves_dir):
        os.makedirs(moves_dir, exist_ok=True)


    for ii,model_copy in enumerate(model_copy_list):
        #figure out how many iters within a copy, i.e. rows in the csv i.e. jobs in the array
        participant_iters, niters= get_participant_iters(model_copy, participants_dir, results_dir, moves_dir)


        # create an empty results_df for each tournament (within each model copy)
        results_df = pd.DataFrame(index=participant_iters, columns=participant_iters)
        results_df.to_csv(os.path.join(results_dir, results_name + '.csv')) # actually not used?

        copy_id  = model_copy.split('-')[1] 
        script_name = f'{copy_id}.s'
        script_name = os.path.join(script_dir_name, script_name)

        f = open(script_name,'w')
        f.write(f'''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:00:00
#SBATCH --mem=20GB
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=round-{ii}
#SBATCH --mail-type=END
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=./{script_dir_name}/slurm_%j.out
#SBATCH --array=0-{niters-1}
module purge

cd $HOME/projects/fiar/4IAR-RL/classes 

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi


singularity exec $nv \
--overlay /scratch/zz737/conda_related/fourinarow-20201223.ext3:ro \
/scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
/bin/bash -c "source /ext3/env.sh; python -W ignore -u tournament_parallel_within_modelclass.py $SLURM_ARRAY_TASK_ID {model_copy} {participants_dir} {results_dir} {moves_dir} round_robin {ngames}"

exit
            ''')


        script_name_list.append(script_name)
        f.close()
    for script in script_name_list:
        subprocess.Popen(['bash','-c','chmod 744 ' + script])
        subprocess.Popen(['bash','-c','sbatch '+script])


if __name__ == '__main__':
    main()


