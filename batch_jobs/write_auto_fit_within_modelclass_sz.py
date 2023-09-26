import sys, os, subprocess
# for each model copy, get the number of jobs (5 * number of iters), input the correct direc

script_name_list = []
results_name = 'auto_fit'

root_direc= '/scratch/zz737/fiar/tournaments/tournament_4/moves/splits'

def main():
    script_dir_name = 'script_autofit_within_modelclass'
    if not os.path.exists(script_dir_name):
        os.makedirs(script_dir_name, exist_ok=True) # makedirs recursively

    # model_copy_dirs = os.listdir(root_direc) # get folders like [checkpoints_mcts100_cpuct2_id-xxxx, ...]
    model_copy_dirs = ['checkpoints_mcts100_cpuct2_id-3752918']#['checkpoints_mcts100_cpuct2_id-3754964'] # if only doing one model copy! To be modified if doing more!
    for d in model_copy_dirs:
        if d.startswith('checkpoints'):
            model_copy_dir = os.path.join(root_direc, d) # e.g. /scratch/zz737/fiar/tournaments/tournament_4/moves/splits/checkpoints_mcts100_cpuct2_id-3751934
            model_copy_dir_subdir = os.listdir(model_copy_dir) # [1,2,...]
            model_copy_dir_subdir = [x for x in model_copy_dir_subdir if x.isnumeric()] # check no anomaly subfolders, should all be numerically named
            niters = len(model_copy_dir_subdir)

            model_copy_dir_sub1 = os.path.join(model_copy_dir,'1') # e.g. /scratch/zz737/fiar/tournaments/tournament_4/moves/splits/checkpoints_mcts100_cpuct2_id-3751934/1
            if 'params1.csv' in os.listdir(model_copy_dir_sub1): # if the copy has already been run, continue to skip the loop
                continue

            script_name = f'{results_name}_{d}.sh'
            script_name = os.path.join(script_dir_name, script_name)

            f = open(script_name,'w')
            f.write(f'''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
##SBATCH --array=0-{5*niters-1}
#SBATCH --array=60-64
#SBATCH --time=12:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=cogmodel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=./{script_dir_name}/autofit_%j.out

player=$((${{SLURM_ARRAY_TASK_ID}}/5+1))
group=$((${{SLURM_ARRAY_TASK_ID}}%5+1))
#group=0
direc={model_copy_dir}
codedirec=/home/zz737/projects/fiar/cog_model/fourinarow/Model\\ code/matlab\\ wrapper

module purge
module load matlab/2020b


echo $player $group

echo "addpath(genpath('${{codedirec}}')); cross_val($player,$group,'${{direc}}'); exit;" | matlab -nodisplay

echo "Done"''')
            script_name_list.append(script_name)
            f.close()

    for script in script_name_list:
        subprocess.Popen(['bash','-c','chmod 744 ' + script])
        subprocess.Popen(['bash','-c','sbatch '+script])


if __name__ == '__main__':
    main()







