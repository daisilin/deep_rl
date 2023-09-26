#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=60-64
#SBATCH --time=12:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=cogmodel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=./script_autofit_within_modelclass/autofit_%j.out

player=$((${SLURM_ARRAY_TASK_ID}/5+1))
group=$((${SLURM_ARRAY_TASK_ID}%5+1))
#group=0
direc=/scratch/zz737/fiar/tournaments/tournament_4/moves/splits/checkpoints_mcts100_cpuct2_id-3752918
codedirec=/home/zz737/projects/fiar/cog_model/fourinarow/Model\ code/matlab\ wrapper

module purge
module load matlab/2020b


echo $player $group

echo "addpath(genpath('${codedirec}')); cross_val($player,$group,'${direc}'); exit;" | matlab -nodisplay

echo "Done"