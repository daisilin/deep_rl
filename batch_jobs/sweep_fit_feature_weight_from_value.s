#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=1-100
#SBATCH --cpus-per-task=1
#SBATCH --time=5-0:00:00
#SBATCH --mem=16GB
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=feat
#SBATCH --mail-type=END
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=feat_out/slurm_%j.out

module purge

iter_csv_template="mcts100_cpuct2\;{}.csv"
moves_dir_no_raw=$SCRATCH/fiar/tournaments/tournament_4/moves/
model_instance_str=checkpoints_mcts100_cpuct2_id-3752918

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi



singularity exec $nv \
            --overlay /scratch/zz737/conda_related/fourinarow-20201223.ext3:ro \
            /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
            /bin/bash -c "source /ext3/env.sh; python -W ignore -u fit_feature_weight_from_value.py ${SLURM_ARRAY_TASK_ID} -t ${iter_csv_template} -d ${moves_dir_no_raw} -m ${model_instance_str}"

exit