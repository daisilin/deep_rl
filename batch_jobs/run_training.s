#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-0:00:00
#SBATCH --mem=64GB
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=4IAR_AZ_run-1
#SBATCH --mail-type=END
#SBATCH --mail-user=jake.topping@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
conda init bash
conda activate fourinarow
module load cudnn/10.1v7.6.5.32

RUNDIR=$SCRATCH/4IAR-RL/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cp -avr $HOME/code/4IAR-RL/classes $RUNDIR
mkdir -p $RUNDIR/checkpoints

cd $RUNDIR
python -W ignore -u $RUNDIR/classes/main.py