#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=4IAR-sl
#SBATCH --mail-type=END
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=train_out/sl_%j.out
#SBATCH --account=cds
#SBATCH --array=0-0

module purge

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

RUNDIR=$SCRATCH/fiar/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR/checkpoints

# cp -avr $HOME/projects/fiar/4IAR-RL/classes $RUNDIR
cp -avr $HOME/projects/fiar/fiarrl/classes $RUNDIR
cd $RUNDIR

singularity exec $nv \
            --overlay /scratch/zz737/conda_related/fourinarow-20201223.ext3:ro \
            /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
            /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/supervised_learning.py $SLURM_ARRAY_TASK_ID"
            # /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/supervised_learning_pvcorr.py $SLURM_ARRAY_TASK_ID"
            
            
            
            
            

exit

