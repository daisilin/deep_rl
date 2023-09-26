#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
##SBATCH --cpus-per-task=1
#SBATCH --time=5-0:00:00
##SBATCH --time=1:00:00
#SBATCH --mem=64GB
##SBATCH --mem=16GB
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=4IAR-train
#SBATCH --mail-type=END
#SBATCH --mail-user=xl1005@nyu.edu
#SBATCH --output=train_out/rescol_%j.out


module purge

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

RUNDIR=$SCRATCH/deep-master/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR/checkpoints

# cp -avr $HOME/projects/fiar/4IAR-RL/classes $RUNDIR
cp -avr $SCRATCH/deep-master/classes $RUNDIR
cd $RUNDIR

singularity exec $nv \
            --overlay /scratch/xl1005/conda_related/fourinarow-20201223.ext3:ro \
            /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
            /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/main.py 0"
            # /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/main_cog_val.py"
            

exit

