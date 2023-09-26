#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-0:00:00
#SBATCH --mem=6GB
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=depth
#SBATCH --mail-type=END
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=depth_out/slurm_%a.out
#SBATCH --array=0-509
##SBATCH --account=cds

module purge

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

TO_COMPUTE=depth
RUNDIR=$SCRATCH/fiar/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cp -avr $HOME/projects/fiar/fiarrl/classes $RUNDIR
cp -avr $HOME/projects/fiar/fiarrl/analysis $RUNDIR
cd $RUNDIR

singularity exec $nv \
            --overlay /scratch/zz737/conda_related/fourinarow-20201223.ext3:ro \
            /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
            /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/analysis/derived_metrics.py $SLURM_ARRAY_TASK_ID $TO_COMPUTE"
            # /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/compute_depth_parallel.py $SLURM_ARRAY_TASK_ID"
            # 

            

exit






