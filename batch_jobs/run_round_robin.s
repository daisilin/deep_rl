#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-0:00:00
#SBATCH --mem=4GB
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=4IAR_tournament
#SBATCH --mail-type=END
#SBATCH --mail-user=xl1005@nyu.edu
#SBATCH --output=out_fix/tour_%j.out
##SBATCH --array=0-834
#SBATCH --array=0-509
##SBATCH --array=619,620,621,622,703,704,705,706,711,712,713,714,727,728,729,730,795,796,797,798,803,804,805,806,819,820,821,822,827,828,829,830,623,624,625,626,699,700,701,702,707,708,709,710,723,724,725,726,799,800,801,802,807,808,809,810,823,824,825,826,831,832,833,834
##SBATCH --account=cds

module purge

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

RUNDIR=$SCRATCH/deep-master/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cp -avr $SCRATCH/deep-master/classes $RUNDIR
cd $RUNDIR

singularity exec $nv \
            --overlay /scratch/xl1005/conda_related/fourinarow-20201223.ext3:ro \
            /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
            /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/repeated_tournament.py $SLURM_ARRAY_TASK_ID"
            # /bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/tournament_new.py $SLURM_ARRAY_TASK_ID"

exit






