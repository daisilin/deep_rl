#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=0-944
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=fourinarow
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=4inarow_%j.out
dataset=tournament_3
i=${SLURM_ARRAY_TASK_ID}
direc=$SCRATCH/fiar/tournaments/${dataset}
codedirec=$SCRATCH/fourinarow/Model\ code
paramfile=${direc}/movesparams_transfer_pilot_long.txt
player=$(($i/5))
group=$(($i%5+1))
outputfile=${direc}/depth/depth_${dataset}_${player}_${group}.txt
echo $paramfile $outputfile $player $group
time ./compute_planning_depth ../data_hvh.txt $paramfile $player $group $outputfile 10;
echo "Done"