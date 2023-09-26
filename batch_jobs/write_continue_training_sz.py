import os,sys,subprocess

root_dir = '/scratch/zz737/fiar/tournaments/tournament_4/'

script_name_list = []

def main():
    checkpoint_l, iter_l=get_args_list(root_dir)

    script_dir_name = 'continue_training_sz'
    if not os.path.exists(script_dir_name):
        os.mkdir(script_dir_name)

    for ii, (ch, it) in enumerate(zip(checkpoint_l, iter_l)):
        script_name = f'{script_dir_name}_{ii}.s'
        script_name = os.path.join(script_dir_name, script_name)

        f = open(script_name,'w')
        f.write(f'''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-0:00:00
#SBATCH --mem=64GB
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:p100:4
#SBATCH --job-name=4IAR-con
#SBATCH --mail-type=END
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output=slurm_%j.out
module purge

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

RUNDIR=$SCRATCH/fiar/run-${{SLURM_JOB_ID/.*}}
mkdir -p $RUNDIR/checkpoints

cp -avr $HOME/projects/fiar/4IAR-RL/classes $RUNDIR
cd $RUNDIR

singularity exec $nv \
--overlay /scratch/zz737/conda_related/fourinarow-20201223.ext3:ro \
/scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
/bin/bash -c "source /ext3/env.sh; python -W ignore -u $RUNDIR/classes/main_continue.py -ch {ch} -li {it} -lm 1"

exit
            ''')



        script_name_list.append(script_name)
        f.close()
    for script in script_name_list:
        subprocess.Popen(['bash','-c','chmod 744 ' + script])
        subprocess.Popen(['bash','-c','sbatch '+script])


def get_args_list(root_dir):
    subdirs = os.listdir(root_dir)
    checkpoint_l = []
    iter_l = []

    for dr in subdirs:
        if dr.startswith('check'):
            model_dir = os.path.join(root_dir,dr)
            model_fn_l =os.listdir(model_dir)
            it = get_iter(model_fn_l)
            checkpoint_l.append(model_dir)
            iter_l.append(it)

    return checkpoint_l, iter_l

def get_iter(model_fn_l):
    iter_max = 0
    for fn in model_fn_l:
        if fn.endswith('examples'):
            half = fn.split('_')[1]
            it = int(half.split('.')[0])
            if it > iter_max:
                iter_max = it
    return iter_max

if __name__ == '__main__':
    main()


