#!/bin/bash
#SBATCH --job-name=circle_value_array

#SBATCH --nodes=1                               # node count
#SBATCH --ntasks=1                              # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                        # memory per cpu-core (4G is default)
#SBATCH --time=2:59:59                         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=duyn@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH -p gpu-ee
#SBATCH --array=0-14                            # job array for 5 seeds x 3 epsilons

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

echo $CUDA_VISIBLE_DEVICES
source $HOME/.bashrc
JOBID=$SLURM_JOB_ID
cd $WORKDIR
mkdir -p safety_sb3-experiments
cd safety_sb3-experiments
mkdir -p safety_sb3_$JOBID/
cd safety_sb3_$JOBID/
cp -r $HOME/SAFE/safety-stable-baselines/* .

module purge
module load anaconda3/2025.6
conda activate safety_sb3

wandb offline

# Map SLURM_ARRAY_TASK_ID to (seed, epsilon)
seeds=(0 1 2 3 4)
epsilons=(0.0 0.1 0.2)
seed_idx=$((SLURM_ARRAY_TASK_ID % 5))
eps_idx=$((SLURM_ARRAY_TASK_ID / 5))
seed=${seeds[$seed_idx]}
epsilon=${epsilons[$eps_idx]}

echo "Running with seed $seed and epsilon $epsilon"

python3 examples/car_circle_naive_train_withFilter.py --exp-suffix circle-batch --total-timesteps 200000 --seed $seed --epsilon $epsilon