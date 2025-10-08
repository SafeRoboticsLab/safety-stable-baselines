#!/bin/bash
#SBATCH --job-name=goal_rollout_array

#SBATCH --nodes=1                               # node count
#SBATCH --ntasks=1                              # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                        # memory per cpu-core (4G is default)
#SBATCH --time=9:59:59                         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=duyn@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4                            # job array for 5 seeds

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2025.6
conda activate safety_sb3

wandb offline

python examples/car_goal_naive_train_withRolloutFilter.py --horizon 50 --velocity-threshold 0.0 --exp-suffix goal-batch --total-timesteps 300000 --seed $SLURM_ARRAY_TASK_ID