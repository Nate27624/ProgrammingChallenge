#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00 
#SBATCH --job-name=test_run
#SBATCH --account=PAS3361

#SBATCH --mem=64gb

#SBATCH --cpus-per-task=11
#SBATCH --gpus-per-node=1


module load cuda/11.8.0
module load miniconda3/24.1.2-py310
source activate local
cd 'YOUR CODE DIRECTORY'
export WANDB_API_KEY=YOUR WANDB API
python train_wandb_example.py