#!/bin/bash
#SBATCH --exclusive
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high
#SBATCH -o logs/1_shot_trainval.sh.log-%j

python scripts/train/few_shot/run_trainval.py --model.model_path 1_shot_results/best_model.pt
