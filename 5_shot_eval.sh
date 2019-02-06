#!/bin/bash
#SBATCH --exclusive
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high
#SBATCH -o logs/5_shot_eval.sh.log-%j

python scripts/predict/few_shot/run_eval.py --model.model_path 5_shot_results/best_model.pt
