#!/bin/bash
#SBATCH --exclusive
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high
#SBATCH -o logs/5_shot.sh.log-%j

python scripts/train/few_shot/run_train.py --data.dataset miniimagenet --data.split ravi --data.way 20 --model.x_dim '3,84,84' --data.cuda --log.exp_dir 5_shot_resnet_16_1_results --model.model_name protonet_resnet
