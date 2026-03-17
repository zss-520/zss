#!/bin/bash
#SBATCH -J amp_eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o amp_eval.%j.out
#SBATCH -e amp_eval.%j.err

set -e  # 遇到报错立即退出

cd /share/home/zhangss/vlab_workspace
source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
conda activate eval_env
python eval_script.py
echo "finish"