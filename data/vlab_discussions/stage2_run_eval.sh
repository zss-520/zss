#!/bin/bash
#SBATCH -J amp_eval
#SBATCH --array=0-3%4   # 极度重要: 智能替换为 (模型总数量 - 1)
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o amp_eval_%A_%a.out
#SBATCH -e amp_eval_%A_%a.err

cd /share/home/zhangss/vlab_workspace
source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
conda activate eval_env
python evaluate_pipeline.py
echo "finish"
