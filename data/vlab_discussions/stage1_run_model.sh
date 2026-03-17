#!/bin/bash
#SBATCH --job-name=stage1_exploration
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# 创建日志目录
mkdir -p logs

# 加载必要的模块 (根据集群实际情况调整，此处假设基础环境已就绪)
# module load python/3.8

echo "Start Time: $(date)"
echo "Running Stage 1 Exploration Script..."

# 执行 Python 脚本
# 确保脚本路径正确，此处假设脚本在当前目录
python stage1_exploration.py

echo "End Time: $(date)"
echo "Job Finished."
