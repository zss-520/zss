#!/bin/bash
#SBATCH --job-name=amp_stage1
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# 加载必要的模块 (如果需要)
# module load python/3.8

# 确保日志目录存在
mkdir -p logs
mkdir -p data

# 打印开始时间
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"

# 运行 Python 评估脚本
# 确保当前环境下 python 可用，或者使用绝对路径 /usr/bin/python3
python3 stage1_exploration.py

# 打印结束时间
echo "Job finished at: $(date)"
