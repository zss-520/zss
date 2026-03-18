#!/bin/bash
#SBATCH --job-name=amp_stage1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# 加载必要的模块或环境 (根据集群实际情况调整)
module load anaconda3/2023.09  # 示例模块，请根据实际集群调整

# 创建日志目录
mkdir -p logs
mkdir -p data

# 打印开始时间
echo "Job started at: $(date)"

# 执行 Python 脚本
# 确保脚本路径正确，这里假设脚本在当前目录
python stage1_exploration.py

# 打印结束时间
echo "Job finished at: $(date)"
