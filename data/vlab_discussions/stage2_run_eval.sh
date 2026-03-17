#!/bin/bash
   #SBATCH -J amp_vlab_eval
   #SBATCH -N 1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=4
   #SBATCH --gres=gpu:1
   #SBATCH -p gpu
   #SBATCH -o eval_job.%j.out
   #SBATCH -e eval_job.%j.err

   cd /share/home/zhangss/vlab_workspace
   source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
   conda activate eval_env
   python eval_script.py
   echo "finish"
   ```
# PI 审阅与任务明确要求

## 1. 任务阶段确认
本次为**第二阶段（评测阶段）**。核心目标是**基于已有的预测结果进行标准化评估**，严禁重新运行模型推理命令。
- **输入源**：直接读取第一阶段生成的标准化文件（`ground_truth.csv`, `macrel.out.prediction.gz`, `ampscanner_out.csv`, `AmpNet_out/*.csv`）。
- **禁止项**：绝对禁止包含 `subprocess.run` 调用模型推理的代码，禁止从 FASTA 解析标签。

## 2. 数据清洗与合并红线
- **ID 清洗**：所有 ID 列必须执行 `str(x).split()[0].strip()`，确保无 `>` 符号且无多余空格。
- **合并策略**：必须以 `ground_truth` 为左表 (`how='left'`)，确保评估样本集一致。
- **缺失值兜底**：合并后的概率列缺失值必须强制 `fillna(0.0)`。
- **类型安全**：概率列必须确保为 `float` 类型，防止字符串污染。

## 3. 输出与格式规范
- **指标计算**：ACC, Recall, MCC, AUROC, AUPRC。
- **JSON 结构**：双层嵌套字典 `{ModelName: {Metric: float_value}}`，所有数值必须转为 Python 原生 `float`。
- **绘图要求**：`matplotlib.use('Agg')`，使用 `numpy.interp` 进行 ROC 插值，保存为 `evaluation_curves.png`。
- **结果表**：保存 `final_results_with_predictions.csv`，包含真实标签及各模型预测概率/标签。
- **代码结构**：单一完整 Python 脚本，包含 `def main():` 及入口判断；Slurm 脚本独立成块。

---

# 代码工程师最终完整代码
