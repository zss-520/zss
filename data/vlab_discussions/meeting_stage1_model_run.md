## User

现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。

会议任务如下：
围绕以下从记忆库提取的预测模型：PepNet, PepNet，编写自动化代码。
请严格遵守以下共享基础要求：

1. 【历史数据清理机制】：
   在执行模型预测前，必须执行以下清理逻辑：
   - `shutil.rmtree('data/PepNet_out', ignore_errors=True)` 或清理对应的结果文件
   - `shutil.rmtree('data/PepNet_out', ignore_errors=True)` 或清理对应的结果文件


2. 【强校验命令执行（极度重要）】：
   必须严格照抄以下代码块来执行预测（切勿自行添加 try-except 吞咽错误）：

   # PepNet 调用代码模板
   print("开始运行 PepNet...")
   PepNet_cmd = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate pepnet_env && python predict.py --input data/combined_test.fasta --output data/PepNet_out"'
   res_PepNet = subprocess.run(PepNet_cmd, shell=True, capture_output=True, text=True)
   if res_PepNet.returncode != 0:
       print(f"!!! PepNet 真实报错日志:\n{res_PepNet.stderr}")
       raise RuntimeError("PepNet 预测执行失败，已阻断程序！")

   # PepNet 调用代码模板
   print("开始运行 PepNet...")
   PepNet_cmd = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate pepnet_env && python predict.py --input data/combined_test.fasta --output data/PepNet_out"'
   res_PepNet = subprocess.run(PepNet_cmd, shell=True, capture_output=True, text=True)
   if res_PepNet.returncode != 0:
       print(f"!!! PepNet 真实报错日志:\n{res_PepNet.stderr}")
       raise RuntimeError("PepNet 预测执行失败，已阻断程序！")


3. 【代码完整性要求】：
   - Python 代码必须是单一、完整、可运行的脚本
   - 必须包含完整 import，包含 `def main():` 并以 `if __name__ == '__main__':` 为入口


【这次是第一次会议（探索勘探阶段），请严格执行以下要求】：
1. 本次会议的唯一目标是：运行给定列表中的所有模型，并“勘探”它们实际生成的文件结构。
2. 你的 Python 脚本在运行完模型（subprocess）后，必须编写代码遍历 `data/` 目录下所有带有 `_out` 后缀的目录或新生成的文件：
   - 使用 `glob` 或 `os.listdir` 找到模型实际生成的所有结果文件（特别是未知的 .gz, .csv, .txt）。
   - 尝试读取这些文件的头 5 到 10 行内容（如果发现是 .gz 文件，请 import gzip 读取纯文本）。
   - 将你找到的文件绝对/相对路径，以及文件的头部内容摘要，写出到一份勘探报告中。
3. 勘探报告必须保存为：`data/stage1_observation.txt`。
4. 绝对禁止（重要！）：本次会议的脚本绝对不许用 pandas 做任何 merge 操作！不许清洗数据！不许计算指标！

## PI

# 🛑 PI 审阅与任务明确要求 (Phase 1: Exploration)

各位工程师，大家好。我是本次项目的 PI。针对本次 **Stage 1 探索勘探阶段** 的任务，我已审阅了需求文档。为了确保后续 AMP 评估的准确性，我们必须首先搞清楚模型到底输出了什么格式的数据。

以下是本次代码编写的 **硬约束 (Hard Constraints)** 和 **执行策略**，请工程师在编写代码时严格遵守，不得有任何偏差：

### 1. 核心目标确认
*   **任务性质**：仅为“勘探” (Observation)。**绝对禁止** 在此阶段进行任何数据清洗、Pandas Merge 或指标计算。
*   **产出物**：必须生成一份纯文本勘探报告 `data/stage1_observation.txt`，记录文件路径及文件头内容。

### 2. 历史数据清理机制 (强制)
*   在调用任何模型预测前，**必须** 执行清理命令，防止旧数据污染本次勘探结果。
*   代码中必须显式包含：`shutil.rmtree('data/PepNet_out', ignore_errors=True)`。

### 3. 强校验命令执行 (极度重要)
*   **禁止吞咽错误**：不得使用 `try-except` 包裹 `subprocess` 来忽略错误。如果模型运行失败，必须抛出 `RuntimeError` 阻断程序。
*   **模板一致性**：PepNet 的调用命令必须与提供的模板 **完全一致**（包括 conda 路径、环境变量激活方式）。
*   **逻辑修正说明**：任务列表中提到 "PepNet, PepNet"，但两者输出路径均为 `data/PepNet_out` 且清理机制会删除该目录。若执行两次，第二次会删除第一次的结果，导致勘探失败。**基于 PI 决策，本脚本将仅执行一次 PepNet 预测**，以确保有结果可供勘探。

### 4. 勘探逻辑要求
*   **遍历范围**：`data/` 目录下所有后缀为 `_out` 的目录或文件。
*   **文件读取**：
    *   普通文本文件：直接读取前 5-10 行。
    *   压缩文件 (`.gz`)：必须使用 `gzip` 模块解压后读取文本内容。
*   **报告格式**：清晰记录 `文件路径` 和 `文件头摘要`。

### 5. 代码规范
*   单一脚本，包含 `import`，`def main():`，`if __name__ == '__main__':`。
*   确保 `data/` 目录存在（用于存放报告）。

---

# 💻 工程师最终完整代码 (Stage 1 Implementation)

```python
import os
import shutil
import subprocess
import glob
import gzip

def explore_output_directory(base_dir='data'):
    """
    勘探 data 目录下所有 *_out 相关的文件，读取头部内容。
    """
    observation_log = []
    observation_log.append("=" * 50)
    observation_log.append("Stage 1: Model Output Observation Report")
    observation_log.append("=" * 50)
    observation_log.append("")

    # 查找所有 *_out 结尾的目录或文件
    # 使用 glob 查找 data/ 下直接子目录或文件匹配 *_out
    pattern = os.path.join(base_dir, '*_out')
    found_items = glob.glob(pattern)
    
    # 同时也查找 data/ 下可能存在的子目录中的 *_out (递归搜索以防万一)
    # 但根据任务描述，主要是模型直接生成的 output 目录
    # 这里我们主要关注找到的 items 是目录还是文件
    
    if not found_items:
        observation_log.append("警告：未在 data/ 目录下找到任何 *_out 后缀的目录或文件。")
    else:
        for item_path in found_items:
            observation_log.append(f"勘探对象：{item_path}")
            observation_log.append("-" * 30)
            
            files_to_read = []
            if os.path.isdir(item_path):
                # 如果是目录，遍历内部文件
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        files_to_read.append(os.path.join(root, file))
            else:
                # 如果是文件
                files_to_read.append(item_path)
            
            for file_path in files_to_read:
                observation_log.append(f"  文件路径：{file_path}")
                try:
                    lines = []
                    # 处理 .gz 文件
                    if file_path.endswith('.gz'):
                        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if i >= 10: break
                                lines.append(line.strip())
                    else:
                        # 处理普通文本文件
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if i >= 10: break
                                lines.append(line.strip())
                    
                    if lines:
                        observation_log.append("  文件头内容摘要 (前 10 行):")
                        for line in lines:
                            observation_log.append(f"    | {line}")
                    else:
                        observation_log.append("  文件头内容摘要：[文件为空]")
                except Exception as e:
                    observation_log.append(f"  读取失败：{str(e)}")
            
            observation_log.append("")

    observation_log.append("=" * 50)
    observation_log.append("End of Observation")
    observation_log.append("=" * 50)
    
    return "\n".join(observation_log)

def main():
    # 0. 确保 data 目录存在，以便后续保存报告
    os.makedirs('data', exist_ok=True)

    # 1. 【历史数据清理机制】
    # 在执行模型预测前，必须执行清理逻辑
    print("正在执行历史数据清理...")
    shutil.rmtree('data/PepNet_out', ignore_errors=True)
    # 注：任务要求中列出了两次 PepNet 清理，逻辑上清理一次即可确保目录干净
    print("清理完成。")

    # 2. 【强校验命令执行】
    # 必须严格照抄以下代码块来执行预测（切勿自行添加 try-except 吞咽错误）
    # 注：虽然任务列表提到两次 PepNet，但因输出路径相同且伴随清理，重复执行会导致数据丢失。
    # 此处严格执行一次预测以保留结果供勘探。
    print("开始运行 PepNet...")
    PepNet_cmd = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate pepnet_env && python predict.py --input data/combined_test.fasta --output data/PepNet_out"'
    res_PepNet = subprocess.run(PepNet_cmd, shell=True, capture_output=True, text=True)
    if res_PepNet.returncode != 0:
        print(f"!!! PepNet 真实报错日志:\n{res_PepNet.stderr}")
        raise RuntimeError("PepNet 预测执行失败，已阻断程序！")
    
    print("PepNet 运行成功。")

    # 3. 【勘探逻辑】
    # 运行完模型后，遍历 data/ 目录下所有带有 _out 后缀的目录或新生成的文件
    print("开始勘探模型输出结构...")
    report_content = explore_output_directory('data')
    
    # 4. 【保存勘探报告】
    # 勘探报告必须保存为：data/stage1_observation.txt
    report_path = 'data/stage1_observation.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"勘探报告已保存至：{report_path}")
    print("Stage 1 任务完成。")

if __name__ == '__main__':
    main()
```

## User

下面请严格根据 PI 的要求输出最终结果。

【这是第一次会议，必须严格遵守以下额外约束】
1. 本次 Python 脚本的唯一目标是：运行模型、勘探输出目录、保存勘探报告。
2. 本次必须产出：data/stage1_observation.txt
3. 本次禁止做数据合并、禁止计算指标、禁止画图。
4. Python 脚本必须包含所有 import，必须定义 def main():。
【PI_PROMPT 原文】
你是一位顶级的计算生物学 PI。当前的评测任务是：【{task_desc}】。
请引导 MLOps 工程师写出评测的 Python 脚本和 Slurm 提交脚本。特别提醒工程师：
1. 严禁编写从 FASTA 提取标签的代码，必须直接读取 ground_truth.csv！
2. 合并前 ID 必须进行 `split()[0].strip()` 强清洗，合并后必须用 `fillna(0.0)` 兜底。
3. 寻找概率列时必须校验 `dtype == float`，避开读取英文字符串标签导致的 DataFrame 灾难！
4. 保存 JSON 前，一定要记得将所有指标通过 `float(val)` 转为 Python 原生类型，防止 Numpy 序列化报错！同时规定 JSON 的双层字典嵌套结构。
5. 评测脚本必须是一个完整可运行的 Python 脚本，而不能只有片段。


## Code Engineer

# 🛑 PI 审阅确认 (Stage 1: Exploration)

收到。作为 Code Engineer，我已明确本次 **Stage 1 探索勘探阶段** 的核心约束。
本次任务 **仅关注** 模型运行与输出结构勘探，**严禁** 涉及任何数据合并、指标计算或 Ground Truth 比对（这些属于后续阶段）。

我将严格遵循以下原则输出代码：
1.  **清理机制**：严格执行 `shutil.rmtree` 清理旧结果。
2.  **命令执行**：严格照抄 PepNet 调用模板，不吞咽错误。
3.  **勘探逻辑**：遍历 `data/*_out`，读取文件头（兼容 `.gz`），生成 `data/stage1_observation.txt`。
4.  **禁忌**：不使用 pandas 进行 merge，不计算指标，不保存 JSON 指标文件。

以下是最终完整代码。

---

# 📄 1. Python 评测脚本 (`stage1_exploration.py`)

```python
import os
import shutil
import subprocess
import glob
import gzip

def explore_output_directory(base_dir='data'):
    """
    勘探 data 目录下所有 *_out 相关的文件，读取头部内容。
    严格遵守 Stage 1 要求：不 merge，不计算指标，仅记录文件结构与内容摘要。
    """
    observation_log = []
    observation_log.append("=" * 60)
    observation_log.append("Stage 1: Model Output Observation Report")
    observation_log.append("=" * 60)
    observation_log.append("")

    # 查找所有 *_out 结尾的目录或文件
    pattern = os.path.join(base_dir, '*_out')
    found_items = glob.glob(pattern)
    
    if not found_items:
        observation_log.append("警告：未在 data/ 目录下找到任何 *_out 后缀的目录或文件。")
    else:
        # 排序以保证报告一致性
        found_items.sort()
        for item_path in found_items:
            observation_log.append(f"勘探对象：{item_path}")
            observation_log.append("-" * 60)
            
            files_to_read = []
            if os.path.isdir(item_path):
                # 如果是目录，遍历内部文件
                for root, dirs, files in os.walk(item_path):
                    # 排序文件以保证一致性
                    files.sort()
                    for file in files:
                        files_to_read.append(os.path.join(root, file))
            else:
                # 如果是文件
                files_to_read.append(item_path)
            
            for file_path in files_to_read:
                observation_log.append(f"  文件路径：{file_path}")
                try:
                    lines = []
                    # 处理 .gz 文件
                    if file_path.endswith('.gz'):
                        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if i >= 10: break
                                lines.append(line.strip())
                    else:
                        # 处理普通文本文件
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if i >= 10: break
                                lines.append(line.strip())
                    
                    if lines:
                        observation_log.append("  文件头内容摘要 (前 10 行):")
                        for line in lines:
                            # 防止日志中出现换行符破坏格式
                            safe_line = line.replace('\n', ' ').replace('\r', ' ')
                            observation_log.append(f"    | {safe_line}")
                    else:
                        observation_log.append("  文件头内容摘要：[文件为空]")
                except Exception as e:
                    observation_log.append(f"  读取失败：{str(e)}")
            
            observation_log.append("")

    observation_log.append("=" * 60)
    observation_log.append("End of Observation")
    observation_log.append("=" * 60)
    
    return "\n".join(observation_log)

def main():
    # 0. 确保 data 目录存在，以便后续保存报告
    os.makedirs('data', exist_ok=True)

    # 1. 【历史数据清理机制】
    # 在执行模型预测前，必须执行清理逻辑
    print("正在执行历史数据清理...")
    shutil.rmtree('data/PepNet_out', ignore_errors=True)
    print("清理完成。")

    # 2. 【强校验命令执行】
    # 必须严格照抄以下代码块来执行预测（切勿自行添加 try-except 吞咽错误）
    print("开始运行 PepNet...")
    PepNet_cmd = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate pepnet_env && python predict.py --input data/combined_test.fasta --output data/PepNet_out"'
    res_PepNet = subprocess.run(PepNet_cmd, shell=True, capture_output=True, text=True)
    if res_PepNet.returncode != 0:
        print(f"!!! PepNet 真实报错日志:\n{res_PepNet.stderr}")
        raise RuntimeError("PepNet 预测执行失败，已阻断程序！")
    
    print("PepNet 运行成功。")

    # 3. 【勘探逻辑】
    # 运行完模型后，遍历 data/ 目录下所有带有 _out 后缀的目录或新生成的文件
    print("开始勘探模型输出结构...")
    report_content = explore_output_directory('data')
    
    # 4. 【保存勘探报告】
    # 勘探报告必须保存为：data/stage1_observation.txt
    report_path = 'data/stage1_observation.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"勘探报告已保存至：{report_path}")
    print("Stage 1 任务完成。")

if __name__ == '__main__':
    main()
```

---

# 📄 2. Slurm 提交脚本 (`submit_stage1.sh`)

```bash
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
```

