## User

现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。

会议任务如下：
当前需要评测的计算生物学模型清单：Macrel, AMP-Scanner-v2, amPEPpy, AI4AMP, AMPlify。

请严格遵守以下编程规范，完全由你自主编写高质量的 Python 自动化评估脚本：

1. 【模型执行参数矩阵（并行 Job Array 模式）】：
你的 Python 脚本将被 Slurm Array 并行调用。请在脚本开头通过 `task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))` 获取当前任务索引。
根据 task_id 仅选取并执行以下列表中的【一个】对应模型（如果 task_id 超出范围请安全打印并退出）：
- 【模型名称】: Macrel
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"
  【输出目录】: data/Macrel_out

- 【模型名称】: AMP-Scanner-v2
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/020419_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"
  【输出目录】: data/AMP-Scanner-v2_out

- 【模型名称】: amPEPpy
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"
  【输出目录】: data/amPEPpy_out

- 【模型名称】: AI4AMP
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv"
  【输出目录】: data/AI4AMP_out

- 【模型名称】: AMPlify
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out"
  【输出目录】: data/AMPlify_out


🚨 注意：如果【预设运行命令】中执行的 Python 脚本已经是以 `/` 开头的绝对路径，你绝对不允许擅自去修改或拼接它！直接原样使用！只有当它使用的是相对路径，并且我为你提供了【真实源码目录树】时，你才需要结合目录树，以 `/share/home/zhangss/[模型名称]` 作为基准路径去修正它！

2. 【智能生命周期与动态自愈机制 (Self-Healing - 极度重要)】：
   生物信息学软件的脾气各不相同（有的不会自己建目录，有的讨厌目录已存在）。你必须为每个模型的执行编写极其强壮的容错逻辑：
   - **执行隔离**：必须用 `try...except Exception as e:` 独立包裹每一个模型。绝对禁止使用 `raise` 阻断主程序。
   - **第一步：清理与重建**：- 🚨【智能目录管理（防 Macrel 冲突死命令）】：执行命令前，先用 `shutil.rmtree('输出目录', ignore_errors=True)` 彻底删除历史脏数据。**【极度致命】绝对禁止提前执行 `os.makedirs` 创建空目录！** 因为像 Macrel 这样的模型，只要输出目录已存在就会直接报错退出！
   - 🚨【缺失自愈重试机制】：清理完旧目录后，直接用 subprocess 运行模型。如果运行失败（returncode != 0），必须在代码里检查错误输出（stderr 或 stdout）。如果错误信息中包含 `'No such file'`、`'NotFoundError'` 或 `'not found'`，说明该模型（如 AMP-Scanner）要求必须提前建好目录。此时你必须立刻在代码里执行 `os.makedirs('输出目录', exist_ok=True)`，然后**原地重新执行一次模型命令**！
   - **第二步：盲测**：直接使用 `subprocess.run(cmd, shell=True, capture_output=True, text=True)` 执行命令。
   - **第三步：动态自愈 (重试机制)**：如果 `returncode != 0`，你必须联合检查 `res.stdout` 和 `res.stderr` 的报错信息：
       * 如果报错信息包含 "No such file or directory" 或 "Failed to save"（这说明模型不会自己建目录）：请在 Python 中执行 `os.makedirs('输出目录', exist_ok=True)` 帮它建好房子，然后**再次重试**执行 `subprocess.run`！
       * 如果报错信息包含 "already exists" 或 "exists"（这说明模型极度讨厌预先存在的目录）：请在 Python 中执行 `shutil.rmtree('输出目录', ignore_errors=True)` 把刚才的文件夹删掉，然后**再次重试**执行 `subprocess.run`！
   - **第四步：终极日志**：如果重试后依然失败，必须将错误流完整追加写入该任务对应的独立观测文件中。

3. 【代码结构与完整性规范】：
   - 必须是一个单一的、可以直接运行的 Python 脚本。
   - 必须包含完整 import (`os, subprocess, shutil, glob` 等)。
   - 所有逻辑封装在 `def main():` 中，并使用 `if __name__ == '__main__':` 启动。


【这次是第一次会议（探索勘探阶段），请严格执行以下要求】：
1. 本次会议的唯一目标是：通过 Job Array 并行运行给定的模型，并“勘探”它们实际生成的文件结构。
2. 你的 Python 脚本在运行完分配给当前 `task_id` 的模型后，必须编写代码遍历该模型对应的 `data/XXX_out` 目录或新生成的文件：
   - 使用 `glob` 或 `os.listdir` 找到模型实际生成的所有结果文件（特别是未知的 .gz, .csv, .txt）。
   - 尝试读取这些文件的头 5 到 10 行内容（如果发现是 .gz 文件，请 import gzip 读取纯文本）。
   - 将你找到的文件绝对/相对路径，以及文件的头部内容摘要，写出到一份独立的勘探报告中。
3. 🚨 **隔离写入死纪律（极其致命）**：
   - 绝对禁止使用 `fcntl` 或任何文件锁机制！
   - 绝对禁止将结果写入统一的 `stage1_observation.txt`！
   - 你必须强制按 `task_id` 将报告保存为独立的文件！你的代码中必须出现这一行精确的赋值语句：
     `log_file = f"data/stage1_obs_{task_id}.txt"`
   如果你违背此项，系统将无法取回碎片文件并导致整个评测管线当场崩溃！
4. 绝对禁止（重要！）：本次会议的脚本绝对不许用 pandas 做任何 merge 操作！不许清洗数据！不许计算指标！
5. 严禁生成只有 CPU 资源的 sbatch 脚本。

## PI

# 会议记录：AMP 模型自动化评估管线（阶段一：勘探与执行）

## 🟢 PI 审阅与要求确认 (Principal Investigator Review)

各位同事，针对本次 AMP 模型评测任务，作为 PI，我必须再次强调以下**不可逾越的红线**。这直接关系到我们后续数据整合的成败以及计算资源的浪费。请代码工程师在编写脚本时，逐条核对以下约束：

### 1. 任务调度与隔离性 (Job Array & Isolation)
*   **Slurm 集成**：脚本必须通过 `SLURM_ARRAY_TASK_ID` 环境变量识别当前任务。
*   **一对一映射**：`task_id` 0-4 分别对应 Macrel, AMP-Scanner-v2, amPEPpy, AI4AMP, AMPlify。**严禁**在一个任务中循环运行所有模型。
*   **越界保护**：若 `task_id` 超出 [0, 4] 范围，必须优雅退出并打印警告，**严禁**抛出未捕获异常导致 Slurm 任务状态混乱。

### 2. 命令完整性与路径安全 (Command Integrity)
*   **绝对路径锁死**：提供的运行命令中，凡是涉及 `/share/home/zhangss/...` 的绝对路径，**一字不改**。不要尝试拼接或修正，除非明确告知是相对路径。
*   **Conda 环境激活**：命令中已包含 `source ... && conda activate ...`，保持原样，确保依赖环境正确加载。

### 3. 智能自愈机制 (Self-Healing Protocol)
这是本脚本的核心难点，必须严格执行以下顺序：
1.  **预清理**：运行前，使用 `shutil.rmtree` 强制删除目标输出目录（防止 Macrel 因目录存在而崩溃）。**禁止**预先 `os.makedirs`。
2.  **首次盲测**：直接 `subprocess.run`。
3.  **动态诊断与重试**：
    *   若失败且报错含 `"already exists"` -> 执行 `rmtree` -> **重试**。
    *   若失败且报错含 `"No such file"` -> 执行 `os.makedirs` -> **重试**。
    *   其他错误 -> 记录日志，终止该模型任务，**不阻断主程序**。

### 4. 勘探报告隔离写入 (Isolated Observation Logging)
*   **文件命名铁律**：每个任务的观察报告必须独立保存。代码中必须出现且仅出现这一行赋值逻辑：
    ```python
    log_file = f"data/stage1_obs_{task_id}.txt"
    ```
*   **内容规范**：仅记录文件路径、类型及头部内容摘要。**严禁**在此阶段进行任何 Pandas 合并、清洗或指标计算。
*   **无锁机制**：禁止使用 `fcntl` 等文件锁，依靠 Slurm Array 的任务隔离性保证写入安全。

### 5. 代码交付标准
*   单一 Python 文件，可直接 `python script.py` 或在 Slurm 中调用。
*   包含完整的 `import` 和 `if __name__ == '__main__':` 入口。
*   处理 `.gz` 文件的读取需引入 `gzip` 模块。

---

## 🔵 代码工程师交付 (Engineer Implementation)

收到 PI 指令。以下是严格遵循上述约束编写的 Python 自动化评估与勘探脚本。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMP Model Evaluation Pipeline - Stage 1: Execution & Exploration
Author: Engineer
Date: 2023-10-27
Constraints: Slurm Array Mode, Self-Healing, Isolated Logging, No Pandas Merge
"""

import os
import sys
import subprocess
import shutil
import glob
import gzip
import time

# -----------------------------------------------------------------------------
# Configuration: Model Definitions
# -----------------------------------------------------------------------------
MODELS = [
    {
        "name": "Macrel",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives\"",
        "out_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/020419_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv\"",
        "out_dir": "data/AMP-Scanner-v2_out"
    },
    {
        "name": "amPEPpy",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model\"",
        "out_dir": "data/amPEPpy_out"
    },
    {
        "name": "AI4AMP",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv\"",
        "out_dir": "data/AI4AMP_out"
    },
    {
        "name": "AMPlify",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out\"",
        "out_dir": "data/AMPlify_out"
    }
]

MAX_RETRIES = 2

def get_head_content(file_path, max_lines=10):
    """Read head content handling gzip compression."""
    lines = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())
        return "\n".join(lines)
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

def run_model_with_healing(model_config, task_id):
    """Execute model with self-healing directory management."""
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["out_dir"]
    
    print(f"[Task {task_id}] Starting execution for {name}...")
    
    # Step 1: Pre-cleanup (Critical for Macrel compatibility)
    # DO NOT use os.makedirs here. Only remove if exists.
    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[Task {task_id}] Cleaned existing output directory: {out_dir}")
        except Exception as e:
            print(f"[Task {task_id}] Warning: Failed to clean directory {out_dir}: {e}")

    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            res = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=600 # 10 min timeout per model
            )
            
            if res.returncode == 0:
                print(f"[Task {task_id}] Success: {name} completed.")
                return True
            
            # Analyze Error for Self-Healing
            error_msg = res.stderr + res.stdout
            last_error = error_msg
            
            print(f"[Task {task_id}] Attempt {attempt+1} failed for {name}. Return code: {res.returncode}")
            
            # Healing Logic
            if "already exists" in error_msg.lower() or "exists" in error_msg.lower():
                print(f"[Task {task_id}] Healing: Directory exists conflict detected. Removing and retrying...")
                shutil.rmtree(out_dir, ignore_errors=True)
                continue
                
            elif "No such file" in error_msg or "NotFoundError" in error_msg or "not found" in error_msg:
                print(f"[Task {task_id}] Healing: Missing directory detected. Creating and retrying...")
                os.makedirs(out_dir, exist_ok=True)
                continue
            
            else:
                # Unknown error, stop retrying
                print(f"[Task {task_id}] Fatal Error for {name}: {error_msg[:200]}...")
                break
                
        except subprocess.TimeoutExpired:
            print(f"[Task {task_id}] Timeout for {name}.")
            last_error = "Timeout"
            break
        except Exception as e:
            print(f"[Task {task_id}] Exception during run: {e}")
            last_error = str(e)
            break

    # Final Failure Logging
    if last_error:
        err_log_path = f"data/error_task_{task_id}.log"
        os.makedirs("data", exist_ok=True)
        with open(err_log_path, "a") as f:
            f.write(f"--- Task {task_id} ({name}) Failed ---\n")
            f.write(last_error + "\n\n")
        return False
        
    return True

def explore_output(task_id, model_config):
    """Explore generated files and write isolated observation report."""
    out_dir = model_config["out_dir"]
    
    # Ensure data directory exists for log file
    os.makedirs("data", exist_ok=True)
    
    # 🚨 ISOLATION LOGIC: Strictly follow the variable assignment requirement
    log_file = f"data/stage1_obs_{task_id}.txt"
    
    report_lines = []
    report_lines.append(f"=== Stage 1 Exploration Report ===")
    report_lines.append(f"Task ID: {task_id}")
    report_lines.append(f"Model: {model_config['name']}")
    report_lines.append(f"Target Dir: {out_dir}")
    report_lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50 + "\n")
    
    if not os.path.exists(out_dir):
        report_lines.append(f"[WARNING] Output directory does not exist: {out_dir}")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        return

    # Traverse and find files
    found_files = []
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            full_path = os.path.join(root, file)
            found_files.append(full_path)
    
    if not found_files:
        report_lines.append("[INFO] No files found in output directory.")
    else:
        report_lines.append(f"[INFO] Found {len(found_files)} files:")
        for f_path in sorted(found_files):
            rel_path = os.path.relpath(f_path, "data")
            size = os.path.getsize(f_path)
            header = get_head_content(f_path)
            
            report_lines.append(f"\nFile: {rel_path}")
            report_lines.append(f"Size: {size} bytes")
            report_lines.append(f"Head Content:\n{header}")
            report_lines.append("-" * 30)

    # Write Isolated Log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[Task {task_id}] Exploration report saved to: {log_file}")

def main():
    # 1. Get Task ID from Environment
    try:
        task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        task_id = int(task_id_str)
    except ValueError:
        print("Error: SLURM_ARRAY_TASK_ID is not a valid integer. Exiting.")
        sys.exit(1)

    # 2. Validate Range
    if task_id < 0 or task_id >= len(MODELS):
        print(f"Warning: Task ID {task_id} is out of range [0-{len(MODELS)-1}]. Exiting safely.")
        # Create a minimal log indicating skip
        os.makedirs("data", exist_ok=True)
        log_file = f"data/stage1_obs_{task_id}.txt"
        with open(log_file, "w") as f:
            f.write(f"Task {task_id} skipped: Out of range.\n")
        sys.exit(0)

    # 3. Select Model
    current_model = MODELS[task_id]
    print(f"Selected Model for Task {task_id}: {current_model['name']}")

    # 4. Execute with Healing
    success = run_model_with_healing(current_model, task_id)

    # 5. Explore Output (Even if failed, check what happened)
    explore_output(task_id, current_model)

    print(f"Task {task_id} finished.")

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
5. 必须同时输出完整的 Bash 提交脚本，文件名为 stage1_run_model.sh。
6. stage1_run_model.sh 必须显式申请 GPU，且必须包含：
   #SBATCH --gres=gpu:{SLURM_GPUS}
7. 如果使用 Job Array，则每个 array task 至少申请 1 张 GPU；禁止生成不申请 GPU 的脚本。

【PI_PROMPT 原文】
你是一位顶级的计算生物学 PI。当前的评测任务是：【{task_desc}】。
请引导 MLOps 工程师写出评测的 Python 脚本和 Slurm 提交脚本。特别提醒工程师：
1. 严禁编写从 FASTA 提取标签的代码，必须直接读取 ground_truth.csv！
2. 🚨 【强力清洗与合并防线】：合并前，真值表和模型预测的 ID/Sequence 列必须进行极度暴力的强清洗：转字符串、去首尾空格、转大写、并强制剔除 FASTA 的 '>' 符号（`.astype(str).str.strip().str.upper().str.replace('>', '', regex=False)`）。
3. 🚨 【禁止掩盖 Bug】：绝对禁止在 pd.merge 后盲目使用 `fillna(0.0)` 兜底！如果合并后预测列全是 NaN，必须直接报错抛出异常，绝不接受假数据！
4. 保存 JSON 前，一定要记得将所有指标通过 `float(val)` 转为 Python 原生类型，防止 Numpy 序列化报错！同时规定 JSON 的双层字典嵌套结构。
5. 评测脚本必须是一个完整可运行的 Python 脚本，而不能只有片段。

【CODER_PROMPT 原文】
你是一位精通超算的 MLOps 工程师。根据 PI 的要求编写评测代码。

【代码输出的格式底线（极其重要）】：
1. 你提供的 Python 代码必须是一个**单一的、完全完整的脚本**，全部包含在一个 ` ```python ... ``` ` 代码块中！绝对不允许把代码切分成多个小块。
2. 脚本必须包含所有的 `import`，必须有明确定义的 `def main():` 函数，并以 `if __name__ == '__main__':` 作为程序入口去调用 `main()`。
3. 🚨 **基础语法死纪律**：**绝对不允许在定义变量名时包含空格！**
4. 🚨 **零缩进死命令（针对 SyntaxError）**：生成的脚本中，所有顶级代码（如 `import` 语句、`def` 定义、`if __name__ == "__main__":`）**必须从每一行的第 1 个字符（第 0 列）开始编写**。绝对严禁在脚本开头或顶级语句前添加任何空格或制表符缩进！
5. 🚨 **防中文符号污染死命令（极度致命）**：你写的是纯 Python 代码，**绝对禁止在任何 Python 语句末尾或语法部位使用中文标点符号（尤其是中文句号 `。` 和中文逗号 `，`）！** 你的代码如果出现 `invalid character '。'` 会导致系统当场崩溃！

【评测功能要求】：
1. 必须保存严格嵌套结构的 `eval_result.json`，以及 `evaluation_curves.png` 和 `final_results_with_predictions.csv`。
2. 🚨【绝对禁止静默掩盖 Bug】：在调用 `pd.merge` 或 `map` 映射数据后，必须检查预测列的 NaN 比例！如果全部是 NaN，说明合并彻底失败，必须将该模型的输出设为 NaN 而不是 0.0，并在日志中大声报错！
3. 🚨【Sklearn 指标计算防崩溃装甲】：在调用 `roc_auc_score` 等指标前，必须检查 `len(np.unique(y_true)) > 1`。如果不满足（比如单类别数据），或者预测值全为空/常数，请利用 try-except 捕获 ValueError，并将发生崩溃的指标（如 AUROC、AUPRC）安全置为 `NaN`（`float('nan')`），绝对不允许整个脚本因此崩溃！
4. 必须提供 `run_eval.sh`，Bash 脚本必须写在单独的 ` ```bash ... ``` ` 块中。
   🚨 **针对并行运算的 Bash 模板强制要求**：为了实现一次最多调用 4 张显卡的并行阵列，你必须原样使用以下模板：
   ```bash
   #!/bin/bash
   #SBATCH -J amp_eval
   #SBATCH --array=0-[MAX_INDEX]%4   # 极度重要: 智能替换为 (模型总数量 - 1)
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
   python eval_script.py
   echo "finish"
   ```

## Code Engineer

```python
import os
import sys
import subprocess
import shutil
import glob
import gzip
import time

MODELS = [
    {
        "name": "Macrel",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives\"",
        "out_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/020419_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv\"",
        "out_dir": "data/AMP-Scanner-v2_out"
    },
    {
        "name": "amPEPpy",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model\"",
        "out_dir": "data/amPEPpy_out"
    },
    {
        "name": "AI4AMP",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv\"",
        "out_dir": "data/AI4AMP_out"
    },
    {
        "name": "AMPlify",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out\"",
        "out_dir": "data/AMPlify_out"
    }
]

MAX_RETRIES = 2

def get_head_content(file_path, max_lines=10):
    lines = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())
        return "\n".join(lines)
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

def run_model_with_healing(model_config, task_id):
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["out_dir"]
    
    print(f"[Task {task_id}] Starting execution for {name}...")
    
    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[Task {task_id}] Cleaned existing output directory: {out_dir}")
        except Exception as e:
            print(f"[Task {task_id}] Warning: Failed to clean directory {out_dir}: {e}")

    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            res = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=600
            )
            
            if res.returncode == 0:
                print(f"[Task {task_id}] Success: {name} completed.")
                return True
            
            error_msg = res.stderr + res.stdout
            last_error = error_msg
            
            print(f"[Task {task_id}] Attempt {attempt+1} failed for {name}. Return code: {res.returncode}")
            
            if "already exists" in error_msg.lower() or "exists" in error_msg.lower():
                print(f"[Task {task_id}] Healing: Directory exists conflict detected. Removing and retrying...")
                shutil.rmtree(out_dir, ignore_errors=True)
                continue
                
            elif "No such file" in error_msg or "NotFoundError" in error_msg or "not found" in error_msg:
                print(f"[Task {task_id}] Healing: Missing directory detected. Creating and retrying...")
                os.makedirs(out_dir, exist_ok=True)
                continue
            
            else:
                print(f"[Task {task_id}] Fatal Error for {name}: {error_msg[:200]}...")
                break
                
        except subprocess.TimeoutExpired:
            print(f"[Task {task_id}] Timeout for {name}.")
            last_error = "Timeout"
            break
        except Exception as e:
            print(f"[Task {task_id}] Exception during run: {e}")
            last_error = str(e)
            break

    if last_error:
        err_log_path = f"data/error_task_{task_id}.log"
        os.makedirs("data", exist_ok=True)
        with open(err_log_path, "a") as f:
            f.write(f"--- Task {task_id} ({name}) Failed ---\n")
            f.write(last_error + "\n\n")
        return False
        
    return True

def explore_output(task_id, model_config):
    out_dir = model_config["out_dir"]
    
    os.makedirs("data", exist_ok=True)
    
    # 🚨 ISOLATION LOGIC: Strictly follow the variable assignment requirement
    log_file = f"data/stage1_obs_{task_id}.txt"
    
    report_lines = []
    report_lines.append(f"=== Stage 1 Exploration Report ===")
    report_lines.append(f"Task ID: {task_id}")
    report_lines.append(f"Model: {model_config['name']}")
    report_lines.append(f"Target Dir: {out_dir}")
    report_lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50 + "\n")
    
    if not os.path.exists(out_dir):
        report_lines.append(f"[WARNING] Output directory does not exist: {out_dir}")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        return

    found_files = []
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            full_path = os.path.join(root, file)
            found_files.append(full_path)
    
    if not found_files:
        report_lines.append("[INFO] No files found in output directory.")
    else:
        report_lines.append(f"[INFO] Found {len(found_files)} files:")
        for f_path in sorted(found_files):
            rel_path = os.path.relpath(f_path, "data")
            size = os.path.getsize(f_path)
            header = get_head_content(f_path)
            
            report_lines.append(f"\nFile: {rel_path}")
            report_lines.append(f"Size: {size} bytes")
            report_lines.append(f"Head Content:\n{header}")
            report_lines.append("-" * 30)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[Task {task_id}] Exploration report saved to: {log_file}")

def main():
    try:
        task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        task_id = int(task_id_str)
    except ValueError:
        print("Error: SLURM_ARRAY_TASK_ID is not a valid integer. Exiting.")
        sys.exit(1)

    if task_id < 0 or task_id >= len(MODELS):
        print(f"Warning: Task ID {task_id} is out of range [0-{len(MODELS)-1}]. Exiting safely.")
        os.makedirs("data", exist_ok=True)
        log_file = f"data/stage1_obs_{task_id}.txt"
        with open(log_file, "w") as f:
            f.write(f"Task {task_id} skipped: Out of range.\n")
        sys.exit(0)

    current_model = MODELS[task_id]
    print(f"Selected Model for Task {task_id}: {current_model['name']}")

    success = run_model_with_healing(current_model, task_id)

    explore_output(task_id, current_model)

    print(f"Task {task_id} finished.")

if __name__ == '__main__':
    main()
```

```bash
#!/bin/bash
#SBATCH -J amp_stage1
#SBATCH --array=0-4%4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o stage1_%A_%a.out
#SBATCH -e stage1_%A_%a.err

cd /share/home/zhangss/vlab_workspace
source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
conda activate base
python stage1_explore.py
echo "finish"
```

