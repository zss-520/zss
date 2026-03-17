## User

现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。

会议任务如下：
当前需要评测的计算生物学模型清单：Macrel, AMP-Scanner-v2。

请严格遵守以下编程规范，完全由你自主编写高质量的 Python 自动化评估脚本：

1. 【模型执行参数矩阵】：
你需要通过代码依次执行以下模型：
- 【模型名称】: Macrel
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out"
  【输出目录】: data/Macrel_out

- 【模型名称】: AMP-Scanner-v2
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"
  【输出目录】: data/AMP-Scanner-v2_out



2. 【智能生命周期与动态自愈机制 (Self-Healing - 极度重要)】：
   生物信息学软件的脾气各不相同（有的不会自己建目录，有的讨厌目录已存在）。你必须为每个模型的执行编写极其强壮的容错逻辑：
   - **执行隔离**：必须用 `try...except Exception as e:` 独立包裹每一个模型。绝对禁止使用 `raise` 阻断主程序。
   - **第一步：清理**：执行前先用 `shutil.rmtree('输出目录', ignore_errors=True)` 清理上一轮的历史脏数据。
   - **第二步：盲测**：直接使用 `subprocess.run(cmd, shell=True, capture_output=True, text=True)` 执行命令。
   - **第三步：动态自愈 (重试机制)**：如果 `returncode != 0`，你必须联合检查 `res.stdout` 和 `res.stderr` 的报错信息：
       * 如果报错信息包含 "No such file or directory" 或 "Failed to save"（这说明模型不会自己建目录）：请在 Python 中执行 `os.makedirs('输出目录', exist_ok=True)` 帮它建好房子，然后**再次重试**执行 `subprocess.run`！
       * 如果报错信息包含 "already exists" 或 "exists"（这说明模型极度讨厌预先存在的目录）：请在 Python 中执行 `shutil.rmtree('输出目录', ignore_errors=True)` 把刚才的文件夹删掉，然后**再次重试**执行 `subprocess.run`！
   - **第四步：终极日志**：如果重试后依然失败，必须将错误流完整追加写入 `data/stage1_observation.txt` 文件中。

3. 【代码结构与完整性规范】：
   - 必须是一个单一的、可以直接运行的 Python 脚本。
   - 必须包含完整 import (`os, subprocess, shutil, glob` 等)。
   - 所有逻辑封装在 `def main():` 中，并使用 `if __name__ == '__main__':` 启动。


【这次是第一次会议（探索勘探阶段），请严格执行以下要求】：
1. 本次会议的唯一目标是：运行给定列表中的所有模型，并“勘探”它们实际生成的文件结构。
2. 你的 Python 脚本在运行完模型（subprocess）后，必须编写代码遍历 `data/` 目录下所有带有 `_out` 后缀的目录或新生成的文件：
   - 使用 `glob` 或 `os.listdir` 找到模型实际生成的所有结果文件（特别是未知的 .gz, .csv, .txt）。
   - 尝试读取这些文件的头 5 到 10 行内容（如果发现是 .gz 文件，请 import gzip 读取纯文本）。
   - 将你找到的文件绝对/相对路径，以及文件的头部内容摘要，写出到一份勘探报告中。
3. 勘探报告必须保存为：`data/stage1_observation.txt`。
4. 绝对禁止（重要！）：本次会议的脚本绝对不许用 pandas 做任何 merge 操作！不许清洗数据！不许计算指标！
【🔴 极度重要的工业级容错与执行纪律 (Fatal Error Prevention)】
当你编写执行模型命令的 Python 脚本时，必须遵循以下防御性编程规范：

1. **绝对隔离，永不连坐**：必须使用 `try...except Exception as e:` 独立包裹每一个模型的执行逻辑。如果某个模型崩溃，必须将其错误流写入 `observation.txt`，并在终端打印警告，然后**强制继续执行下一个模型**。
2. **智能目录探测与重试机制 (Self-Healing) - 极其关键**：
   生物学模型的脾气非常古怪，你必须在代码中实现以下完美的重试逻辑：
   - **执行前兜底**：在执行 `subprocess.run` 前，必须先执行 `os.makedirs([output_dir], exist_ok=True)` 提前建好输出文件夹。
   - **联合错误嗅探**：如果 `returncode != 0`，你必须**同时去 `res.stdout` 和 `res.stderr` 中寻找错误关键字**（因为很多老代码会把致命错误打在 stdout 里！）。
   - **解决 "already exists" 冲突**：如果报错信息包含 "already exists" 或 "exists" (如 Macrel 极度讨厌已存在的文件夹)，请在异常处理逻辑中，用 `shutil.rmtree([output_dir])` 将刚才建的空文件夹直接删掉，然后原封不动地重新再执行一次 `subprocess.run`，它就会成功！
3. **安全命令拼接**：使用 `subprocess.run(cmd, shell=True, capture_output=True, text=True)` 来捕获完整的标准输出和标准错误。

## PI

# 🧬 计算生物学模型评估会议记录 (Stage 1: 勘探)

## 1. PI 审阅与任务要求重申 (PI Review & Requirements)

**致代码工程师：**

我已审阅本次 Stage 1 勘探任务。我们的目标不是立即获得完美的评估指标，而是**确保流程的鲁棒性**并**摸清模型输出的真实结构**。生物信息学工具链极其脆弱，环境依赖复杂，因此本次脚本的核心不在于“快”，而在于“稳”。

请严格遵守以下**硬性约束 (Hard Requirements)**，任何偏离都将导致实验不可复现：

1.  **绝对隔离原则 (Absolute Isolation)**：
    *   每个模型的执行必须被独立的 `try...except Exception` 块包裹。
    *   **禁止**因为 Macrel 失败而导致 AMP-Scanner 不运行。任何异常必须被捕获、记录，然后程序继续。
    *   **禁止**使用 `raise` 中断主流程。

2.  **智能自愈机制 (Self-Healing Mechanism)**：
    *   这是本次脚本的灵魂。你不能假设目录状态是干净的。
    *   **前置清理**：执行前必须 `shutil.rmtree` 输出目录。
    *   **错误嗅探**：执行失败后，必须同时检查 `stdout` 和 `stderr`。
    *   **动态重试**：
        *   遇 "No such file..." -> Python 侧 `makedirs` -> 重试。
        *   遇 "already exists" -> Python 侧 `rmtree` -> 重试。
    *   所有最终失败的错误流必须完整写入 `data/stage1_observation.txt`。

3.  **勘探报告规范 (Exploration Protocol)**：
    *   模型运行结束后，必须遍历 `data/` 目录。
    *   识别所有 `_out` 目录及新生成文件。
    *   **读取文件头**：纯文本直接读，`.gz` 文件必须用 `gzip` 模块解压读取。
    *   报告内容：文件路径 + 前 10 行内容摘要。
    *   **禁止事项**：本阶段**严禁**引入 `pandas`，**严禁**做任何数据 merge 或指标计算。我们只要原始观测记录。

4.  **工程规范**：
    *   单文件脚本，包含所有必要 import。
    *   入口为 `if __name__ == '__main__':`。
    *   确保 `data/` 目录存在后再写入日志。

请基于以上要求，输出最终的可执行代码。

---

## 2. 工程师最终实现代码 (Engineer Implementation)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Model Execution & Structure Exploration Script
Target Models: Macrel, AMP-Scanner-v2
Constraint: No pandas, No metrics, Robust Self-Healing, Isolated Execution.
"""

import os
import subprocess
import shutil
import glob
import gzip
import sys

# ==============================================================================
# Configuration & Constants
# ==============================================================================

DATA_DIR = "data"
OBSERVATION_LOG = os.path.join(DATA_DIR, "stage1_observation.txt")

# Model Definition Matrix
MODELS = [
    {
        "name": "Macrel",
        "output_dir": os.path.join(DATA_DIR, "Macrel_out"),
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out"'
    },
    {
        "name": "AMP-Scanner-v2",
        "output_dir": os.path.join(DATA_DIR, "AMP-Scanner-v2_out"),
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"'
    }
]

# ==============================================================================
# Helper Functions
# ==============================================================================

def ensure_data_dir():
    """Ensure the base data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message, mode='a'):
    """Append message to the observation log."""
    try:
        with open(OBSERVATION_LOG, mode, encoding='utf-8', errors='ignore') as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[CRITICAL] Failed to write to log: {e}")

def read_file_head(file_path, lines=10):
    """Read the first N lines of a file, handling .gz compression."""
    content_lines = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_lines.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_lines.append(line.strip())
    except Exception as e:
        return [f"[Error reading file: {str(e)}]"]
    return content_lines

def run_model_with_healing(model_config):
    """
    Execute a model with strict self-healing logic.
    1. Clean output dir.
    2. Run command.
    3. If fail, check errors and retry based on keywords.
    4. Log final failures.
    """
    name = model_config["name"]
    out_dir = model_config["output_dir"]
    cmd = model_config["cmd"]
    
    print(f"[INFO] Starting execution for model: {name}")
    log_message(f"=== Model Execution: {name} ===")
    
    try:
        # Step 1: Clean (Pre-execution hygiene)
        # Even if it doesn't exist, ignore_errors=True prevents crashes
        shutil.rmtree(out_dir, ignore_errors=True)
        
        # Step 2: Blind Test (First Attempt)
        # Ensure parent dir exists just in case
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        
        res = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=3600  # 1 hour timeout safety
        )
        
        # Step 3: Dynamic Self-Healing (If returncode != 0)
        if res.returncode != 0:
            combined_output = res.stdout + res.stderr
            print(f"[WARN] {name} failed initially. Analyzing errors...")
            
            retry_needed = False
            retry_action = ""
            
            # Check for specific error patterns
            if "No such file or directory" in combined_output or "Failed to save" in combined_output:
                # Model complains about missing directory -> Create it
                print(f"[HEALING] {name} needs directory creation. Executing makedirs...")
                os.makedirs(out_dir, exist_ok=True)
                retry_needed = True
                retry_action = "mkdir"
                
            elif "already exists" in combined_output or "exists" in combined_output:
                # Model complains about existing directory -> Remove it
                print(f"[HEALING] {name} hates existing directory. Executing rmtree...")
                shutil.rmtree(out_dir, ignore_errors=True)
                # Re-create empty dir for consistency before retry
                os.makedirs(out_dir, exist_ok=True) 
                retry_needed = True
                retry_action = "rmtree"
            
            if retry_needed:
                # Retry Execution
                print(f"[RETRY] Retrying {name} after {retry_action}...")
                res = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=3600
                )
        
        # Step 4: Ultimate Log (If still failed after retry)
        if res.returncode != 0:
            error_msg = f"[FAILURE] Model {name} failed after healing attempts.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n"
            print(f"[ERROR] {name} execution failed permanently.")
            log_message(error_msg)
        else:
            print(f"[SUCCESS] {name} completed successfully.")
            log_message(f"[SUCCESS] Model {name} completed successfully.\n")

    except Exception as e:
        # Absolute Isolation: Catch ANY exception to prevent blocking main program
        error_msg = f"[CRITICAL EXCEPTION] Model {name} crashed with exception: {str(e)}\n"
        print(f"[CRITICAL] {name} crashed: {e}")
        log_message(error_msg)

def explore_outputs():
    """
    Traverse data/ directory, find output files, read headers, and report.
    """
    print("[INFO] Starting output structure exploration...")
    log_message("\n=== Stage 1 Exploration Report ===\n")
    
    # Ensure data dir exists before globbing
    if not os.path.exists(DATA_DIR):
        log_message("[ERROR] Data directory does not exist.")
        return

    # Find all directories ending with _out
    out_dirs = glob.glob(os.path.join(DATA_DIR, "*_out"))
    # Also find any files directly in data/ that might be outputs (though less likely per spec)
    # Focusing on *_out directories as per task description
    
    found_files = []
    for d in out_dirs:
        if os.path.isdir(d):
            # Walk through the directory
            for root, dirs, files in os.walk(d):
                for file in files:
                    found_files.append(os.path.join(root, file))
    
    # Also check for any specific files mentioned in commands that might be outside *_out dirs 
    # (e.g. if a model dumps a log in data/ directly)
    # But sticking to *_out structure primarily as per task "traverse data/ ... with _out suffix"
    
    if not found_files:
        log_message("[INFO] No output files found in *_out directories.")
        # Fallback: Check any file in data/ just in case
        for f in os.listdir(DATA_DIR):
            full_path = os.path.join(DATA_DIR, f)
            if os.path.isfile(full_path) and not f.endswith('.txt'): # Avoid log itself
                found_files.append(full_path)

    for file_path in found_files:
        log_message(f"File Path: {file_path}")
        head_content = read_file_head(file_path, lines=10)
        log_message("Header Content (First 10 lines):")
        for line in head_content:
            log_message(f"  {line}")
        log_message("-" * 50)

    print(f"[INFO] Exploration complete. Report saved to {OBSERVATION_LOG}")

# ==============================================================================
# Main Execution Flow
# ==============================================================================

def main():
    print("=" * 60)
    print("Stage 1: Model Execution & Exploration Pipeline")
    print("=" * 60)
    
    # 0. Initialization
    ensure_data_dir()
    
    # Clear previous observation log for fresh start (optional, but good for stage 1)
    # Using 'w' mode initially to reset, then subsequent logs use 'a'
    if os.path.exists(OBSERVATION_LOG):
        os.remove(OBSERVATION_LOG)
    log_message(f"Pipeline Start Time: {subprocess.run('date', shell=True, capture_output=True, text=True).stdout.strip()}")
    
    # 1. Execute Models (Isolated & Self-Healing)
    for model in MODELS:
        run_model_with_healing(model)
        # Small delay to ensure file system sync
        subprocess.run("sleep 1", shell=True)
        
    # 2. Explore Outputs (Reconnaissance)
    explore_outputs()
    
    log_message(f"Pipeline End Time: {subprocess.run('date', shell=True, capture_output=True, text=True).stdout.strip()}")
    print("=" * 60)
    print("Pipeline Finished. Check data/stage1_observation.txt for details.")
    print("=" * 60)

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

# 计算生物学模型评估 - Stage 1 勘探任务交付

## 1. Python 自动化评估脚本 (`stage1_exploration.py`)

本脚本严格遵循 Stage 1 勘探目标：**仅执行模型、勘探文件结构、生成观测报告**。
- ✅ **完全隔离**：每个模型独立 `try...except` 包裹。
- ✅ **智能自愈**：包含目录清理、错误嗅探、动态重试机制。
- ✅ **禁止事项**：未引入 `pandas`，未计算指标，未合并数据。
- ✅ **完整结构**：包含所有 import，`def main()` 入口。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Model Execution & Structure Exploration Script
Target Models: Macrel, AMP-Scanner-v2
Constraint: No pandas, No metrics, No plotting. Robust Self-Healing.
Output: data/stage1_observation.txt
"""

import os
import subprocess
import shutil
import glob
import gzip
import sys
import datetime

# ==============================================================================
# Configuration & Constants
# ==============================================================================

DATA_DIR = "data"
OBSERVATION_LOG = os.path.join(DATA_DIR, "stage1_observation.txt")

# Model Definition Matrix
MODELS = [
    {
        "name": "Macrel",
        "output_dir": os.path.join(DATA_DIR, "Macrel_out"),
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out"'
    },
    {
        "name": "AMP-Scanner-v2",
        "output_dir": os.path.join(DATA_DIR, "AMP-Scanner-v2_out"),
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"'
    }
]

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_data_dir():
    """Ensure the base data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message, mode='a'):
    """Append message to the observation log."""
    try:
        # Ensure data dir exists before writing
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OBSERVATION_LOG, mode, encoding='utf-8', errors='ignore') as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[CRITICAL] Failed to write to log: {e}")

def read_file_head(file_path, lines=10):
    """
    Read the first N lines of a file.
    Handles .gz compression automatically.
    Returns a list of strings.
    """
    content_lines = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_lines.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_lines.append(line.strip())
    except Exception as e:
        return [f"[Error reading file: {str(e)}]"]
    return content_lines

def run_model_with_healing(model_config):
    """
    Execute a model with strict self-healing logic.
    1. Clean output dir.
    2. Run command.
    3. If fail, check errors and retry based on keywords.
    4. Log final failures.
    """
    name = model_config["name"]
    out_dir = model_config["output_dir"]
    cmd = model_config["cmd"]
    
    print(f"[INFO] Starting execution for model: {name} at {get_timestamp()}")
    log_message(f"\n=== Model Execution: {name} ===\nTime: {get_timestamp()}")
    
    try:
        # Step 1: Clean (Pre-execution hygiene)
        # Even if it doesn't exist, ignore_errors=True prevents crashes
        shutil.rmtree(out_dir, ignore_errors=True)
        
        # Step 2: Blind Test (First Attempt)
        # Ensure parent dir exists just in case
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        
        res = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=3600  # 1 hour timeout safety
        )
        
        # Step 3: Dynamic Self-Healing (If returncode != 0)
        if res.returncode != 0:
            combined_output = res.stdout + res.stderr
            print(f"[WARN] {name} failed initially. Analyzing errors...")
            
            retry_needed = False
            retry_action = ""
            
            # Check for specific error patterns
            if "No such file or directory" in combined_output or "Failed to save" in combined_output:
                # Model complains about missing directory -> Create it
                print(f"[HEALING] {name} needs directory creation. Executing makedirs...")
                os.makedirs(out_dir, exist_ok=True)
                retry_needed = True
                retry_action = "mkdir"
                
            elif "already exists" in combined_output or "exists" in combined_output:
                # Model complains about existing directory -> Remove it
                print(f"[HEALING] {name} hates existing directory. Executing rmtree...")
                shutil.rmtree(out_dir, ignore_errors=True)
                # Re-create empty dir for consistency before retry
                os.makedirs(out_dir, exist_ok=True) 
                retry_needed = True
                retry_action = "rmtree"
            
            if retry_needed:
                # Retry Execution
                print(f"[RETRY] Retrying {name} after {retry_action}...")
                res = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=3600
                )
        
        # Step 4: Ultimate Log (If still failed after retry)
        if res.returncode != 0:
            error_msg = f"[FAILURE] Model {name} failed after healing attempts.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n"
            print(f"[ERROR] {name} execution failed permanently.")
            log_message(error_msg)
        else:
            print(f"[SUCCESS] {name} completed successfully.")
            log_message(f"[SUCCESS] Model {name} completed successfully.\n")

    except Exception as e:
        # Absolute Isolation: Catch ANY exception to prevent blocking main program
        error_msg = f"[CRITICAL EXCEPTION] Model {name} crashed with exception: {str(e)}\n"
        print(f"[CRITICAL] {name} crashed: {e}")
        log_message(error_msg)

def explore_outputs():
    """
    Traverse data/ directory, find output files, read headers, and report.
    Strictly follows Stage 1 constraints: No pandas, no metrics.
    """
    print("[INFO] Starting output structure exploration...")
    log_message("\n=== Stage 1 Exploration Report ===\n")
    log_message(f"Exploration Time: {get_timestamp()}\n")
    
    # Ensure data dir exists before globbing
    if not os.path.exists(DATA_DIR):
        log_message("[ERROR] Data directory does not exist.")
        return

    # Find all directories ending with _out
    out_dirs = glob.glob(os.path.join(DATA_DIR, "*_out"))
    
    found_files = []
    for d in out_dirs:
        if os.path.isdir(d):
            # Walk through the directory
            for root, dirs, files in os.walk(d):
                for file in files:
                    found_files.append(os.path.join(root, file))
    
    # Fallback: Check any file in data/ just in case (excluding log itself)
    if not found_files:
        for f in os.listdir(DATA_DIR):
            full_path = os.path.join(DATA_DIR, f)
            if os.path.isfile(full_path) and not f.endswith('.txt'): 
                found_files.append(full_path)

    if not found_files:
        log_message("[INFO] No output files found in *_out directories or data/ root.")
    else:
        log_message(f"[INFO] Found {len(found_files)} output files.\n")
        for file_path in found_files:
            log_message(f"File Path: {file_path}")
            log_message(f"File Size: {os.path.getsize(file_path)} bytes")
            head_content = read_file_head(file_path, lines=10)
            log_message("Header Content (First 10 lines):")
            for line in head_content:
                log_message(f"  {line}")
            log_message("-" * 50)

    print(f"[INFO] Exploration complete. Report saved to {OBSERVATION_LOG}")

# ==============================================================================
# Main Execution Flow
# ==============================================================================

def main():
    print("=" * 60)
    print("Stage 1: Model Execution & Exploration Pipeline")
    print("=" * 60)
    
    # 0. Initialization
    ensure_data_dir()
    
    # Clear previous observation log for fresh start
    if os.path.exists(OBSERVATION_LOG):
        os.remove(OBSERVATION_LOG)
        
    log_message(f"Pipeline Start Time: {get_timestamp()}")
    
    # 1. Execute Models (Isolated & Self-Healing)
    for model in MODELS:
        run_model_with_healing(model)
        # Small delay to ensure file system sync
        subprocess.run("sleep 1", shell=True)
        
    # 2. Explore Outputs (Reconnaissance)
    explore_outputs()
    
    log_message(f"Pipeline End Time: {get_timestamp()}")
    print("=" * 60)
    print("Pipeline Finished. Check data/stage1_observation.txt for details.")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

## 2. Slurm 提交脚本 (`submit_stage1.slurm`)

本脚本用于在 HPC 集群上提交上述 Python 任务。
- ✅ 配置了合理的资源请求（CPU/内存/时间）。
- ✅ 包含了错误退出码检查。
- ✅ 指向正确的 Python 脚本路径。

```bash
#!/bin/bash
#SBATCH --job-name=stage1_exploration
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# Load necessary modules if required by your cluster
# module load python/3.8

# Ensure log directory exists
mkdir -p logs
mkdir -p data

# Change to the directory where the script is submitted
cd $SLURM_SUBMIT_DIR

echo "Starting Stage 1 Exploration at $(date)"

# Run the Python script
# Ensure the python environment has access to necessary system paths
/usr/bin/python3 stage1_exploration.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Stage 1 Pipeline completed successfully at $(date)"
else
    echo "Stage 1 Pipeline failed with exit code $? at $(date)"
    exit 1
fi
```

## 3. 执行说明

1.  **文件准备**：
    *   将 Python 脚本保存为 `stage1_exploration.py`。
    *   将 Slurm 脚本保存为 `submit_stage1.slurm`。
    *   确保 `data/combined_test.fasta` 存在于当前目录或脚本指定的相对路径下。
2.  **提交任务**：
    ```bash
    sbatch submit_stage1.slurm
    ```
3.  **结果查看**：
    *   任务完成后，查看 `data/stage1_observation.txt` 获取模型输出文件的结构勘探报告。
    *   查看 `logs/stage1_<job_id>.out` 获取标准运行日志。

