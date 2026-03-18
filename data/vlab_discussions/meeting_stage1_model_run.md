## User

现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。

会议任务如下：
当前需要评测的计算生物学模型清单：Macrel, AMP-Scanner-v2。

请严格遵守以下编程规范，完全由你自主编写高质量的 Python 自动化评估脚本：

1. 【模型执行参数矩阵】：
你需要通过代码依次执行以下模型：
- 【模型名称】: Macrel
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"
  【输出目录】: data/Macrel_out

- 【模型名称】: AMP-Scanner-v2
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"
  【输出目录】: data/AMP-Scanner-v2_out


🚨 注意：如果【预设运行命令】中执行的 Python 脚本已经是以 `/` 开头的绝对路径，你绝对不允许擅自去修改或拼接它！直接原样使用！只有当它使用的是相对路径，并且我为你提供了【真实源码目录树】时，你才需要结合目录树，以 `/share/home/zhangss/[模型名称]` 作为基准路径去修正它！

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

# 🧬 计算生物学模型自动化评估会议记录 (Stage 1: 探索勘探)

## 1. PI 任务审阅与要求重申 (Principal Investigator Review)

各位工程师，大家好。针对本次 **AMP 模型自动化评估 (Stage 1: 探索勘探)** 任务，我代表项目组审阅了技术需求。生物信息学流程的稳定性至关重要，尤其是涉及外部依赖复杂的模型（如 Macrel, AMP-Scanner）。为了确保本次勘探任务的成功执行及后续流程的可靠性，我再次重申以下 **硬性约束 (Hard Requirements)**，请在代码实现中严格落地：

1.  **命令完整性保护 (Command Integrity)**：
    *   提供的【预设运行命令】中包含绝对路径（如 `/share/home/zhangss/...`）。**绝对禁止**修改、拼接或相对化这些路径。必须原样嵌入脚本。
    *   任何对命令字符串的篡改都可能导致环境加载失败，这是不可接受的低级错误。

2.  **智能自愈机制 (Self-Healing Protocol)**：
    *   生物软件常因目录状态（存在/不存在）而崩溃。你必须实现 **双重防御**：
        *   **执行前**：强制清理旧目录 (`rmtree`) 并预建目录 (`makedirs`)。
        *   **失败后**：根据 `stdout/stderr` 关键词动态调整策略（报 "exists" 则删目录重试；报 "No such file" 则建目录重试）。
    *   **隔离性**：每个模型必须包裹在独立的 `try...except` 块中。Macrel 崩溃绝不能影响 AMP-Scanner 的运行。

3.  **勘探报告规范 (Exploration Report)**：
    *   本次任务 **仅限勘探**。禁止使用 `pandas`，禁止数据清洗，禁止计算指标。
    *   所有模型运行结束后，必须遍历 `data/` 下的输出文件，读取文件头（支持 `.gz` 解压），并将 **文件路径 + 内容摘要** 写入 `data/stage1_observation.txt`。
    *   模型运行的错误日志也必须追加到同一文件中，以便追溯。

4.  **工程纪律 (Engineering Discipline)**：
    *   脚本必须是单一、可独立运行的 Python 文件。
    *   必须包含 `if __name__ == '__main__':` 入口。
    *   确保 `data/` 目录在执行任何写入操作前已存在。

请代码工程师根据以上要求，输出最终的可执行代码。

---

## 2. 代码工程师最终实现 (Engineer Implementation)

```python
import os
import subprocess
import shutil
import glob
import gzip

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================

# 确保基础数据目录存在
os.makedirs('data', exist_ok=True)

# 观察报告路径
OBSERVATION_FILE = 'data/stage1_observation.txt'

# 模型执行配置清单
# 注意：命令中的绝对路径严禁修改
MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
        "output_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "output_dir": "data/AMP-Scanner-v2_out"
    }
]

# ==============================================================================
# 工具函数 (Utility Functions)
# ==============================================================================

def log_observation(message):
    """
    将消息追加写入到观察报告文件中。
    """
    try:
        with open(OBSERVATION_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[CRITICAL] Failed to write to observation log: {e}")

def read_file_header(file_path, lines=10):
    """
    读取文件头部内容，自动处理 .gz 压缩文件。
    """
    content_summary = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_summary.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_summary.append(line.strip())
    except Exception as e:
        content_summary.append(f"[Error reading file: {str(e)}]")
    
    return content_summary

def execute_model_with_healing(model_config):
    """
    执行单个模型，包含智能自愈机制和异常隔离。
    """
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["output_dir"]
    
    log_observation(f"\n{'='*60}")
    log_observation(f"[START] Model: {name}")
    log_observation(f"[CMD] {cmd}")
    log_observation(f"[OUT] {out_dir}")
    print(f"[INFO] Starting {name}...")

    try:
        # --- 第一步：清理 (Clean) ---
        # 无论之前状态如何，先清理脏数据
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        
        # --- 第二步：执行前兜底 (Pre-flooring) ---
        # 根据防御性编程规范，执行前先建好目录
        os.makedirs(out_dir, exist_ok=True)

        # --- 第三步：盲测 (Blind Test) ---
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # --- 第四步：动态自愈 (Self-Healing) ---
        if res.returncode != 0:
            error_output = res.stdout + res.stderr
            log_observation(f"[WARN] Initial execution failed (Return Code: {res.returncode})")
            log_observation(f"[DEBUG] Error Output Snippet: {error_output[:500]}")
            
            retry_success = False
            
            # 策略 A: 处理 "already exists" 冲突
            # 如果报错包含 exists，说明模型讨厌预存在的目录。我们需要删掉刚才建的空目录，然后重试（不再预建）
            if "already exists" in error_output or "exists" in error_output:
                log_observation("[HEALING] Detected 'exists' error. Removing directory and retrying without pre-mkdir...")
                shutil.rmtree(out_dir, ignore_errors=True)
                # 重试时不再执行 makedirs，让模型自己创建或处理
                res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res_retry.returncode == 0:
                    retry_success = True
                    res = res_retry
            
            # 策略 B: 处理 "No such file" 冲突
            # 如果报错包含 No such file，说明模型需要目录存在但我们可能清理得太干净或路径有误
            elif "No such file or directory" in error_output or "Failed to save" in error_output:
                log_observation("[HEALING] Detected 'No such file' error. Ensuring directory exists and retrying...")
                os.makedirs(out_dir, exist_ok=True)
                res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res_retry.returncode == 0:
                    retry_success = True
                    res = res_retry
            
            # 记录最终状态
            if not retry_success:
                log_observation(f"[FAIL] Model {name} failed after healing attempts.")
                log_observation(f"[STDERR] {res.stderr}")
                log_observation(f"[STDOUT] {res.stdout}")
                print(f"[ERROR] {name} failed. Check observation log.")
            else:
                log_observation(f"[SUCCESS] Model {name} recovered and completed successfully.")
                print(f"[SUCCESS] {name} completed (after healing).")
        else:
            log_observation(f"[SUCCESS] Model {name} completed successfully.")
            print(f"[SUCCESS] {name} completed.")

    except Exception as e:
        # 捕获 Python 层面的异常（如权限错误、命令格式错误等）
        error_msg = f"[CRITICAL EXCEPTION] Model {name} crashed with exception: {str(e)}"
        log_observation(error_msg)
        print(error_msg)

def explore_outputs():
    """
    勘探 data/ 目录下所有 *_out 目录及文件，生成报告。
    """
    log_observation(f"\n{'='*60}")
    log_observation("[STAGE 1 EXPLORATION REPORT]")
    log_observation(f"{'='*60}\n")
    
    # 查找所有包含 _out 的目录
    # 使用 glob 查找 data/ 下的直接子目录，匹配 *_out
    out_dirs = glob.glob('data/*_out')
    
    # 同时也查找 data/ 下可能直接生成的文件 (以防万一)
    # 但主要关注 *_out 目录
    if not out_dirs:
        log_observation("[INFO] No directories matching 'data/*_out' found.")
        # 尝试查找 data 目录下所有文件
        all_files = glob.glob('data/*')
        for f in all_files:
            if os.path.isfile(f):
                log_observation(f"[FILE] {os.path.abspath(f)}")
                headers = read_file_header(f)
                log_observation("[HEADER CONTENT]:")
                for line in headers:
                    log_observation(f"  {line}")
                log_observation("")
        return

    for out_dir in sorted(out_dirs):
        log_observation(f"[DIR] {os.path.abspath(out_dir)}")
        
        # 遍历目录内所有文件
        found_files = False
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                found_files = True
                file_path = os.path.join(root, file)
                log_observation(f"  [FILE] {os.path.abspath(file_path)}")
                
                # 读取头部内容
                headers = read_file_header(file_path)
                log_observation("  [HEADER CONTENT]:")
                for line in headers:
                    # 替换制表符以便日志阅读，限制长度防止日志过大
                    safe_line = line.replace('\t', ' | ').replace('\n', '')
                    if len(safe_line) > 200:
                        safe_line = safe_line[:200] + "..."
                    log_observation(f"    {safe_line}")
                log_observation("")
        
        if not found_files:
            log_observation("  [WARNING] Directory is empty.")
        log_observation("-" * 40 + "\n")

    print("[INFO] Exploration report saved to", os.path.abspath(OBSERVATION_FILE))

# ==============================================================================
# 主程序入口 (Main Entry)
# ==============================================================================

def main():
    print("="*60)
    print("AMP Model Evaluation - Stage 1: Exploration")
    print("="*60)
    
    # 初始化日志文件（清空或创建）
    # 为了保留错误日志，我们使用追加模式，但为了报告清晰，可以在开头加个分隔符
    # 这里为了符合“追加写入错误流”的要求，我们不在这里清空文件，而是直接开始记录
    log_observation(f"\n### NEW SESSION STARTED: {os.path.basename(__file__)} ###\n")

    # 1. 执行模型评估
    for model in MODELS:
        execute_model_with_healing(model)
    
    # 2. 执行输出勘探
    explore_outputs()
    
    print("="*60)
    print("Stage 1 Complete. Check data/stage1_observation.txt")
    print("="*60)

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

# 🧬 计算生物学模型自动化评估 (Stage 1: 探索勘探)

## 1. 最终 Python 评估脚本 (`stage1_exploration.py`)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Model Exploration & Execution Script
Target: Macrel, AMP-Scanner-v2
Goal: Run models, explore output structure, save observation report.
Constraints: No pandas, no metrics, no merging. Strict self-healing.
"""

import os
import subprocess
import shutil
import glob
import gzip
import datetime

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================

# 确保基础数据目录存在
os.makedirs('data', exist_ok=True)

# 观察报告路径
OBSERVATION_FILE = 'data/stage1_observation.txt'

# 模型执行配置清单
# 🚨 注意：命令中的绝对路径严禁修改，必须原样使用
MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
        "output_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "output_dir": "data/AMP-Scanner-v2_out"
    }
]

# ==============================================================================
# 工具函数 (Utility Functions)
# ==============================================================================

def log_observation(message):
    """
    将消息追加写入到观察报告文件中。
    """
    try:
        with open(OBSERVATION_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[CRITICAL] Failed to write to observation log: {e}")

def read_file_header(file_path, lines=10):
    """
    读取文件头部内容，自动处理 .gz 压缩文件。
    """
    content_summary = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_summary.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_summary.append(line.strip())
    except Exception as e:
        content_summary.append(f"[Error reading file: {str(e)}]")
    
    return content_summary

def execute_model_with_healing(model_config):
    """
    执行单个模型，包含智能自愈机制和异常隔离。
    严格遵守防御性编程规范：绝对隔离，永不连坐。
    """
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["output_dir"]
    
    log_observation(f"\n{'='*60}")
    log_observation(f"[START] Model: {name}")
    log_observation(f"[TIME] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_observation(f"[CMD] {cmd}")
    log_observation(f"[OUT] {out_dir}")
    print(f"[INFO] Starting {name}...")

    try:
        # --- 第一步：清理 (Clean) ---
        # 无论之前状态如何，先清理脏数据
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        
        # --- 第二步：执行前兜底 (Pre-flooring) ---
        # 根据防御性编程规范，执行前先建好目录
        os.makedirs(out_dir, exist_ok=True)

        # --- 第三步：盲测 (Blind Test) ---
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # --- 第四步：动态自愈 (Self-Healing) ---
        if res.returncode != 0:
            error_output = res.stdout + res.stderr
            log_observation(f"[WARN] Initial execution failed (Return Code: {res.returncode})")
            log_observation(f"[DEBUG] Error Output Snippet: {error_output[:500]}")
            
            retry_success = False
            
            # 策略 A: 处理 "already exists" 冲突
            # 如果报错包含 exists，说明模型讨厌预存在的目录。我们需要删掉刚才建的空目录，然后重试
            if "already exists" in error_output or "exists" in error_output:
                log_observation("[HEALING] Detected 'exists' error. Removing directory and retrying...")
                shutil.rmtree(out_dir, ignore_errors=True)
                # 重试时不再执行 makedirs，让模型自己创建或处理
                res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res_retry.returncode == 0:
                    retry_success = True
                    res = res_retry
            
            # 策略 B: 处理 "No such file" 冲突
            # 如果报错包含 No such file，说明模型需要目录存在但我们可能清理得太干净或路径有误
            elif "No such file or directory" in error_output or "Failed to save" in error_output:
                log_observation("[HEALING] Detected 'No such file' error. Ensuring directory exists and retrying...")
                os.makedirs(out_dir, exist_ok=True)
                res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res_retry.returncode == 0:
                    retry_success = True
                    res = res_retry
            
            # 记录最终状态
            if not retry_success:
                log_observation(f"[FAIL] Model {name} failed after healing attempts.")
                log_observation(f"[STDERR] {res.stderr}")
                log_observation(f"[STDOUT] {res.stdout}")
                print(f"[ERROR] {name} failed. Check observation log.")
            else:
                log_observation(f"[SUCCESS] Model {name} recovered and completed successfully.")
                print(f"[SUCCESS] {name} completed (after healing).")
        else:
            log_observation(f"[SUCCESS] Model {name} completed successfully.")
            print(f"[SUCCESS] {name} completed.")

    except Exception as e:
        # 捕获 Python 层面的异常（如权限错误、命令格式错误等）
        error_msg = f"[CRITICAL EXCEPTION] Model {name} crashed with exception: {str(e)}"
        log_observation(error_msg)
        print(error_msg)

def explore_outputs():
    """
    勘探 data/ 目录下所有 *_out 目录及文件，生成报告。
    本次禁止使用 pandas，仅进行文件结构勘探。
    """
    log_observation(f"\n{'='*60}")
    log_observation("[STAGE 1 EXPLORATION REPORT]")
    log_observation(f"[TIME] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_observation(f"{'='*60}\n")
    
    # 查找所有包含 _out 的目录
    out_dirs = glob.glob('data/*_out')
    
    if not out_dirs:
        log_observation("[INFO] No directories matching 'data/*_out' found.")
        # 尝试查找 data 目录下所有文件以防万一
        all_files = glob.glob('data/*')
        for f in all_files:
            if os.path.isfile(f) and not f.endswith('.txt'):
                log_observation(f"[FILE] {os.path.abspath(f)}")
                headers = read_file_header(f)
                log_observation("[HEADER CONTENT]:")
                for line in headers:
                    log_observation(f"  {line}")
                log_observation("")
        return

    for out_dir in sorted(out_dirs):
        log_observation(f"[DIR] {os.path.abspath(out_dir)}")
        
        # 遍历目录内所有文件
        found_files = False
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                found_files = True
                file_path = os.path.join(root, file)
                log_observation(f"  [FILE] {os.path.abspath(file_path)}")
                
                # 读取头部内容
                headers = read_file_header(file_path)
                log_observation("  [HEADER CONTENT]:")
                for line in headers:
                    # 替换制表符以便日志阅读，限制长度防止日志过大
                    safe_line = line.replace('\t', ' | ').replace('\n', '')
                    if len(safe_line) > 200:
                        safe_line = safe_line[:200] + "..."
                    log_observation(f"    {safe_line}")
                log_observation("")
        
        if not found_files:
            log_observation("  [WARNING] Directory is empty.")
        log_observation("-" * 40 + "\n")

    print("[INFO] Exploration report saved to", os.path.abspath(OBSERVATION_FILE))

# ==============================================================================
# 主程序入口 (Main Entry)
# ==============================================================================

def main():
    print("="*60)
    print("AMP Model Evaluation - Stage 1: Exploration")
    print("="*60)
    
    # 初始化日志文件（添加 session 头）
    log_observation(f"\n### NEW SESSION STARTED: {os.path.basename(__file__)} ###\n")

    # 1. 执行模型评估
    for model in MODELS:
        execute_model_with_healing(model)
    
    # 2. 执行输出勘探
    explore_outputs()
    
    print("="*60)
    print("Stage 1 Complete. Check data/stage1_observation.txt")
    print("="*60)

if __name__ == '__main__':
    main()
```

## 2. Slurm 提交脚本 (`submit_stage1.slurm`)

```bash
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
```

