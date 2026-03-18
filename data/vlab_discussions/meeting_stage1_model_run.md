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

### PI 审阅与任务重申

各位好，会议正式开始。我是本次项目的 PI。

在代码工程师提交最终脚本之前，我必须再次强调本次**探索勘探阶段（Stage 1）**的核心纪律。这不是普通的脚本编写，这是为了后续大规模评测建立稳定的基础设施。任何细微的容错疏忽都会导致后续流水线崩溃。

请工程师注意以下**不可妥协的硬约束**：

1.  **执行隔离性（Isolation）**：
    *   每个模型的执行必须被独立的 `try...except` 块包裹。
    *   **严禁**因为一个模型失败而导致整个脚本退出。捕获异常后，必须记录错误并继续下一个模型。
    *   **严禁**使用 `raise` 阻断主程序。

2.  **智能自愈机制（Self-Healing）**：
    *   这是本次任务的核心。生物信息学工具对文件系统的状态极其敏感。
    *   你必须实现**双重检查**：执行前预建目录，执行后若报错，必须解析 `stdout` 和 `stderr` 的具体内容。
    *   针对 "No such file" 类错误，必须尝试 `makedirs` 后重试。
    *   针对 "already exists" 类错误，必须尝试 `rmtree` 后重试。
    *   只有重试依然失败，才写入最终错误日志。

3.  **勘探报告规范（Observation）**：
    *   脚本运行结束后，必须自动遍历 `data/` 下的输出。
    *   必须能识别并读取 `.gz` 压缩文件（使用 `gzip` 模块）。
    *   报告内容必须包含：文件路径 + 文件头部内容（5-10 行）。
    *   报告统一保存为 `data/stage1_observation.txt`。

4.  **代码禁令**：
    *   **绝对禁止**引入 `pandas`。
    *   **绝对禁止**在此阶段进行任何数据清洗或指标计算。
    *   代码必须是单文件，包含 `if __name__ == '__main__':`。

请工程师严格按照上述规范，编写完整的 Python 自动化评估脚本。现在开始 coding。

---

### 代码工程师最终完整代码

```python
import os
import subprocess
import shutil
import glob
import gzip

# 配置常量
DATA_DIR = "data"
OBSERVATION_LOG = os.path.join(DATA_DIR, "stage1_observation.txt")

# 模型执行配置清单
MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out"',
        "output_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "output_dir": "data/AMP-Scanner-v2_out"
    }
]

def ensure_data_dir():
    """确保数据目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message, mode='a'):
    """将消息写入观察日志"""
    ensure_data_dir()
    with open(OBSERVATION_LOG, mode, encoding='utf-8') as f:
        f.write(message + "\n")

def read_file_head(file_path, lines=10):
    """读取文件头部内容，支持 .gz 压缩文件"""
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
        return [f"Error reading file: {str(e)}"]
    return content_lines

def execute_model(model_config):
    """
    执行单个模型，包含智能自愈机制
    """
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["output_dir"]
    
    print(f"[INFO] Starting execution of model: {name}")
    log_message(f"--- Model Execution: {name} ---")
    
    try:
        # 第一步：清理 (Clean)
        # 即使目录不存在，ignore_errors=True 也不会报错
        shutil.rmtree(out_dir, ignore_errors=True)
        
        # 第二步：执行前兜底 (Pre-mkdir)
        # 根据 Fatal Error Prevention 规范，执行前必须建好房子
        os.makedirs(out_dir, exist_ok=True)
        
        # 第三步：盲测 (Blind Test)
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 第四步：动态自愈 (Self-Healing)
        if res.returncode != 0:
            # 联合检查 stdout 和 stderr
            combined_output = res.stdout + res.stderr
            print(f"[WARN] {name} initial execution failed. Analyzing error...")
            log_message(f"[ERROR] Initial execution failed for {name}. Returncode: {res.returncode}")
            log_message(f"[DEBUG] Stdout/Stderr snippet: {combined_output[:500]}")
            
            retry_success = False
            
            # 情况 A: 缺少目录 (No such file or directory / Failed to save)
            if "No such file or directory" in combined_output or "Failed to save" in combined_output:
                print(f"[HEALING] {name} detected missing directory error. Attempting to recreate directory...")
                log_message(f"[HEALING] Detected 'No such file' error. Recreating {out_dir}...")
                try:
                    shutil.rmtree(out_dir, ignore_errors=True) # 先清理确保干净
                    os.makedirs(out_dir, exist_ok=True)
                    # 重试
                    res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if res_retry.returncode == 0:
                        retry_success = True
                        print(f"[SUCCESS] {name} healed successfully after mkdir retry.")
                        log_message(f"[SUCCESS] {name} healed successfully after mkdir retry.")
                except Exception as heal_e:
                    log_message(f"[ERROR] Healing process failed for {name}: {str(heal_e)}")
            
            # 情况 B: 目录已存在冲突 (already exists / exists)
            elif "already exists" in combined_output or "exists" in combined_output:
                print(f"[HEALING] {name} detected existing directory conflict. Attempting to remove and retry...")
                log_message(f"[HEALING] Detected 'exists' error. Removing {out_dir} and retrying...")
                try:
                    shutil.rmtree(out_dir, ignore_errors=True)
                    # 注意：这里不再预先 makedirs，让工具自己创建，或者重试前再建
                    # 根据规范：删掉然后再次重试执行 subprocess.run
                    # 但为了安全，我们让工具自己处理目录，或者再次确保父目录存在
                    os.makedirs(os.path.dirname(out_dir), exist_ok=True) 
                    
                    res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if res_retry.returncode == 0:
                        retry_success = True
                        print(f"[SUCCESS] {name} healed successfully after rmtree retry.")
                        log_message(f"[SUCCESS] {name} healed successfully after rmtree retry.")
                except Exception as heal_e:
                    log_message(f"[ERROR] Healing process failed for {name}: {str(heal_e)}")
            
            # 如果重试依然失败，记录终极日志
            if not retry_success:
                error_msg = f"[FATAL] Model {name} failed after healing attempts.\nStdout: {res.stdout}\nStderr: {res.stderr}"
                print(f"[FATAL] {name} execution failed permanently.")
                log_message(error_msg)
                # 不 raise，继续执行
                return False
            else:
                return True
        else:
            print(f"[SUCCESS] {name} executed successfully.")
            log_message(f"[SUCCESS] {name} executed successfully.")
            return True
            
    except Exception as e:
        # 绝对隔离，捕获所有未预料的异常
        error_msg = f"[CRITICAL] Unexpected exception during {name} execution: {str(e)}"
        print(error_msg)
        log_message(error_msg)
        return False

def explore_outputs():
    """
    勘探输出文件结构并生成报告
    """
    print("[INFO] Starting output exploration...")
    log_message("\n--- Stage 1 Exploration Report ---\n")
    
    ensure_data_dir()
    
    # 查找所有带有 _out 后缀的目录
    # 使用 glob 查找 data/ 下的目录
    out_dirs = glob.glob(os.path.join(DATA_DIR, "*_out"))
    # 同时也查找 data/ 下可能直接生成的文件 (以防万一)
    all_files = glob.glob(os.path.join(DATA_DIR, "*"))
    
    found_files = []
    
    # 收集目录内的文件
    for d in out_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for file in files:
                    found_files.append(os.path.join(root, file))
    
    # 收集 data 目录下直接的文件 (排除目录本身)
    for f in all_files:
        if os.path.isfile(f) and f not in found_files:
            # 排除日志文件本身以免递归
            if f != OBSERVATION_LOG:
                found_files.append(f)
    
    if not found_files:
        log_message("No output files found in data/ directory.")
        print("[WARN] No output files found.")
        return

    log_message(f"Total files found: {len(found_files)}\n")
    
    for file_path in found_files:
        log_message(f"File Path: {file_path}")
        try:
            head_content = read_file_head(file_path, lines=10)
            log_message("Head Content (first 10 lines):")
            for line in head_content:
                log_message(f"  {line}")
            log_message("-" * 50)
        except Exception as e:
            log_message(f"Error reading content: {str(e)}")
            log_message("-" * 50)

def main():
    """
    主程序入口
    """
    print("=" * 60)
    print("Starting AMP Model Evaluation Stage 1 (Exploration)")
    print("=" * 60)
    
    # 初始化日志
    ensure_data_dir()
    # 清空或创建日志文件头 (使用 'w' 模式开始新的一次运行记录，或者用 'a' 追加)
    # 根据规范 "追加写入"，但为了每次运行清晰，这里选择覆盖写入头部，后续错误追加
    # 为了严格遵守 "追加写入错误流"，我们初始化时写一个 header
    with open(OBSERVATION_LOG, 'w', encoding='utf-8') as f:
        f.write("=== Stage 1 Observation Log ===\n")
        f.write(f"Start Time: {subprocess.run('date', shell=True, capture_output=True, text=True).stdout.strip()}\n\n")
    
    # 1. 执行模型
    for model in MODELS:
        execute_model(model)
        print("-" * 30)
    
    # 2. 勘探输出
    explore_outputs()
    
    print("=" * 60)
    print("Stage 1 Complete. Check data/stage1_observation.txt for details.")
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

### 1. Python 自动化评估脚本 (`stage1_exploration.py`)

```python
import os
import subprocess
import shutil
import glob
import gzip
import sys

# ================= 配置区域 =================
DATA_DIR = "data"
OBSERVATION_LOG = os.path.join(DATA_DIR, "stage1_observation.txt")

# 模型执行配置清单
MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out"',
        "output_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "output_dir": "data/AMP-Scanner-v2_out"
    }
]
# ===========================================

def ensure_data_dir():
    """确保数据目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message, mode='a'):
    """将消息写入观察日志"""
    ensure_data_dir()
    try:
        with open(OBSERVATION_LOG, mode, encoding='utf-8') as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[CRITICAL] Failed to write log: {e}")

def read_file_head(file_path, lines=10):
    """
    读取文件头部内容，支持 .gz 压缩文件
    严格遵守 Stage 1 约束：不使用 pandas，直接读取文本
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
        return [f"Error reading file: {str(e)}"]
    return content_lines

def execute_model(model_config):
    """
    执行单个模型，包含智能自愈机制 (Self-Healing)
    严格遵守防御性编程规范：绝对隔离，永不连坐
    """
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["output_dir"]
    
    print(f"[INFO] Starting execution of model: {name}")
    log_message(f"--- Model Execution: {name} ---")
    
    try:
        # 第一步：清理 (Clean)
        # 即使目录不存在，ignore_errors=True 也不会报错
        shutil.rmtree(out_dir, ignore_errors=True)
        
        # 第二步：执行前兜底 (Pre-mkdir)
        # 根据规范，执行前必须建好房子
        os.makedirs(out_dir, exist_ok=True)
        
        # 第三步：盲测 (Blind Test)
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 第四步：动态自愈 (Self-Healing)
        if res.returncode != 0:
            # 联合检查 stdout 和 stderr
            combined_output = res.stdout + res.stderr
            print(f"[WARN] {name} initial execution failed. Analyzing error...")
            log_message(f"[ERROR] Initial execution failed for {name}. Returncode: {res.returncode}")
            log_message(f"[DEBUG] Stdout/Stderr snippet: {combined_output[:500]}")
            
            retry_success = False
            
            # 情况 A: 缺少目录 (No such file or directory / Failed to save)
            if "No such file or directory" in combined_output or "Failed to save" in combined_output:
                print(f"[HEALING] {name} detected missing directory error. Attempting to recreate directory...")
                log_message(f"[HEALING] Detected 'No such file' error. Recreating {out_dir}...")
                try:
                    shutil.rmtree(out_dir, ignore_errors=True) # 先清理确保干净
                    os.makedirs(out_dir, exist_ok=True)
                    # 重试
                    res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if res_retry.returncode == 0:
                        retry_success = True
                        print(f"[SUCCESS] {name} healed successfully after mkdir retry.")
                        log_message(f"[SUCCESS] {name} healed successfully after mkdir retry.")
                except Exception as heal_e:
                    log_message(f"[ERROR] Healing process failed for {name}: {str(heal_e)}")
            
            # 情况 B: 目录已存在冲突 (already exists / exists)
            elif "already exists" in combined_output or "exists" in combined_output:
                print(f"[HEALING] {name} detected existing directory conflict. Attempting to remove and retry...")
                log_message(f"[HEALING] Detected 'exists' error. Removing {out_dir} and retrying...")
                try:
                    shutil.rmtree(out_dir, ignore_errors=True)
                    # 注意：这里不再预先 makedirs，让工具自己创建，或者重试前再建
                    # 根据规范：删掉然后再次重试执行 subprocess.run
                    os.makedirs(os.path.dirname(out_dir), exist_ok=True) 
                    
                    res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if res_retry.returncode == 0:
                        retry_success = True
                        print(f"[SUCCESS] {name} healed successfully after rmtree retry.")
                        log_message(f"[SUCCESS] {name} healed successfully after rmtree retry.")
                except Exception as heal_e:
                    log_message(f"[ERROR] Healing process failed for {name}: {str(heal_e)}")
            
            # 如果重试依然失败，记录终极日志
            if not retry_success:
                error_msg = f"[FATAL] Model {name} failed after healing attempts.\nStdout: {res.stdout}\nStderr: {res.stderr}"
                print(f"[FATAL] {name} execution failed permanently.")
                log_message(error_msg)
                # 不 raise，继续执行 (绝对隔离)
                return False
            else:
                return True
        else:
            print(f"[SUCCESS] {name} executed successfully.")
            log_message(f"[SUCCESS] {name} executed successfully.")
            return True
            
    except Exception as e:
        # 绝对隔离，捕获所有未预料的异常
        error_msg = f"[CRITICAL] Unexpected exception during {name} execution: {str(e)}"
        print(error_msg)
        log_message(error_msg)
        return False

def explore_outputs():
    """
    勘探输出文件结构并生成报告
    严格遵守 Stage 1 约束：不许 merge，不许计算指标，只记录文件结构和头部内容
    """
    print("[INFO] Starting output exploration...")
    log_message("\n--- Stage 1 Exploration Report ---\n")
    
    ensure_data_dir()
    
    # 查找所有带有 _out 后缀的目录
    out_dirs = glob.glob(os.path.join(DATA_DIR, "*_out"))
    # 同时也查找 data/ 下可能直接生成的文件
    all_files = glob.glob(os.path.join(DATA_DIR, "*"))
    
    found_files = []
    
    # 收集目录内的文件
    for d in out_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for file in files:
                    found_files.append(os.path.join(root, file))
    
    # 收集 data 目录下直接的文件 (排除目录本身)
    for f in all_files:
        if os.path.isfile(f) and f not in found_files:
            # 排除日志文件本身以免递归
            if f != OBSERVATION_LOG:
                found_files.append(f)
    
    if not found_files:
        log_message("No output files found in data/ directory.")
        print("[WARN] No output files found.")
        return

    log_message(f"Total files found: {len(found_files)}\n")
    
    for file_path in found_files:
        log_message(f"File Path: {file_path}")
        try:
            head_content = read_file_head(file_path, lines=10)
            log_message("Head Content (first 10 lines):")
            for line in head_content:
                log_message(f"  {line}")
            log_message("-" * 50)
        except Exception as e:
            log_message(f"Error reading content: {str(e)}")
            log_message("-" * 50)

def main():
    """
    主程序入口
    严格遵守代码结构规范：单一脚本，完整 import，def main
    """
    print("=" * 60)
    print("Starting AMP Model Evaluation Stage 1 (Exploration)")
    print("=" * 60)
    
    # 初始化日志
    ensure_data_dir()
    # 初始化日志文件
    with open(OBSERVATION_LOG, 'w', encoding='utf-8') as f:
        f.write("=== Stage 1 Observation Log ===\n")
        try:
            date_res = subprocess.run('date', shell=True, capture_output=True, text=True)
            f.write(f"Start Time: {date_res.stdout.strip()}\n\n")
        except:
            f.write("Start Time: Unknown\n\n")
    
    # 1. 执行模型 (严格隔离)
    for model in MODELS:
        execute_model(model)
        print("-" * 30)
    
    # 2. 勘探输出 (禁止 pandas merge/metrics)
    explore_outputs()
    
    print("=" * 60)
    print("Stage 1 Complete. Check data/stage1_observation.txt for details.")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

### 2. Slurm 提交脚本 (`submit_stage1.slurm`)

```bash
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
```

