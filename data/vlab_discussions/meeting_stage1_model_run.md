## User

现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。

会议任务如下：
当前需要评测的计算生物学模型清单：Macrel, amPEPpy, AMPlify。

请严格遵守以下编程规范，完全由你自主编写高质量的 Python 自动化评估脚本：

1. 【模型执行参数矩阵（并行 Job Array 模式）】：
你的 Python 脚本将被 Slurm Array 并行调用。请在脚本开头通过 `task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))` 获取当前任务索引。
根据 task_id 仅选取并执行以下列表中的【一个】对应模型（如果 task_id 超出范围请安全打印并退出）：
- 【模型名称】: Macrel
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"
  【输出目录】: data/Macrel_out

- 【模型名称】: amPEPpy
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"
  【输出目录】: data/amPEPpy_out

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

## PI 审阅任务并提出明确要求

### 硬性要求确认：
1. **Slurm Array 模式执行**：通过 `SLURM_ARRAY_TASK_ID` 获取任务索引，仅执行对应模型
2. **三个模型矩阵**：Macrel(task_id=0), amPEPpy(task_id=1), AMPlify(task_id=2)
3. **智能自愈机制**：清理→执行→失败检查→自愈重试→日志记录
4. **Stage 1 探勘目标**：发现各模型实际输出文件结构和格式
5. **隔离写入纪律**：`log_file = f"data/stage1_obs_{task_id}.txt"`，绝对禁止共享文件

### 关键风险点强调：
- Macrel 会因目录已存在而失败，必须先清理再执行
- 部分模型需要自愈重试机制处理目录创建问题
- .gz 文件读取需要特殊处理
- 错误日志必须独立记录

## 代码工程师最终完整代码

```python
import os
import subprocess
import shutil
import glob
import gzip

def read_file_head(filepath, lines=10):
    """读取文件头部内容，支持gzip压缩文件"""
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt') as f:
                content = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content.append(line.rstrip())
                return '\n'.join(content)
        else:
            with open(filepath, 'r') as f:
                content = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content.append(line.rstrip())
                return '\n'.join(content)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def explore_output_directory(output_dir, model_name):
    """勘探输出目录中的所有文件并生成报告"""
    report_lines = [f"=== Model: {model_name} Output Exploration Report ==="]
    
    if not os.path.exists(output_dir):
        report_lines.append(f"Output directory {output_dir} does not exist!")
        return '\n'.join(report_lines)
    
    # 查找所有文件（包括子目录）
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    if not all_files:
        report_lines.append(f"No files found in {output_dir}")
        return '\n'.join(report_lines)
    
    report_lines.append(f"Found {len(all_files)} files in {output_dir}:")
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, output_dir)
        abs_path = os.path.abspath(filepath)
        file_size = os.path.getsize(filepath)
        
        report_lines.append(f"\n--- File: {rel_path} ---")
        report_lines.append(f"Absolute path: {abs_path}")
        report_lines.append(f"Size: {file_size} bytes")
        
        # 读取文件头部内容
        head_content = read_file_head(filepath, 10)
        report_lines.append("Head content:")
        report_lines.append(head_content)
        report_lines.append("-" * 50)
    
    return '\n'.join(report_lines)

def main():
    # 获取任务ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    # 定义模型配置
    models = [
        {
            "name": "Macrel",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
            "output_dir": "data/Macrel_out"
        },
        {
            "name": "amPEPpy", 
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"',
            "output_dir": "data/amPEPpy_out"
        },
        {
            "name": "AMPlify",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out"',
            "output_dir": "data/AMPlify_out"
        }
    ]
    
    # 检查task_id是否有效
    if task_id >= len(models):
        print(f"Task ID {task_id} exceeds available models (0-{len(models)-1}). Exiting.")
        return
    
    model = models[task_id]
    model_name = model["name"]
    cmd = model["cmd"]
    output_dir = model["output_dir"]
    
    print(f"Starting task {task_id}: Running {model_name}")
    
    # 执行模型并实现自愈机制
    try:
        # 第一步：清理历史数据（防止冲突）
        print(f"Cleaning old output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
        
        # 执行模型
        print(f"Executing {model_name}...")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 检查是否执行失败
        if res.returncode != 0:
            print(f"{model_name} execution failed. Return code: {res.returncode}")
            print(f"Stderr: {res.stderr}")
            
            # 检查错误类型并尝试自愈
            error_msg = res.stderr + res.stdout
            if "No such file" in error_msg or "NotFoundError" in error_msg or "not found" in error_msg:
                print("Detected missing directory error. Creating output directory and retrying...")
                os.makedirs(output_dir, exist_ok=True)
                
                # 重新执行
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if res.returncode != 0:
                    print(f"Retry failed for {model_name}. Final stderr: {res.stderr}")
                    
                    # 再次检查错误类型
                    final_error_msg = res.stderr + res.stdout
                    if "already exists" in final_error_msg or "exists" in final_error_msg:
                        print("Detected directory exists conflict. Removing and retrying...")
                        shutil.rmtree(output_dir, ignore_errors=True)
                        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 记录执行结果
        log_file = f"data/stage1_obs_{task_id}.txt"
        os.makedirs("data", exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"Model Execution Report for Task {task_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Return Code: {res.returncode}\n")
            f.write(f"Stdout:\n{res.stdout}\n")
            f.write(f"Stderr:\n{res.stderr}\n")
            f.write("="*60 + "\n\n")
            
            # 勘探输出目录
            exploration_report = explore_output_directory(output_dir, model_name)
            f.write(exploration_report)
        
        print(f"Exploration report saved to {log_file}")
        
    except Exception as e:
        print(f"Exception occurred during execution of {model_name}: {str(e)}")
        log_file = f"data/stage1_obs_{task_id}.txt"
        os.makedirs("data", exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"Model Execution Error Report for Task {task_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Exception: {str(e)}\n")
            f.write("="*60 + "\n\n")
            
            # 即使出错也要尝试勘探目录
            exploration_report = explore_output_directory(output_dir, model_name)
            f.write(exploration_report)

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
import subprocess
import shutil
import glob
import gzip

def read_file_head(filepath, lines=10):
    """读取文件头部内容，支持gzip压缩文件"""
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt') as f:
                content = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content.append(line.rstrip())
                return '\n'.join(content)
        else:
            with open(filepath, 'r') as f:
                content = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content.append(line.rstrip())
                return '\n'.join(content)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def explore_output_directory(output_dir, model_name):
    """勘探输出目录中的所有文件并生成报告"""
    report_lines = [f"=== Model: {model_name} Output Exploration Report ==="]
    
    if not os.path.exists(output_dir):
        report_lines.append(f"Output directory {output_dir} does not exist!")
        return '\n'.join(report_lines)
    
    # 查找所有文件（包括子目录）
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    if not all_files:
        report_lines.append(f"No files found in {output_dir}")
        return '\n'.join(report_lines)
    
    report_lines.append(f"Found {len(all_files)} files in {output_dir}:")
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, output_dir)
        abs_path = os.path.abspath(filepath)
        file_size = os.path.getsize(filepath)
        
        report_lines.append(f"\n--- File: {rel_path} ---")
        report_lines.append(f"Absolute path: {abs_path}")
        report_lines.append(f"Size: {file_size} bytes")
        
        # 读取文件头部内容
        head_content = read_file_head(filepath, 10)
        report_lines.append("Head content:")
        report_lines.append(head_content)
        report_lines.append("-" * 50)
    
    return '\n'.join(report_lines)

def main():
    # 获取任务ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    # 定义模型配置
    models = [
        {
            "name": "Macrel",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
            "output_dir": "data/Macrel_out"
        },
        {
            "name": "amPEPpy", 
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"',
            "output_dir": "data/amPEPpy_out"
        },
        {
            "name": "AMPlify",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out"',
            "output_dir": "data/AMPlify_out"
        }
    ]
    
    # 检查task_id是否有效
    if task_id >= len(models):
        print(f"Task ID {task_id} exceeds available models (0-{len(models)-1}). Exiting.")
        return
    
    model = models[task_id]
    model_name = model["name"]
    cmd = model["cmd"]
    output_dir = model["output_dir"]
    
    print(f"Starting task {task_id}: Running {model_name}")
    
    # 执行模型并实现自愈机制
    try:
        # 第一步：清理历史数据（防止冲突）
        print(f"Cleaning old output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
        
        # 执行模型
        print(f"Executing {model_name}...")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 检查是否执行失败
        if res.returncode != 0:
            print(f"{model_name} execution failed. Return code: {res.returncode}")
            print(f"Stderr: {res.stderr}")
            
            # 检查错误类型并尝试自愈
            error_msg = res.stderr + res.stdout
            if "No such file" in error_msg or "NotFoundError" in error_msg or "not found" in error_msg:
                print("Detected missing directory error. Creating output directory and retrying...")
                os.makedirs(output_dir, exist_ok=True)
                
                # 重新执行
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if res.returncode != 0:
                    print(f"Retry failed for {model_name}. Final stderr: {res.stderr}")
                    
                    # 再次检查错误类型
                    final_error_msg = res.stderr + res.stdout
                    if "already exists" in final_error_msg or "exists" in final_error_msg:
                        print("Detected directory exists conflict. Removing and retrying...")
                        shutil.rmtree(output_dir, ignore_errors=True)
                        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 记录执行结果
        log_file = f"data/stage1_obs_{task_id}.txt"
        os.makedirs("data", exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"Model Execution Report for Task {task_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Return Code: {res.returncode}\n")
            f.write(f"Stdout:\n{res.stdout}\n")
            f.write(f"Stderr:\n{res.stderr}\n")
            f.write("="*60 + "\n\n")
            
            # 勘探输出目录
            exploration_report = explore_output_directory(output_dir, model_name)
            f.write(exploration_report)
        
        print(f"Exploration report saved to {log_file}")
        
    except Exception as e:
        print(f"Exception occurred during execution of {model_name}: {str(e)}")
        log_file = f"data/stage1_obs_{task_id}.txt"
        os.makedirs("data", exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"Model Execution Error Report for Task {task_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Exception: {str(e)}\n")
            f.write("="*60 + "\n\n")
            
            # 即使出错也要尝试勘探目录
            exploration_report = explore_output_directory(output_dir, model_name)
            f.write(exploration_report)

if __name__ == '__main__':
    main()
```

```bash
#!/bin/bash
#SBATCH -J stage1_explore
#SBATCH --array=0-2%3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o stage1_explore_%A_%a.out
#SBATCH -e stage1_explore_%A_%a.err

cd /share/home/zhangss/vlab_workspace
source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
conda activate eval_env
python stage1_exploration.py
echo "finish"
```

