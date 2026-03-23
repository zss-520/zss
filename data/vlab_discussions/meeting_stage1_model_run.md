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
  【运行命令】: bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"
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

### PI 审查与要求确认

#### ✅ 任务目标明确：
我们处于**第一阶段探索性运行（Stage 1 Exploration）**，核心目标是：
- 并行启动五个生物信息学模型之一（由 SLURM_ARRAY_TASK_ID 控制）
- 每个模型运行完成后，自动探测其输出目录下的所有文件结构和内容
- 输出每个文件的路径 + 头部内容（支持 `.gz` 解压读取）
- 所有结果必须写入各自独立的日志文件中（`data/stage1_obs_{task_id}.txt`）

---

## 🔍 关键硬性约束再强调：

| 编号 | 要求类型 | 描述 |
|------|----------|------|
| ⚠️ 1 | **Job Array 分发控制** | 根据 `SLURM_ARRAY_TASK_ID` 只能选择一个模型执行；超出范围则安全退出 |
| ⚠️ 2 | **命令原样保留原则** | 若命令中使用了绝对路径，则不得修改；若为相对路径且提供目录树，需基于 `/share/home/zhangss/[模型名]` 补全 |
| ⚠️ 3 | **智能容错机制（Self-healing）** | 必须实现两轮重试策略：<br>① 清理旧目录<br>② 第一次运行<br>③ 出错判断是否因目录问题<br>④ 动态修复后第二次尝试 |
| ⚠️ 4 | **文件遍历与头部提取** | 使用 `glob`, `gzip` 等工具扫描并提取前几行文本 |
| ⚠️ 5 | **日志隔离写入规则** | 日志文件命名格式必须为 `data/stage1_obs_{task_id}.txt`，禁止合并或加锁 |

---

## 🧬 模型配置表（供参考）

```python
MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
        "out_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "out_dir": "data/AMP-Scanner-v2_out"
    },
    {
        "name": "amPEPpy",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"',
        "out_dir": "data/amPEPpy_out"
    },
    {
        "name": "AI4AMP",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv"',
        "out_dir": "data/AI4AMP_out"
    },
    {
        "name": "AMPlify",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out"',
        "out_dir": "data/AMPlify_out"
    }
]
```

---

## 🛠 工程师，请开始编码！

请按照上述规范编写完整的自动化脚本。以下是最终交付模板框架建议：

```python
#!/usr/bin/env python3

import os
import subprocess
import shutil
import glob
import gzip

MODELS = [ ... ]  # 上述模型列表

def run_model(model):
    ...

def inspect_output(out_dir, log_file):
    ...

def main():
    ...

if __name__ == "__main__":
    main()
```

请确保满足以下几点后再提交完整代码：
- ✅ 正确获取 `SLURM_ARRAY_TASK_ID`
- ✅ 模型命令原样执行
- ✅ 实现双轮容错逻辑（rmtree → run → mkdir → retry）
- ✅ 遍历输出目录下所有文件并读取头部内容
- ✅ 结果写入 `data/stage1_obs_{task_id}.txt`
- ✅ 不引入 pandas、不进行 merge、不计算指标

现在请你提供完整可运行的 Python 脚本代码。

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
2. 合并前 ID 必须进行 `split()[0].strip()` 强清洗，合并后必须用 `fillna(0.0)` 兜底。
3. 寻找概率列时必须校验 `dtype == float`，避开读取英文字符串标签导致的 DataFrame 灾难！
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
2. 必须提供 `run_eval.sh`，Bash 脚本必须写在单独的 ` ```bash ... ``` ` 块中。
   🚨 **针对并行运算的 Bash 模板强制要求**：为了实现一次最多调用 4 张显卡的并行阵列，你必须原样使用以下模板：
   ```bash
   #!/bin/bash
   #SBATCH -J amp_eval
   #SBATCH --array=0-[MAX_INDEX]%4   # 极度重要: 智能替换为 (模型总数量 - 1)
   #SBATCH -N 1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:3
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
#!/usr/bin/env python3

import os
import subprocess
import shutil
import glob
import gzip

MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
        "out_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "out_dir": "data/AMP-Scanner-v2_out"
    },
    {
        "name": "amPEPpy",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"',
        "out_dir": "data/amPEPpy_out"
    },
    {
        "name": "AI4AMP",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv"',
        "out_dir": "data/AI4AMP_out"
    },
    {
        "name": "AMPlify",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out"',
        "out_dir": "data/AMPlify_out"
    }
]

def run_model(model):
    out_dir = model["out_dir"]
    cmd = model["cmd"]

    # Step 1: Clean up previous outputs
    print(f"[INFO] Removing existing output directory: {out_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)

    # Step 2: First attempt to run the command
    print(f"[INFO] Running model '{model['name']}' with first attempt...")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if res.returncode == 0:
        print("[SUCCESS] Model executed successfully on first try.")
        return True

    stderr_output = res.stderr.lower()
    stdout_output = res.stdout.lower()
    combined_error = stderr_output + stdout_output

    print(f"[ERROR] First execution failed:\nSTDOUT={res.stdout}\nSTDERR={res.stderr}")

    # Step 3: Self-healing logic based on error messages
    if any(keyword in combined_error for keyword in ["no such file", "failed to save", "notfounderror"]):
        print("[HEALING] Detected missing directory issue. Creating output directory...")
        os.makedirs(out_dir, exist_ok=True)
        print("[RETRY] Retrying after creating directory...")
        res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res_retry.returncode == 0:
            print("[SUCCESS] Model executed successfully after retry.")
            return True
        else:
            print(f"[FAIL] Retry also failed:\nSTDOUT={res_retry.stdout}\nSTDERR={res_retry.stderr}")
    
    elif "already exists" in combined_error or "exists" in combined_error:
        print("[HEALING] Detected pre-existing directory conflict. Removing it again...")
        shutil.rmtree(out_dir, ignore_errors=True)
        print("[RETRY] Retrying after removing conflicting directory...")
        res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res_retry.returncode == 0:
            print("[SUCCESS] Model executed successfully after retry.")
            return True
        else:
            print(f"[FAIL] Retry also failed:\nSTDOUT={res_retry.stdout}\nSTDERR={res_retry.stderr}")

    # Final failure handling
    log_file = f"data/stage1_obs_{int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))}.txt"
    with open(log_file, "w") as f:
        f.write("=== Execution Failed ===\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"Return Code: {res.returncode}\n")
        f.write(f"STDOUT:\n{res.stdout}\n")
        f.write(f"STDERR:\n{res.stderr}\n")
    print(f"[FATAL] All attempts failed. Error details saved to {log_file}")
    return False

def inspect_output(out_dir, log_file):
    print(f"[INSPECT] Inspecting output directory: {out_dir}")
    files = []
    pattern_all = os.path.join(out_dir, "**/*")
    for file_path in glob.glob(pattern_all, recursive=True):
        if os.path.isfile(file_path):
            files.append(file_path)

    with open(log_file, "w") as lf:
        lf.write(f"=== Output Inspection Report ===\n")
        lf.write(f"Output Directory: {out_dir}\n\n")

        for idx, fp in enumerate(files):
            lf.write(f"\n--- File #{idx+1}: {fp} ---\n")
            try:
                if fp.endswith(".gz"):
                    with gzip.open(fp, "rt", encoding="utf-8") as gf:
                        lines = [next(gf) for _ in range(10)]
                        lf.writelines(lines)
                else:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as tf:
                        lines = [next(tf) for _ in range(10)]
                        lf.writelines(lines)
            except Exception as e:
                lf.write(f"[ERROR] Could not read file content: {str(e)}\n")

def main():
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    total_tasks = len(MODELS)

    if task_id >= total_tasks or task_id < 0:
        print(f"[WARNING] Invalid task_id {task_id}, valid range is [0, {total_tasks - 1}]")
        return

    selected_model = MODELS[task_id]
    success = run_model(selected_model)

    if success:
        log_file = f"data/stage1_obs_{task_id}.txt"
        inspect_output(selected_model["out_dir"], log_file)
        print(f"[DONE] Observation report written to {log_file}")

if __name__ == "__main__":
    main()
```

```bash
#!/bin/bash
#SBATCH -J amp_eval
#SBATCH --array=0-4%5
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
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

