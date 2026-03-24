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

非常感谢提供详细的任务描述。作为 PI，我会首先审阅任务需求并提炼出关键点和硬性约束条件，随后指导工程师编写符合所有规范的自动化评估脚本。

---

## ✅ **PI 审查与明确要求**

### 🔍 一、核心目标（探索勘探阶段）
> 在 Slurm Job Array 并行模式下，自动运行指定的五个生物信息学模型之一，并记录其输出文件结构及内容样本，用于后续分析流程设计。

### 🧪 二、输入控制机制
- 使用 `SLURM_ARRAY_TASK_ID` 控制当前执行哪个模型。
- 模型列表共 5 个，索引从 0 到 4。
- 若超出范围，则打印提示并安全退出。

### ⚙️ 三、模型执行策略（Job Array 单任务）
- 每次只运行一个模型。
- 不允许并发或合并操作。
- 必须支持动态环境变量获取方式启动。

### 🛡️ 四、健壮性保障（极度重要）
#### 1. 异常捕获原则：
- 所有模型执行必须包裹在 try-except 中。
- 禁止 raise 错误中断主线程。

#### 2. 输出目录处理逻辑：
| 条件 | 动作 |
|------|------|
| 执行前 | 删除旧目录（rmtree） |
| 第一次失败且含 “No such file” | 创建目录后重试 |
| 第一次失败且含 “already exists” | 删除目录后重试 |

#### 3. 日志输出规则：
- 每个任务独立写入日志文件：`data/stage1_obs_{task_id}.txt`
- 禁止任何形式的共享文件写入或锁机制！

### 📁 五、结果采集要求（勘探阶段）
- 遍历模型输出目录下的所有文件（包括隐藏文件）。
- 支持 `.gz`, `.csv`, `.txt` 等常见格式。
- 对每个文件提取前几行内容进行摘要。
- 写入到独立的日志文件中供人工查看。

### 🚫 六、禁用项（严重违规将导致任务失败）
- ❌ 不得使用 pandas。
- ❌ 不得做数据清洗或指标计算。
- ❌ 不得生成 sbatch 脚本。
- ❌ 不得使用统一观测文件（必须分片）。
- ❌ 不得提前创建输出目录（尤其注意 Macrel）。

---

## ✅ 最终确认：是否满足以下全部要求？

✅ 是单个可运行 Python 脚本  
✅ 包含完整 import  
✅ 主函数封装于 `main()`  
✅ 使用 `if __name__ == '__main__':` 启动  
✅ 根据 SLURM_ARRAY_TASK_ID 分配模型  
✅ 清理 + 自适应目录管理 + 双重重试机制  
✅ 捕获异常但不抛出  
✅ 每个任务写入独立 log 文件  
✅ 遍历输出目录并读取文件头若干行  
✅ 不使用 pandas / merge / lock  

---

## 👨‍💻 工程师，请你现在开始编写最终版本的 Python 脚本。

你可以基于上述审查要点来构建代码框架。如果你已完成初稿，请贴上你的实现；否则我可以为你草拟一份模板供参考。

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

def main():
    # 获取当前任务 ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # 定义模型配置列表
    models = [
        {
            "name": "Macrel",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives\"",
            "output_dir": "data/Macrel_out"
        },
        {
            "name": "AMP-Scanner-v2",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv\"",
            "output_dir": "data/AMP-Scanner-v2_out"
        },
        {
            "name": "amPEPpy",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model\"",
            "output_dir": "data/amPEPpy_out"
        },
        {
            "name": "AI4AMP",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv\"",
            "output_dir": "data/AI4AMP_out"
        },
        {
            "name": "AMPlify",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out\"",
            "output_dir": "data/AMPlify_out"
        }
    ]

    # 检查任务 ID 是否合法
    if task_id < 0 or task_id >= len(models):
        print(f"[ERROR] Invalid task_id {task_id}. Must be between 0 and {len(models)-1}.")
        return

    model = models[task_id]
    name = model["name"]
    cmd = model["cmd"]
    output_dir = model["output_dir"]

    # 设置日志文件路径（每个任务独立）
    log_file = f"data/stage1_obs_{task_id}.txt"

    with open(log_file, "w") as log:
        log.write(f"[INFO] Running model: {name}\n")
        log.write(f"[CMD] {cmd}\n\n")

        # 第一步：清理旧目录
        log.write("[CLEANUP] Removing previous output directory...\n")
        shutil.rmtree(output_dir, ignore_errors=True)

        # 第二步：尝试首次执行
        log.write("[EXECUTE] First attempt to run command...\n")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # 第三步：判断是否需要自愈重试
        retry_needed = False
        if res.returncode != 0:
            stderr_content = res.stderr.lower()
            stdout_content = res.stdout.lower()
            combined_error = stderr_content + stdout_content

            if "no such file" in combined_error or "not found" in combined_error or "failed to save" in combined_error:
                log.write("[RETRY] Detected missing directory error. Creating directory and retrying...\n")
                os.makedirs(output_dir, exist_ok=True)
                retry_needed = True
            elif "already exists" in combined_error or "exists" in combined_error:
                log.write("[RETRY] Detected existing directory conflict. Removing and retrying...\n")
                shutil.rmtree(output_dir, ignore_errors=True)
                retry_needed = True

        # 如果需要重试则再执行一次
        if retry_needed:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # 记录执行结果
        log.write(f"[RESULT] Return code: {res.returncode}\n")
        if res.returncode != 0:
            log.write("[STDERR]\n")
            log.write(res.stderr)
            log.write("\n[STDOUT]\n")
            log.write(res.stdout)
        else:
            log.write("[SUCCESS] Command executed successfully.\n")

        # 第四步：勘探输出目录结构
        log.write("\n[EXPLORATION] Scanning output directory...\n")
        if not os.path.exists(output_dir):
            log.write(f"[WARNING] Output directory does not exist: {output_dir}\n")
        else:
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                log.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = ' ' * 2 * (level + 1)
                for f in files:
                    file_path = os.path.join(root, f)
                    log.write(f"{subindent}{f}\n")

                    # 尝试读取文件头部内容
                    try:
                        if f.endswith(".gz"):
                            with gzip.open(file_path, 'rt', encoding='utf-8') as gf:
                                lines = [next(gf) for _ in range(5)]
                                log.write("".join(lines[:5]))
                        else:
                            with open(file_path, 'r', encoding='utf-8') as rf:
                                lines = [next(rf) for _ in range(5)]
                                log.write("".join(lines[:5]))
                    except Exception as e:
                        log.write(f"[READ ERROR] Could not read file: {str(e)}\n")
                log.write("\n")


if __name__ == "__main__":
    main()
```

```bash
#!/bin/bash
#SBATCH -J amp_eval
#SBATCH --array=0-4%4
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

