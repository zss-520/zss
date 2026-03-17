import json
from config import (
    CONDA_SH_PATH,
    VLAB_ENV,
    FIRST_STAGE_OBSERVATION_TXT,
)

# =========================
# 文献全篇解析 Agent 提示词 (防幻觉高压模式)
# =========================
PAPER_ANALYST_PROMPT = """你是一位顶尖的计算生物学与 MLOps 文献解析专家。
你将接收到一篇学术论文的原文（可能已截断参考文献，但仍包含大量生物学背景）。

你的任务是像真实的研究员复现代码一样，主动跳过无关背景，寻找以下关键信息：
1. 扫描 "Data and Code Availability" 或全文寻找 GitHub/GitLab 链接。
2. 扫描 "Implementation Details" 寻找 Python 版本和核心依赖包 (如 torch, tensorflow, scikit-learn 等)。
3. 寻找执行推断或预测的命令行示例 (如 .py 或 .sh 的运行命令)。

🚨【极度严厉的纪律警告（CRITICAL RULES）】🚨
1. 🛑 绝对禁止幻觉 (NO HALLUCINATION)：你提取的 `repo_url` 必须在提供的原文中 100% 真实存在！如果你在文中找不到明确的代码链接，必须严格将 `repo_url` 填为 null！绝对不允许凭空捏造 Zenodo 或 GitHub 链接！你的任何伪造行为都会导致下游的物理超算节点直接崩溃！
2. 🛑 严禁漏报 (NO LAZINESS)：你必须逐字扫描 Abstract 末尾、Introduction 结尾以及 "Code Availability" 章节！只要原文中出现了开源仓库链接，你必须精准提取，绝不能因为文本太长而偷懒忽略！
3. 🛑 格式底线：只能输出合法的纯 JSON 字符串，绝对不要包含 ```json 标记，绝对不要有任何前言后语或思考过程。

请严格按照以下 JSON 结构输出：

{
  "paper_title": "文献标题或简称",
  "models": [
    {
      "model_name": "提取出的模型名称",
      "env_name": "为该模型建议的虚拟环境名称 (仅限小写字母和下划线)",
      "repo_url": "代码仓库地址 (如果原文没有明确提供，绝对不要瞎编，必须填 null！)",
      "weights_url": "权重下载说明 (如果没有请填 null)",
      "python_version": "所需的 Python 版本 (未提及请默认推断 3.9)",
      "dependencies": ["依赖包1", "依赖包2==1.0.0"],
      "inference_cmd_template": "推断命令的通用模板，请使用 {fasta_path} 代表输入，{output_dir} 代表输出目录。例如: python predict.py --input {fasta_path} --out {output_dir}"
    }
  ]
}
"""
# =========================
# 共享基础任务描述 (动态化)
# =========================
def build_base_task_desc(models_info: list[dict]) -> str:
    """
    根据记忆库传入的模型配置列表，动态生成任务描述
    """
    model_names = [m['model_name'] for m in models_info]
    
    # 动态构建清理逻辑和运行逻辑
    cleanup_instructions = ""
    run_instructions = ""
    
    for m in models_info:
        m_name = m['model_name']
        e_name = m['env_name']
        cmd_template = m['inference_cmd_template']
        
        # 将模板中的占位符替换为我们系统的实际路径
        actual_cmd = cmd_template.replace("{fasta_path}", "data/combined_test.fasta").replace("{output_dir}", f"data/{m_name}_out")
        
        cleanup_instructions += f"   - `shutil.rmtree('data/{m_name}_out', ignore_errors=True)` 或清理对应的结果文件\n"
        
        run_instructions += f"""
   # {m_name} 调用代码模板
   print("开始运行 {m_name}...")
   {m_name}_cmd = 'bash -c "source {CONDA_SH_PATH} && conda activate {e_name} && {actual_cmd}"'
   res_{m_name} = subprocess.run({m_name}_cmd, shell=True, capture_output=True, text=True)
   if res_{m_name}.returncode != 0:
       print(f"!!! {m_name} 真实报错日志:\\n{{res_{m_name}.stderr}}")
       raise RuntimeError("{m_name} 预测执行失败，已阻断程序！")
"""

    return f"""围绕以下从记忆库提取的预测模型：{', '.join(model_names)}，编写自动化代码。
请严格遵守以下共享基础要求：

1. 【历史数据清理机制】：
   在执行模型预测前，必须执行以下清理逻辑：
{cleanup_instructions}

2. 【强校验命令执行（极度重要）】：
   必须严格照抄以下代码块来执行预测（切勿自行添加 try-except 吞咽错误）：
{run_instructions}

3. 【代码完整性要求】：
   - Python 代码必须是单一、完整、可运行的脚本
   - 必须包含完整 import，包含 `def main():` 并以 `if __name__ == '__main__':` 为入口
"""

PI_PROMPT = """你是一位顶级的计算生物学 PI。当前的评测任务是：【{task_desc}】。
请引导 MLOps 工程师写出评测的 Python 脚本和 Slurm 提交脚本。特别提醒工程师：
1. 严禁编写从 FASTA 提取标签的代码，必须直接读取 ground_truth.csv！
2. 合并前 ID 必须进行 `split()[0].strip()` 强清洗，合并后必须用 `fillna(0.0)` 兜底。
3. 寻找概率列时必须校验 `dtype == float`，避开读取英文字符串标签导致的 DataFrame 灾难！
4. 保存 JSON 前，一定要记得将所有指标通过 `float(val)` 转为 Python 原生类型，防止 Numpy 序列化报错！同时规定 JSON 的双层字典嵌套结构。
5. 评测脚本必须是一个完整可运行的 Python 脚本，而不能只有片段。"""

CODER_PROMPT = f"""你是一位精通超算的 MLOps 工程师。根据 PI 的要求编写代码。

【代码输出的格式底线（极其重要）】：
1. 你提供的 Python 代码必须是一个**单一的、完全完整的脚本**，全部包含在一个 ` ```python ... ``` ` 代码块中！绝对不允许把代码切分成多个小块（比如仅输出清理逻辑，或者仅输出计算逻辑）。
2. 脚本必须包含所有的 `import`，必须有明确定义的 `def main():` 函数，并以 `if __name__ == '__main__':` 作为程序入口去调用 `main()`。如果缺少 `main()` 函数，我们的系统验证器会直接报错阻断！

【运行与评测要求】：
1. 必须提供 `run_eval.sh`，Bash 脚本必须写在单独的 ` ```bash ... ``` ` 块中，并参考以下模板：
   ```bash
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
   source {CONDA_SH_PATH}
   conda activate {VLAB_ENV}
   python eval_script.py
   echo "finish"
   ```"""

CRITIC_PROMPT = """你是一位极其严谨的独立审稿人与领域专家。
【我们的评测指标与权重分配】：{weights_info}
【客观运行结果数据】：{real_data}
【系统自动计算的量化总分 (百分制)】：{quantitative_scores}

请根据上述客观数据以及我们预设的量化打分公式，对参与评测的模型进行深度点评。
你的点评需要包含：
1. 分析量化总分：解释为何某个模型得分更高。
2. 结合应用场景：结合计算生物学或当前任务场景，点评这个权重分配是否科学。
3. 最终判定：基于系统的量化得分，给出你的最终审阅意见。
"""

# =========================
# 第一次会议专用：运行模型 + 勘探目录
# =========================
FIRST_MEETING_APPENDIX = f"""
【这次是第一次会议（探索勘探阶段），请严格执行以下要求】：
1. 本次会议的唯一目标是：运行给定列表中的所有模型，并“勘探”它们实际生成的文件结构。
2. 你的 Python 脚本在运行完模型（subprocess）后，必须编写代码遍历 `data/` 目录下所有带有 `_out` 后缀的目录或新生成的文件：
   - 使用 `glob` 或 `os.listdir` 找到模型实际生成的所有结果文件（特别是未知的 .gz, .csv, .txt）。
   - 尝试读取这些文件的头 5 到 10 行内容（如果发现是 .gz 文件，请 import gzip 读取纯文本）。
   - 将你找到的文件绝对/相对路径，以及文件的头部内容摘要，写出到一份勘探报告中。
3. 勘探报告必须保存为：`data/{FIRST_STAGE_OBSERVATION_TXT}`。
4. 绝对禁止（重要！）：本次会议的脚本绝对不许用 pandas 做任何 merge 操作！不许清洗数据！不许计算指标！
"""

# =========================
# 第二次会议专用：清洗数据 + 评测指标
# =========================
SECOND_MEETING_APPENDIX_TEMPLATE = """
【这次是第二次会议（利用评测阶段），请严格执行以下要求】：
1. PI 请先阅读下方【第一阶段超算传回的勘探报告】，明确第一次会议实际生成的文件名和数据结构。
2. 工程师必须根据勘探报告揭示的“真实文件路径和列名”编写 pandas 解析代码。绝对禁止重新运行任何模型命令！
3. 【数据清洗与合并（必须严格遵守）】：
   - 严禁解析 FASTA 获取标签：必须且只能读取 `data/ground_truth.csv` 获取真实标签。
   - 强清洗 ID：在 merge 之前，必须对所有表的 ID 列转换为字符串，剔除 '>' 符号，并使用 `apply(lambda x: str(x).split()[0].strip())` 只保留干净 ID。
   - 左连接兜底：必须以 `ground_truth` 为左表执行 `how='left'` merge。合并后，必须使用 `pd.to_numeric(..., errors='coerce').fillna(0.0)` 将缺失的概率值填补为 0.0。
   - 统一使用浮点数概率值（`prob > 0.5`）生成最终的 0 和 1 预测标签。
4. 必须使用 sklearn 计算 ACC, Recall, MCC, AUROC, AUPRC，并将结果保存为嵌套字典结构的 `eval_result.json`。
5. 必须保存 `evaluation_curves.png`（需声明 `import matplotlib; matplotlib.use('Agg')`）和 `final_results_with_predictions.csv`。
6. 🚫 你的执行环境是 Python 3.11+。绝对禁止使用 `from scipy import interp`！必须使用 `numpy.interp`！
7. 制图必须满足顶刊级别要求，全面对比出每个模型在各个指标上的差异。

以下是第一阶段传回的勘探报告：
--------------------------------
{stage1_context}
--------------------------------
"""

def build_first_meeting_agenda(models_info: list[dict]) -> str:
    base = build_base_task_desc(models_info)
    return base + "\n\n" + FIRST_MEETING_APPENDIX.strip()

def build_second_meeting_agenda(models_info: list[dict], stage1_context: str) -> str:
    base = build_base_task_desc(models_info)
    return base + "\n\n" + SECOND_MEETING_APPENDIX_TEMPLATE.format(
        stage1_context=stage1_context.strip() or "[未获取到第一阶段勘探报告，可能有报错]"
    ).strip()