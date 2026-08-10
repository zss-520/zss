from pathlib import Path

from agent_md_loader import AgentMDLoader
from config import (
    HPC_TARGET_DIR,
    CONDA_SH_PATH,
)

PROMPT_DIR = Path(__file__).resolve().parent / "agents" / "runtime_prompts"
PROMPT_FILES = {
    "PAPER_ANALYST_PROMPT": "paper_analyst",
    "BENCHMARK_ARCHITECT_PROMPT": "benchmark_architect",
    "DATASET_ETL_AGENT_PROMPT": "dataset_etl_agent",
    "PI_PROMPT": "pi",
    "CODER_PROMPT": "coder",
    "CRITIC_PROMPT": "critic",
    "FIRST_MEETING_APPENDIX": "first_meeting_appendix",
    "DATA_ANALYST_EXTRACTION_PROMPT": "data_analyst_extraction",
    "DATA_ANALYST_REVIEW_PROMPT": "data_analyst_review",
    "SECOND_MEETING_PI_PROMPT": "second_meeting_pi",
    "SECOND_MEETING_APPENDIX_TEMPLATE": "second_meeting_appendix_template",
    "SECOND_MEETING_PI_SUMMARY_PROMPT": "second_meeting_pi_summary",
    "DOWNLOAD_GUIDE_PROMPT": "download_guide",
    "PAPER_PREPROCESSOR_PROMPT": "paper_preprocessor",
    "MULTI_AGENT_SCOUT_PROMPT": "multi_agent_scout",
    "MULTI_AGENT_METRICS_PROMPT": "multi_agent_metrics",
    "MULTI_AGENT_CRITIC_PROMPT": "multi_agent_critic",
    "MULTI_AGENT_CHIEF_PROMPT": "multi_agent_chief",
    "AMP_RESEARCH_ADVISOR_SYSTEM_PROMPT": "amp_research_advisor_system",
    "AMP_RESEARCH_ADVISOR_PROMPT_TEMPLATE": "amp_research_advisor_template",
}
_PROMPT_LOADER = AgentMDLoader(PROMPT_DIR)


def _load_named_prompt(name: str, replacements: dict[str, object] | None = None) -> str:
    """Load a UTF-8 Markdown prompt and replace only explicit config tokens."""
    try:
        stem = PROMPT_FILES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt constant: {name}") from exc
    text = _PROMPT_LOADER.load_composed(stem)
    for key, value in (replacements or {}).items():
        text = text.replace("{{config:" + key + "}}", str(value))
    if "{{config:" in text:
        raise RuntimeError(f"Unresolved config token in prompt {name}")
    return text



# =========================
# 文献全篇解析 Agent 提示词 (防幻觉高压模式 + 单模型收敛)
# =========================
PAPER_ANALYST_PROMPT = _load_named_prompt("PAPER_ANALYST_PROMPT")
# =========================
# 首席基准测试架构师 (Benchmark Architect) Prompt - 纯评测金标准版
# =========================
BENCHMARK_ARCHITECT_PROMPT = _load_named_prompt("BENCHMARK_ARCHITECT_PROMPT")
# =======================================================
# 数据集动态 ETL 工程师 (Dataset ETL Agent) Prompt
# =======================================================
DATASET_ETL_AGENT_PROMPT = _load_named_prompt("DATASET_ETL_AGENT_PROMPT")
# =========================
# 共享基础任务描述 (动态化与声明式)
# =========================
def build_base_task_desc(models_info: list[dict]) -> str:
    """
    根据记忆库传入的模型配置列表，动态生成任务描述。
    彻底抛弃硬编码，让 LLM 根据规则自主生成高质量、带自愈能力的保护性代码。
    """
    model_names = [m['model_name'] for m in models_info]
    
    model_execution_details = ""
    for m in models_info:
        m_name = m['model_name']
        e_name = m['env_name']
        cmd_template = m['inference_cmd_template']
        
        # 动态替换路径，强制每个模型拥有独立的输出空间
        actual_cmd = cmd_template.replace("{fasta_path}", "data/combined_test.fasta").replace("{output_dir}", f"data/{m_name}_out")
        if m_name == "sAMPpred-GAT" and "samppred_gat_adapter.py" in actual_cmd:
            remote_repo = str(m.get("remote_repo_dir") or "/share/home/zhangss/repos/samppred-gat").rstrip("/")
            actual_cmd = (
                f"cd {remote_repo} && "
                "python samppred_gat_adapter.py "
                f"--input {HPC_TARGET_DIR}/data/combined_test.fasta "
                f"--output {HPC_TARGET_DIR}/data/{m_name}_out/predictions.csv"
            )
        
        model_execution_details += f"- 【模型名称】: {m_name}\n"
        # 🚨【核心修复点1】：撤销硬编码的 mkdir -p，让 Python 脚本的自愈机制去建文件夹，防止激怒 Macrel
        model_execution_details += f"  【运行命令】: bash -c \"source {CONDA_SH_PATH} && conda activate {e_name} && {actual_cmd}\"\n"
        model_execution_details += f"  【输出目录】: data/{m_name}_out\n\n"
    return f"""当前需要评测的计算生物学模型清单：{', '.join(model_names)}。

请严格遵守以下编程规范，完全由你自主编写高质量的 Python 自动化评估脚本：

1. 【模型执行参数矩阵（并行 Job Array 模式）】：
你的 Python 脚本将被 Slurm Array 并行调用。请在脚本开头通过 `task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))` 获取当前任务索引。
根据 task_id 仅选取并执行以下列表中的【一个】对应模型（如果 task_id 超出范围请安全打印并退出）：
{model_execution_details}
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
"""

PI_PROMPT = _load_named_prompt("PI_PROMPT")
CODER_PROMPT = _load_named_prompt("CODER_PROMPT")

CRITIC_PROMPT = _load_named_prompt("CRITIC_PROMPT")

# =========================
# 第一次会议专用：运行模型 + 勘探目录
# =========================
FIRST_MEETING_APPENDIX = _load_named_prompt("FIRST_MEETING_APPENDIX")
# =========================
# 首席数据架构师 (Data Analyst) - 阶段1：提取 Schema
# =========================
DATA_ANALYST_EXTRACTION_PROMPT = _load_named_prompt("DATA_ANALYST_EXTRACTION_PROMPT")

# =========================
# 首席数据架构师 (Data Analyst) - 阶段2：代码审查 (Code Review)
# =========================
DATA_ANALYST_REVIEW_PROMPT = _load_named_prompt("DATA_ANALYST_REVIEW_PROMPT")

# =========================
# PI 的第二次会议开场白 (纯逻辑驱动，无硬编码)
# =========================
SECOND_MEETING_PI_PROMPT = _load_named_prompt("SECOND_MEETING_PI_PROMPT")

# =========================
# 第二次会议的补充模板
# =========================
SECOND_MEETING_APPENDIX_TEMPLATE = _load_named_prompt("SECOND_MEETING_APPENDIX_TEMPLATE")

# =========================
# PI 的第二次会议总结陈词 (强化路径与数据质控版)
# =========================
SECOND_MEETING_PI_SUMMARY_PROMPT = _load_named_prompt("SECOND_MEETING_PI_SUMMARY_PROMPT")
# =========================
# 资源拉取向导 Agent 提示词 (Human-in-the-loop 升级版)
# =========================
DOWNLOAD_GUIDE_PROMPT = _load_named_prompt("DOWNLOAD_GUIDE_PROMPT")
# =========================
# 多智能体文献规划会议 (Multi-Agent Orchestrator) Prompts
# =========================
# =======================================================
# 新增：前置精读 Agent (Map 阶段数据提炼)
# =======================================================
# 🚨 修改 prompts.py 中的 PAPER_PREPROCESSOR_PROMPT
PAPER_PREPROCESSOR_PROMPT = _load_named_prompt("PAPER_PREPROCESSOR_PROMPT")


def _meeting_turn(role_prompt: str, turn_instruction: str) -> str:
    """Combine one reusable role prompt with small turn-specific runtime context."""
    return role_prompt.strip() + "\n\n" + turn_instruction.strip()


_MULTI_AGENT_SCOUT_ROLE = _load_named_prompt("MULTI_AGENT_SCOUT_PROMPT")
_MULTI_AGENT_METRICS_ROLE = _load_named_prompt("MULTI_AGENT_METRICS_PROMPT")
_MULTI_AGENT_CRITIC_ROLE = _load_named_prompt("MULTI_AGENT_CRITIC_PROMPT")

MULTI_AGENT_SCOUT_PROMPT = _meeting_turn(_MULTI_AGENT_SCOUT_ROLE, """
## Turn: initial incremental proposal

Compare the historical consensus with the latest literature. Report new AMP binary-classification model and dataset candidates; if none are supported, say so explicitly.

Historical consensus:
{history_context}

Latest literature evidence:
{full_context}
""")

MULTI_AGENT_SCOUT_REBUTTAL_PROMPT = _meeting_turn(_MULTI_AGENT_SCOUT_ROLE, """
## Turn: response to Reviewer

Revise the candidate inventory in response to the Reviewer. Preserve rejected or deferred rows in the audit trace, but keep only code-gate-eligible AMP classifiers in the proposed executable list.

Initial Scout proposal:
{scout_report}

Reviewer audit:
{critic_report}
""")

MULTI_AGENT_METRICS_PROMPT = _meeting_turn(_MULTI_AGENT_METRICS_ROLE, """
## Turn: initial metric proposal

Propose priorities for the eligible metric keys found in the supplied literature context. Do not calculate benchmark values.

Literature evidence:
{full_context}
""")

MULTI_AGENT_METRICS_REBUTTAL_PROMPT = _meeting_turn(_MULTI_AGENT_METRICS_ROLE, """
## Turn: response to Reviewer

Revise the metric rationale after the Reviewer audit. The runtime will normalize and validate the numeric vector.

Initial Metrics proposal:
{metrics_report}

Reviewer audit:
{critic_report}
""")

MULTI_AGENT_CRITIC_PROMPT = _meeting_turn(_MULTI_AGENT_CRITIC_ROLE, """
## Turn: first audit

Audit both proposals and give concrete accept, reject or defer instructions.

Scout proposal:
{scout_report}

Metrics proposal:
{metrics_report}
""")

MULTI_AGENT_CRITIC_ROUND2_PROMPT = _meeting_turn(_MULTI_AGENT_CRITIC_ROLE, """
## Turn: final audit

Confirm whether the revised proposals resolved the first audit. Preserve remaining failures and uncertainty; do not issue the Chief's final decision.

Revised Scout proposal:
{scout_rebuttal}

Revised Metrics proposal:
{metrics_rebuttal}
""")

# 👇 修改：Chief 综合结果，强制要求“全量输出”，绝对拒绝 LLM 偷懒
MULTI_AGENT_CHIEF_PROMPT = _load_named_prompt("MULTI_AGENT_CHIEF_PROMPT")

# =======================================================
# 自动除虫委员会 (Auto-Debugging Committee) Prompts
# =======================================================
def build_first_meeting_agenda(models_info: list[dict]) -> str:
    base = build_base_task_desc(models_info)
    return base + "\n\n" + FIRST_MEETING_APPENDIX.strip()

# 👇 修复了这里缺少 gt_sample 参数导致的隐患！
def build_second_meeting_agenda(models_info: list[dict], stage1_context: str, gt_sample: str) -> str:
    model_names = [m['model_name'] for m in models_info]
    base = f"当前需要评测的计算生物学模型清单：{', '.join(model_names)}。\n"
    return base + "\n\n" + SECOND_MEETING_APPENDIX_TEMPLATE.format(
        stage1_context=stage1_context.strip() or "[未获取到勘探报告]",
        gt_sample=gt_sample.strip() or "[未获取到真值表样本]"
    ).strip()
# =========================
# AMP Research Advisor Prompt
# 基于跨数据集 benchmark 结果，生成 AMP 模型未来发展方向分析报告
# =========================

AMP_RESEARCH_ADVISOR_SYSTEM_PROMPT = _load_named_prompt("AMP_RESEARCH_ADVISOR_SYSTEM_PROMPT")


AMP_RESEARCH_ADVISOR_PROMPT_TEMPLATE = _load_named_prompt("AMP_RESEARCH_ADVISOR_PROMPT_TEMPLATE")

def build_amp_research_advisor_prompt(context_json: str, dynamic_metrics_text: str) -> str:
    """构建 AMP 研究发展建议报告提示词。"""
    prompt = AMP_RESEARCH_ADVISOR_PROMPT_TEMPLATE.format(
        context_json=context_json,
        dynamic_metrics_text=dynamic_metrics_text,
    )
    return prompt + """

【额外硬性要求：Top3 集成学习候选推荐】
请在报告中新增一个独立章节，标题必须为：

## Top3 集成学习候选模型推荐

这一节面向后续 ensemble / stacking 系统设计，必须满足：
1. 只允许从当前 benchmark 中有有效数值结果的模型里选择 Top3；如果有效模型少于 3 个，必须明确写“当前不足以组成 Top3”，并说明还需要补跑哪些模型。
2. Top3 排序必须同时参考动态指标总分、AUPRC、MCC、Recall、跨数据集稳定性和复现状态，不能只按单一指标排序。
2a. 最终排名必须以 context_json.iterative_weight_meeting.final_ranking 为准；重点解释 median_score、score_iqr 和 top3_frequency。不得为了得到预设模型而修改名次。
3. 必须解释为什么推荐集成学习：重点说明不同模型在特征来源、架构、错误模式、Precision/Recall 取舍、排序能力和阈值决策上是否互补。
4. 必须给出“互补性证据”：例如一个模型 AUPRC 高但 Recall 低，另一个模型 Recall 高但 Precision 更弱，二者可能通过 stacking、加权平均、rank fusion 或阈值分层互补。
5. 必须给出推荐的集成策略：至少比较 soft voting、rank averaging、stacking/meta-classifier、high-recall candidate union 四种策略中哪一种最适合当前结果。
6. 如果当前只有一个有效模型，不能硬凑 Top3，也不能声称已经可以做可靠集成；只能写“暂不推荐正式集成，建议继续补齐至少 3 个可复现模型后再训练 ensemble”。
7. 不要把当前未参与本轮评测、没有有效结果、或只是历史注册表中存在的模型写进 Top3。
8. 如果 context_json 中存在 target_model_names，则整份报告只能把这些模型视为本轮参评模型；不要把非目标模型写成“本轮不能运行”或“本轮失败”。
""".strip()
