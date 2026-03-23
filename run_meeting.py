"""两阶段 agent 会议编排：第一次生成模型运行代码，第二次基于第一次输出生成评测代码。"""

import json
from pathlib import Path
from openai import OpenAI

from agent import Agent
from config import (
    FIRST_MEETING_NAME,
    SECOND_MEETING_NAME,
    FIRST_STAGE_PY_NAME,
    FIRST_STAGE_SH_NAME,
    SECOND_STAGE_PY_NAME,
    SECOND_STAGE_SH_NAME,
    MEETING_OUTPUT_DIR,
    MODEL_NAME,
    validate_runtime_config,
)
from prompts import (
    PI_PROMPT,
    CODER_PROMPT,
    DATA_ANALYST_EXTRACTION_PROMPT,
    DATA_ANALYST_REVIEW_PROMPT,
    SECOND_MEETING_PI_PROMPT,
    SECOND_MEETING_PI_SUMMARY_PROMPT,
    build_first_meeting_agenda
)
from workflow_utils import (
    build_stage2_context_from_stage1_outputs,
    save_generated_code_from_meeting,
)


DEFAULT_PI = Agent(
    title="PI",
    expertise="computational biology, AMP evaluation, scientific review",
    goal="guide the engineer to produce correct, runnable, format-compliant code",
    role="review task constraints, restate hard requirements, and push the coder toward robust implementation",
    model=MODEL_NAME,
)

DEFAULT_CODER = Agent(
    title="Code Engineer",
    expertise="HPC, Slurm, Python, pandas, sklearn, MLOps",
    goal="deliver a complete Python script and a complete bash submission script",
    role="write full runnable code strictly following the PI's requirements and output format constraints",
    model=MODEL_NAME,
)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_discussion(discussion: list[dict], save_dir: Path, save_name: str) -> None:
    _ensure_dir(save_dir)
    json_path = save_dir / f"{save_name}.json"
    md_path = save_dir / f"{save_name}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(discussion, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        for item in discussion:
            f.write(f"## {item['agent']}\n\n{item['message']}\n\n")

    print(f">>> [Meeting] 会议记录已保存: {json_path}")
    print(f">>> [Meeting] Markdown 已保存: {md_path}")


def _chat_once(client: OpenAI, agent: Agent, messages: list[dict], temperature: float) -> str:
    response = client.chat.completions.create(
        model=agent.model,
        messages=[agent.message, *messages],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _build_coder_input_for_stage(save_name: str) -> str:
    save_name_lower = save_name.lower()

    if "stage1" in save_name_lower or "first" in save_name_lower:
        return (
            "下面请严格根据 PI 的要求输出最终结果。\n\n"
            "【这是第一次会议，必须严格遵守以下额外约束】\n"
            "1. 本次 Python 脚本的唯一目标是：运行模型、勘探输出目录、保存勘探报告。\n"
            "2. 本次必须产出：data/stage1_observation.txt\n"
            "3. 本次禁止做数据合并、禁止计算指标、禁止画图。\n"
            "4. Python 脚本必须包含所有 import，必须定义 def main():。\n"
            "5. 必须同时输出完整的 Bash 提交脚本，文件名为 stage1_run_model.sh。\n"
            "6. stage1_run_model.sh 必须显式申请 GPU，且必须包含：\n"
            "   #SBATCH --gres=gpu:{SLURM_GPUS}\n"
            "7. 如果使用 Job Array，则每个 array task 至少申请 1 张 GPU；禁止生成不申请 GPU 的脚本。\n\n"
            f"【PI_PROMPT 原文】\n{PI_PROMPT}\n\n"
            f"【CODER_PROMPT 原文】\n{CODER_PROMPT}"
        )

    if "stage2" in save_name_lower or "second" in save_name_lower:
        return (
            "下面请严格听从 PI 的指挥！**请务必基于 PI 刚才对真实数据的分析结果（文件格式、真实列名等）**，输出最终评测结果。\n\n"
            "【这是第二次会议，必须严格基于第一次会议的标准化输出进行评测】\n"
            "1. 优先复用第一次会议已经生成的标准化输出文件。\n"
            "2. 如果第一次会议产物中已包含模型预测结果，则禁止重新运行模型命令。\n"
            "3. 本次必须产出：eval_result.json、evaluation_curves.png、final_results_with_predictions.csv。\n"
            "4. 本次要计算最终评测指标，并输出严格格式结果。\n"
            "5. 🚨【代码格式死命令】：你输出的必须是一个完整的、可独立运行的 Python 脚本！必须在文件最开头包含所有用到的 `import` 语句（如 `import pandas as pd`, `import numpy as np`, `import glob`, `from sklearn.metrics import ...` 等），并且必须包含 `def main():` 结构！\n\n"
            f"【PI_PROMPT 原文】\n{PI_PROMPT}\n\n"
            f"【CODER_PROMPT 原文】\n{CODER_PROMPT}"
        )

    return (
        "下面请严格根据 PI 的要求输出最终结果。\n\n"
        f"【PI_PROMPT 原文】\n{PI_PROMPT}\n\n"
        f"【CODER_PROMPT 原文】\n{CODER_PROMPT}"
    )


def run_two_agent_meeting(
    agenda: str,
    save_dir: str | Path,
    save_name: str,
    pi_agent: Agent | None = None,
    coder_agent: Agent | None = None,
    temperature: float = 0.2,
) -> list[dict]:
    client = OpenAI()
    pi_agent = pi_agent or DEFAULT_PI
    coder_agent = coder_agent or DEFAULT_CODER
    save_dir = _ensure_dir(save_dir)

    discussion: list[dict] = []
    messages: list[dict] = []

    # 🚨 修改点 2：在会议开场白时，针对 Stage 2 强行给 PI 派发“数据分析”任务
    save_name_lower = save_name.lower()
    if "stage2" in save_name_lower or "second" in save_name_lower:
        user_start = (
            "现在开始第二次会议（评测与算分阶段）。\n\n"
            "作为 PI，请你**首先仔细阅读并分析**下方的【第一阶段勘探报告】（重点观察各模型真实生成的文件后缀、表头列名、分隔符等数据特征）。\n"
            "在充分剖析数据特征后，请针对性地向 MLOps 工程师提出极其明确的 Pandas 数据清洗与合并的防崩溃策略。\n\n"
            f"会议任务及第一阶段勘探报告内容如下：\n{agenda}"
        )
    else:
        user_start = (
            "现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。\n\n"
            f"会议任务如下：\n{agenda}"
        )
    messages.append({"role": "user", "content": user_start})
    discussion.append({"agent": "User", "message": user_start})

    pi_msg = _chat_once(client, pi_agent, messages, temperature)
    messages.append({"role": "assistant", "content": pi_msg})
    discussion.append({"agent": pi_agent.title, "message": pi_msg})

    coder_input = _build_coder_input_for_stage(save_name)
    messages.append({"role": "user", "content": coder_input})
    discussion.append({"agent": "User", "message": coder_input})

    coder_msg = _chat_once(client, coder_agent, messages, temperature)
    messages.append({"role": "assistant", "content": coder_msg})
    discussion.append({"agent": coder_agent.title, "message": coder_msg})

    _save_discussion(discussion, save_dir, save_name)
    return discussion


# ======== 新增 models_info 参数 ========
def run_first_meeting(models_info: list[dict], save_dir: str | Path = MEETING_OUTPUT_DIR) -> dict:
    agenda = build_first_meeting_agenda(models_info)
    discussion = run_two_agent_meeting(
        agenda=agenda,
        save_dir=save_dir,
        save_name=FIRST_MEETING_NAME,
    )
    py_code, sh_code, py_path, sh_path = save_generated_code_from_meeting(
        discussion,
        save_dir,
        py_filename=FIRST_STAGE_PY_NAME,
        sh_filename=FIRST_STAGE_SH_NAME,
        meeting_dump_name=FIRST_MEETING_NAME,
    )
    return {
        "discussion": discussion,
        "py_code": py_code,
        "sh_code": sh_code,
        "py_path": py_path,
        "sh_path": sh_path,
    }


def run_second_meeting(
    models_info: list[dict],
    stage1_output_dir: str | Path = MEETING_OUTPUT_DIR,
    save_dir: str | Path = MEETING_OUTPUT_DIR,
    gt_sample: str = "[未提供真实数据切片]",  # <=== 补上这行，接收真值表样本
) -> dict:
    save_dir = _ensure_dir(save_dir)
    client = OpenAI()
    
    # 获取第一阶段勘探报告
    raw_stage1_context = build_stage2_context_from_stage1_outputs(stage1_output_dir)

    if gt_sample and gt_sample != "[未提供真实数据切片]":
        raw_stage1_context += (
            f"\n\n【附：本地真值表 ground_truth.csv 的实际数据切片】\n"
            f"-------------------------\n{gt_sample}\n-------------------------\n"
            f"请严格参照上述真值表头进行 Pandas 的重命名和合并操作！"
        )
    
    # ==========================================
    # 智能体定义
    # ==========================================
    analyst_agent = Agent(
        title="Data Architect",
        expertise="数据解析与代码质检",
        goal="精准解析数据结构，并严苛审查代码中的数据泄露与维度爆炸 Bug",
        role="首席数据质检官",
        model=MODEL_NAME
    )
    pi_agent = Agent(
        title="PI",
        expertise="计算生物学架构规划",
        goal="制定评测逻辑，统筹团队工作",
        role="项目经理",
        model=MODEL_NAME
    )
    coder_agent = Agent(
        title="MLOps Coder",
        expertise="Python, Pandas, Sklearn",
        goal="根据逻辑和约束写出无 bug 的完整脚本",
        role="核心开发工程师",
        model=MODEL_NAME
    )

    discussion = []
    messages = []  # 共享的消息总线

    # ==========================================
    # 第一幕：分析师登场，提取 Schema
    # ==========================================
    print("\n>>> [Agent Debate] 第一幕：数据架构师解析勘探报告...")
    analyst_prompt = DATA_ANALYST_EXTRACTION_PROMPT.format(stage1_context=raw_stage1_context)
    schema_json_str = _chat_once(client, analyst_agent, [{"role": "user", "content": analyst_prompt}], 0.1)
    schema_json_str = schema_json_str.strip().replace("```json", "").replace("```", "")
    
    discussion.append({"agent": "Data Architect", "message": f"我已经成功提取了 Data Schema：\n```json\n{schema_json_str}\n```"})
    print(">>> Schema 提取完成！")

    # ==========================================
    # 第二幕：PI 下达逻辑指令
    # ==========================================
    print(">>> [Agent Debate] 第二幕：PI 下达逻辑架构约束...")
    pi_sys = SECOND_MEETING_PI_PROMPT.format(schema_json=schema_json_str)
    pi_msg1 = _chat_once(client, pi_agent, [{"role": "user", "content": pi_sys}], 0.2)
    
    messages.append({"role": "user", "content": f"PI 刚刚发布了任务规划：\n{pi_msg1}"})
    discussion.append({"agent": "PI", "message": pi_msg1})

    # ==========================================
    # 第三幕：Coder 编写第一版代码 (V1)
    # ==========================================
    print(">>> [Agent Debate] 第三幕：工程师正在编写初版代码 (V1)...")
    coder_req = (
        f"工程师你好，请严格遵守刚才 PI 的规划，以及你的基本架构纪律，编写评测脚本。\n"
        f"【你的核心纪律】：\n{CODER_PROMPT}\n"
    )
    messages.append({"role": "user", "content": coder_req})
    coder_msg1 = _chat_once(client, coder_agent, messages, 0.2)
    
    messages.append({"role": "assistant", "content": coder_msg1}) # 记录到总线
    discussion.append({"agent": "MLOps Coder (V1)", "message": coder_msg1})

    # ==========================================
    # 第四幕：分析师进行 Code Review
    # ==========================================
    print(">>> [Agent Debate] 第四幕：数据架构师进行严苛的代码审查 (Code Review)...")
    review_req = DATA_ANALYST_REVIEW_PROMPT.format(schema_json=schema_json_str)
    # 给分析师看工程师写的代码
    review_msg_input = [{"role": "user", "content": f"{review_req}\n\n【工程师编写的代码】：\n{coder_msg1}"}]
    analyst_review = _chat_once(client, analyst_agent, review_msg_input, 0.2)
    
    messages.append({"role": "user", "content": f"数据架构师给出了代码审查意见：\n{analyst_review}"})
    discussion.append({"agent": "Data Architect (Review)", "message": analyst_review})
    print(f"    -> 审查意见: {analyst_review[:100]}...")

    # ==========================================
    # 第五幕：PI 总结并下达最终命令
    # ==========================================
    print(">>> [Agent Debate] 第五幕：PI 总结审查意见，下达最终指令...")
    pi_summary = _chat_once(client, pi_agent, messages + [{"role": "user", "content": SECOND_MEETING_PI_SUMMARY_PROMPT}], 0.2)
    
    messages.append({"role": "user", "content": f"PI 的最终指示：\n{pi_summary}\n\n请工程师直接输出最终修正后的完整 Python 代码和 Bash 脚本！"})
    discussion.append({"agent": "PI (Summary)", "message": pi_summary})

    # ==========================================
    # 第六幕：Coder 提交最终版代码 (V2)
    # ==========================================
    print(">>> [Agent Debate] 第六幕：工程师正在提交最终版无 Bug 代码 (V2)...")
    coder_msg2 = _chat_once(client, coder_agent, messages, 0.2)
    discussion.append({"agent": "MLOps Coder (Final)", "message": coder_msg2})

    # ==========================================
    # 闭幕：提取与保存代码
    # ==========================================
    _save_discussion(discussion, save_dir, SECOND_MEETING_NAME)
    py_code, sh_code, py_path, sh_path = save_generated_code_from_meeting(
        discussion,
        save_dir,
        py_filename=SECOND_STAGE_PY_NAME,
        sh_filename=SECOND_STAGE_SH_NAME,
        meeting_dump_name=SECOND_MEETING_NAME,
        
    )
    
    return {
        "discussion": discussion,
        "py_code": py_code,
        "sh_code": sh_code,
        "py_path": py_path,
        "sh_path": sh_path,
        "stage1_context": schema_json_str,
    }

def run_full_two_stage_workflow(models_info: list[dict], save_dir: str | Path = MEETING_OUTPUT_DIR) -> dict:
    save_dir = _ensure_dir(save_dir)
    validate_runtime_config(require_hpc=False)

    print("\n>>> [Stage-1] 第一次会议：生成统一金标准数据集跑模型的代码")
    first_result = run_first_meeting(models_info=models_info, save_dir=save_dir)

    print("\n>>> [Stage-2] 第二次会议：PI 复核第一次输出后，生成严格格式的评测代码")
    second_result = run_second_meeting(models_info=models_info, stage1_output_dir=save_dir, save_dir=save_dir)

    summary = {
        "stage1": {
            "python_path": first_result["py_path"],
            "bash_path": first_result["sh_path"],
        },
        "stage2": {
            "python_path": second_result["py_path"],
            "bash_path": second_result["sh_path"],
        },
    }
    summary_path = Path(save_dir) / "workflow_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f">>> [Done] 两阶段会议流程摘要已保存到: {summary_path}")
    return {
        "first_result": first_result,
        "second_result": second_result,
        "summary_path": str(summary_path),
    }