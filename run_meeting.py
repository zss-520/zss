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
    build_first_meeting_agenda,
    build_second_meeting_agenda,
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
            f"【PI_PROMPT 原文】\n{PI_PROMPT}\n"
        )

    if "stage2" in save_name_lower or "second" in save_name_lower:
        return (
            "下面请严格根据 PI 的要求输出最终结果。\n\n"
            "【这是第二次会议，必须严格基于第一次会议的标准化输出进行评测】\n"
            "1. 优先复用第一次会议已经生成的标准化输出文件。\n"
            "2. 如果第一次会议产物中已包含模型预测结果，则禁止重新运行模型命令。\n"
            "3. 本次必须产出：eval_result.json、evaluation_curves.png、"
            "final_results_with_predictions.csv。\n"
            "4. 本次要计算最终评测指标，并输出严格格式结果。\n\n"
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


# ======== 新增 models_info 参数 ========
def run_second_meeting(
    models_info: list[dict],
    stage1_output_dir: str | Path = MEETING_OUTPUT_DIR,
    save_dir: str | Path = MEETING_OUTPUT_DIR,
) -> dict:
    stage1_context = build_stage2_context_from_stage1_outputs(stage1_output_dir)
    agenda = build_second_meeting_agenda(models_info, stage1_context)
    discussion = run_two_agent_meeting(
        agenda=agenda,
        save_dir=save_dir,
        save_name=SECOND_MEETING_NAME,
    )
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
        "stage1_context": stage1_context,
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