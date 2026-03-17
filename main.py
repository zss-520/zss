import ast
import json
import os
from pathlib import Path
import PyPDF2
import re
from openai import OpenAI

from agent import Agent
from config import MODEL_NAME, FIRST_STAGE_OBSERVATION_TXT, METRIC_WEIGHTS, validate_runtime_config
from prompts import CRITIC_PROMPT
from data_prep import auto_prepare_local_data
from run_meeting import run_first_meeting, run_second_meeting
from workflow_utils import run_on_hpc_and_fetch

# ===== 新增导入数据库管理器 =====
from database_manager import ingest_new_paper, get_target_models_for_eval


def _validate_generated_python(py_code: str) -> tuple[bool, str]:
    if not py_code or not py_code.strip():
        return False, "Python 代码为空"

    try:
        tree = ast.parse(py_code)
    except SyntaxError as e:
        return False, f"Python 语法错误: {e}"

    has_import = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    if not has_import:
        return False, "缺少 import 语句"

    func_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if "main" not in func_names:
        return False, "缺少 main() 函数"

    if 'if __name__ == "__main__":' not in py_code and "if __name__ == '__main__':" not in py_code:
        return False, "缺少 __main__ 入口"

    return True, "OK"


def _calculate_weighted_scores(real_data: dict, weights: dict) -> dict:
    """遍历评测结果，根据配置的权重计算百分制量化得分"""
    scores = {}
    for model_name, metrics in real_data.items():
        if isinstance(metrics, dict):
            score = 0.0
            weight_sum = 0.0
            for m_name, m_val in metrics.items():
                matched_key = next((k for k in weights.keys() if k.lower() == m_name.lower()), None)
                if matched_key and isinstance(m_val, (int, float)):
                    score += m_val * weights[matched_key]
                    weight_sum += weights[matched_key]
            
            if weight_sum > 0:
                scores[model_name] = round((score / weight_sum) * 100, 2)
    return scores

def _run_critic(real_data: dict, save_directory: Path) -> str:
    scores = _calculate_weighted_scores(real_data, METRIC_WEIGHTS)
    
    weights_info = json.dumps(METRIC_WEIGHTS, ensure_ascii=False)
    scores_info = json.dumps(scores, ensure_ascii=False)
    
    critic_agent = Agent(
        model=MODEL_NAME,
        title="Critic",
        expertise="模型评测与结果分析",
        goal=CRITIC_PROMPT.format(
            real_data=json.dumps(real_data, indent=2, ensure_ascii=False),
            weights_info=weights_info,
            quantitative_scores=scores_info
        ),
        role="严苛的独立审稿人",
    )
    
    client = OpenAI()
    response = client.chat.completions.create(
        model=critic_agent.model,
        messages=[
            critic_agent.message, 
            {"role": "user", "content": "请根据系统传入的真实数据、权重规则以及量化总分，给出深入的科学点评。"}
        ],
        temperature=0.2,
    )
    
    critic_response = response.choices[0].message.content or ""
    md_path = save_directory / "critic_individual.md"
    with open(md_path, "w", encoding="utf-8") as f: 
        f.write(critic_response)
        
    return critic_response

def main():
    print("\n========== [Phase 0] 配置检查 ==========")
    validate_runtime_config(require_hpc=True)

    print("\n========== [Phase 0.5] 本地数据准备 ==========")
    if not auto_prepare_local_data("data"):
        print("!!! [Fatal] 本地数据准备失败，流程终止。")
        return

    # ================= 新增阶段 =================
    print("\n========== [Phase 0.6] 批量文献自动扫描与智能截断解析 ==========")
    
    papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)
    
    # 引入我们刚才在 database_manager 里加的新函数
    from database_manager import ingest_new_paper, get_target_models_for_eval, is_paper_processed, mark_paper_processed
    
    valid_extensions = {".pdf", ".txt"}
    paper_files = [f for f in papers_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if not paper_files:
        print(f">>> [提示] 文件夹 {papers_dir} 为空。请放入需要复现的文献 PDF 或 TXT。")
    
    new_models_found = False
    
    for paper_file in paper_files:
        filename = paper_file.name
        
        # 1. 记忆库查重：如果这篇文献以前处理过了，直接跳过
        if is_paper_processed(filename):
            print(f">>> [跳过] 文献 '{filename}' 之前已成功解析并入库，跳过读取。")
            continue
            
        print(f"\n>>> [新文献] 正在智能读取: {filename} ...")
        raw_text = ""
        
        # 2. 读取全文
        if paper_file.suffix.lower() == ".pdf":
            try:
                with open(paper_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            raw_text += page_text + "\n"
            except Exception as e:
                print(f"!!! [Error] 读取 PDF 失败 {filename}: {e}")
                continue
        else:
            with open(paper_file, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
                
        # 3. 智能截断：砍掉 References / Bibliography 之后的所有废话，省钱省算力！
        # 使用正则寻找独立的 References 标题（忽略大小写）
        match = re.search(r'\n\s*(references|bibliography|literature cited)\s*\n', raw_text, re.IGNORECASE)
        if match:
            original_len = len(raw_text)
            raw_text = raw_text[:match.start()]
            print(f"    -> [省流] 成功砍掉参考文献部分，字数从 {original_len} 减少至 {len(raw_text)}！")
        else:
            print(f"    -> [提示] 未精准匹配到参考文献标题，使用全文解析，字数: {len(raw_text)}")

        # 4. 送入 Paper Analyst 提取模型
        print(f"    -> 正在呼叫 Paper Analyst 提取模型参数...")
        success = ingest_new_paper(raw_text)
        
        # 5. 如果解析成功且没报错，打上“已处理”标记
        if success:
            mark_paper_processed(filename)
            new_models_found = True
            print(f"    -> [入库] '{filename}' 解析完毕，已加入已读清单。")

    # 6. 从记忆库中获取当前需要评测的模型列表
    models_info = get_target_models_for_eval()
    
    if not models_info:
        print("!!! [Fatal] 记忆库中未获取到任何可用的模型元数据，流程终止。")
        return
        
    print(f"\n>>> 当前准备推流到超算进行评测的模型总数: {len(models_info)}")
    for m in models_info:
        print(f"    - 模型: {m.get('model_name')} (仓库: {m.get('repo_url')})")
    # ============================================

    print("\n========== [Phase 0.8] 检查 OpenAI 兼容环境变量 ==========")
    print(f"OPENAI_BASE_URL = {os.getenv('OPENAI_BASE_URL')}")
    print(f"OPENAI_API_KEY 已设置 = {'是' if os.getenv('OPENAI_API_KEY') else '否'}")

    save_directory = Path("data/vlab_discussions")
    save_directory.mkdir(parents=True, exist_ok=True)

    print("\n========== [Phase 1] 第一次会议：生成统一金标准模型运行代码 ==========")
    # 这里将 models_info 传给第一次会议
    first_result = run_first_meeting(models_info=models_info, save_dir=save_directory)

    stage1_py_code = first_result.get("py_code", "")
    stage1_sh_code = first_result.get("sh_code", "")
    stage1_py_path = first_result.get("py_path", "")
    stage1_sh_path = first_result.get("sh_path", "")

    if not stage1_py_code:
        print("!!! [Fatal] 第一次会议未能提取到 Python 代码。")
        return

    ok, reason = _validate_generated_python(stage1_py_code)
    if not ok:
        print("!!! [Fatal] 第一次会议生成的 Python 代码不完整。")
        print(f"    原因: {reason}")
        return

    print(f">>> [Stage-1] Python脚本已保存: {stage1_py_path}")
    if stage1_sh_code:
        print(f">>> [Stage-1] Bash脚本已保存: {stage1_sh_path}")

    print("\n========== [Phase 2] 在超算执行第一次会议代码，产出标准化模型输出 ==========")
    stage1_fetch_targets = {
        "data/stage1_observation.txt": str(save_directory / "stage1_observation.txt"),
    }
    stage1_real_outputs = run_on_hpc_and_fetch(
        py_code=stage1_py_code,
        sh_code=stage1_sh_code,
        fetch_targets=stage1_fetch_targets,
        models_info=models_info,
    )

    if not stage1_real_outputs:
        print("!!! [Fatal] 第一次会议代码在超算未成功返回标准化模型输出。")
        return

    print("\n========== [Phase 3] 第二次会议：PI复核第一次输出后生成评测代码 ==========")
    # 这里将 models_info 传给第二次会议
    second_result = run_second_meeting(
        models_info=models_info,
        stage1_output_dir=save_directory,
        save_dir=save_directory,
    )

    stage2_py_code = second_result.get("py_code", "")
    stage2_sh_code = second_result.get("sh_code", "")
    stage2_py_path = second_result.get("py_path", "")
    stage2_sh_path = second_result.get("sh_path", "")
    stage1_context = second_result.get("stage1_context", "")

    if not stage2_py_code:
        print("!!! [Fatal] 第二次会议未能提取到 Python 代码。")
        return

    ok, reason = _validate_generated_python(stage2_py_code)
    if not ok:
        print("!!! [Fatal] 第二次会议生成的 Python 代码不完整。")
        print(f"    原因: {reason}")
        return

    print(f">>> [Stage-2] Python脚本已保存: {stage2_py_path}")

    stage1_context_path = save_directory / "stage1_context_for_stage2.txt"
    with open(stage1_context_path, "w", encoding="utf-8") as f:
        f.write(stage1_context)

    print("\n========== [Phase 4] 超算执行：评测并返回最终指标 ==========")
    real_data = run_on_hpc_and_fetch(
        py_code=second_result.get("py_code", ""),
        sh_code=second_result.get("sh_code", ""),
        fetch_targets={
            "eval_result.json": "./eval_result.json",
            "evaluation_curves.png": "data/evaluation_curves.png",
            "final_results_with_predictions.csv": "data/final_results_with_predictions.csv",
        },
    )
    if not real_data:
        print("!!! [Fatal] 第二次会议评测代码执行失败，未获得最终真实数据。")
        return

    print("\n========== [Phase 5] 科学审判：Critic 独立评议 ==========")
    critic_response = _run_critic(real_data=real_data, save_directory=save_directory)

    print("\n✨ >>> [审稿人 (Critic) 最终判决] <<< ✨\n")
    print(critic_response)

    workflow_summary = {
        "stage1": {
            "python_path": stage1_py_path,
            "bash_path": stage1_sh_path,
            "fetched_outputs": stage1_real_outputs,
        },
        "stage2": {
            "python_path": stage2_py_path,
            "bash_path": stage2_sh_path,
        },
        "final_outputs": real_data,
        "critic_md": str(save_directory / "critic_individual.md"),
    }
    summary_path = save_directory / "main_workflow_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(workflow_summary, f, ensure_ascii=False, indent=2)

    print(f"\n>>> [Done] 主流程摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()