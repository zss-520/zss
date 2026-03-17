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
from run_meeting import run_first_meeting, run_second_meeting
from workflow_utils import run_on_hpc_and_fetch

# ===== 新增导入数据库管理器 =====
from database_manager import ingest_new_paper, get_target_models_for_eval
import sys
import time
import threading
from config import CONDA_SH_PATH, VLAB_ENV

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

class WaitingSpinner:
    """一个优雅的终端动态等待动画"""
    def __init__(self, message=">>> 🧠 AI 正在高速运转中"):
        self.message = message
        self.done_event = threading.Event()
        self.thread = threading.Thread(target=self._spin)

    def _spin(self):
        # 酷炫的点阵旋转字符
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        while not self.done_event.is_set():
            sys.stdout.write(f"\r{self.message} {spinner_chars[i % len(spinner_chars)]} ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        # 任务结束后，用空白覆盖清理当前行
        sys.stdout.write('\r' + ' ' * (len(self.message.encode('utf-8')) + 20) + '\r')
        sys.stdout.flush()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done_event.set()
        self.thread.join()

def main():
    print("\n========== [Phase 0] 配置检查 ==========")
    validate_runtime_config(require_hpc=True)

    print("\n========== [Phase 0.5] 检查预处理好的多份数据集 ==========")
    base_datasets_dir = Path("data/datasets")
    if not base_datasets_dir.exists():
        print(f"!!! [Fatal] 未找到 {base_datasets_dir} 目录。请先独立运行 standalone_data_prep.py")
        return
        
    dataset_dirs = [d for d in base_datasets_dir.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print("!!! [Fatal] 数据集目录下没有子文件夹。")
        return
        
    print(f">>> 发现 {len(dataset_dirs)} 份待评测独立数据集:")
    for d in dataset_dirs:
        print(f"    - {d.name}")

    # ================= 新增/修改阶段 =================
    print("\n========== [Phase 0.6] 模型加载阶段 (支持自动解析与手动预设) ==========")
    
    # 【新功能】本地已搭建好环境的模型注册表 (随时可以在这里添加你自己配好的模型)
    LOCAL_PREBUILT_MODELS = {
        "Macrel": {
            "model_name": "Macrel",
            "env_name": "env_macrel",
            "repo_url": "",
            "dependencies": [],
            "inference_cmd_template": "macrel peptides --fasta {fasta_path} --output {output_dir}",
            "skip_env_setup": True
        },
        "AMP-Scanner-v2": {
            "model_name": "AMP-Scanner-v2",
            "env_name": "ascan2_tf1",
            "repo_url": "",
            "dependencies": [],
            "inference_cmd_template": "python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f {fasta_path} -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p {output_dir}/ampscanner_out.csv",
            "skip_env_setup": True
        }
    }
    # ---------------------------------------------------------
    # 控制开关：你想跑自动文献解析，还是直接跑手动指定的已有模型？
    # 选项: "auto" (自动扫文献)  或者  "manual" (直接指定)
    RUN_MODE = "manual"  
    # ---------------------------------------------------------

    models_info = []

    if RUN_MODE == "manual":
        # 在这里输入你想直接评测的模型名称（必须在上面的注册表中存在）
        # 你可以只写 ["Macrel"]，也可以同时测多个 ["Macrel", "AMP-Scanner-v2"]
        target_model_names = ["Macrel", "AMP-Scanner-v2"]
        
        print(f">>> [直接评测模式] 跳过文献解析，直接加载预设模型: {target_model_names}")
        for name in target_model_names:
            if name in LOCAL_PREBUILT_MODELS:
                models_info.append(LOCAL_PREBUILT_MODELS[name])
            else:
                print(f"!!! [警告] 模型 {name} 未在本地注册表中找到。")
                
    elif RUN_MODE == "auto":
        from database_manager import ingest_new_paper, get_target_models_for_eval, is_paper_processed, mark_paper_processed
        import PyPDF2
        
        print(">>> [自动解析模式] 开始扫描 data/papers 文件夹...")
        papers_dir = Path("data/papers")
        papers_dir.mkdir(parents=True, exist_ok=True)
        valid_extensions = {".pdf", ".txt"}
        paper_files = [f for f in papers_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        
        for paper_file in paper_files:
            filename = paper_file.name
            if is_paper_processed(filename):
                print(f">>> [跳过] 文献 '{filename}' 已处理。")
                continue
                
            print(f"\n>>> [新文献] 正在读取: {filename} ...")
            raw_text = ""
            if paper_file.suffix.lower() == ".pdf":
                try:
                    with open(paper_file, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                raw_text += page_text + "\n"
                except Exception as e:
                    print(f"!!! [Error] 读取 PDF 失败: {e}")
                    continue
            else:
                with open(paper_file, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
                    
            import re
            match = re.search(r'\n\s*(references|bibliography|literature cited)\s*\n', raw_text, re.IGNORECASE)
            if match:
                raw_text = raw_text[:match.start()]
                
            if ingest_new_paper(raw_text):
                mark_paper_processed(filename)
                
        models_info = get_target_models_for_eval()

    if not models_info:
        print("!!! [Fatal] 未获取到任何可用的模型元数据，流程终止。")
        return
    print(f"\n>>> 当前准备推流到超算进行评测的模型总数: {len(models_info)}")
    for m in models_info:
        print(f"    - 模型: {m.get('model_name')} (指定环境: {m.get('env_name')})")
    # ============================================

    import os
    import json
    print("\n========== [Phase 0.8] 检查 OpenAI 兼容环境变量 ==========")
    print(f"OPENAI_BASE_URL = {os.getenv('OPENAI_BASE_URL')}")
    print(f"OPENAI_API_KEY 已设置 = {'是' if os.getenv('OPENAI_API_KEY') else '否'}")

    save_directory = Path("data/vlab_discussions")
    save_directory.mkdir(parents=True, exist_ok=True)

    print("\n========== [Phase 1] 第一次会议：生成统一金标准模型运行代码 ==========")
    # 这里将 models_info 传给第一次会议
    with WaitingSpinner(">>> [Meeting] MLOps 工程师与 PI 正在激烈讨论并编写代码，请稍候..."):
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
    first_dataset_dir = dataset_dirs[0]
    print(f">>> [探路者] 使用首个数据集进行结构勘探: {first_dataset_dir.name}")
    
    stage1_fetch_targets = {
        "data/stage1_observation.txt": str(save_directory / "stage1_observation.txt"),
    }
    stage1_real_outputs = run_on_hpc_and_fetch(
        py_code=stage1_py_code,
        sh_code=stage1_sh_code,
        fetch_targets=stage1_fetch_targets,
        models_info=models_info,
        local_data_dir=str(first_dataset_dir)  # <--- 核心修改点
    )

    if not stage1_real_outputs:
        print("!!! [Fatal] 第一次会议代码在超算未成功返回标准化模型输出。")
        return

    print("\n========== [Phase 3] 第二次会议：PI复核第一次输出后生成评测代码 ==========")
    # 这里将 models_info 传给第二次会议
    with WaitingSpinner(">>> [Meeting] 审稿人正在复盘超算勘探报告，编写 Pandas 数据清洗逻辑..."):
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

   # ================= 🚀 核心修改：循环评测多份独立数据 =================
    print("\n========== [Phase 4 & Phase 5] 批量轮询评测与科学审判 ==========")
    import base64
    all_results_summary = {}
    
    stage1_b64 = base64.b64encode(stage1_py_code.encode('utf-8')).decode('utf-8')
    stage2_b64 = base64.b64encode(stage2_py_code.encode('utf-8')).decode('utf-8')

    combined_wrapper_code = f"""import os, subprocess, base64

# 1. 还原代码 (确保 utf-8 编码安全)
with open('stage1_runner.py', 'wb') as f:
    f.write(base64.b64decode('{stage1_b64}'.encode('utf-8')))
with open('stage2_eval.py', 'wb') as f:
    f.write(base64.b64decode('{stage2_b64}'.encode('utf-8')))

# 2. 依次按顺序执行
print("\\n" + "="*60)
print(">>> [HPC Runtime] 启动阶段 1: 运行所有 AI 模型进行预测...")
print("="*60)
subprocess.run('python stage1_runner.py', shell=True)

print("\\n" + "="*60)
print(">>> [HPC Runtime] 启动阶段 2: 提取数据、算分并生成图表...")
print("="*60)
subprocess.run('python stage2_eval.py', shell=True)
"""

    # 🛡️ 硬核强制的防弹 Bash 脚本，绝对不允许 AI 的 Bash 脚本来“夺舍”覆盖文件！
    safe_loop_sh_code = f"""#!/bin/bash
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

echo ">>> [HPC Runtime] Executing master wrapper..."
python eval_script.py
echo "finish"
"""

    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        print(f"\n=======================================================")
        print(f"   🚀 正在超算上独立评测数据集: [{ds_name}] ")
        print(f"=======================================================")
        
        res_dir = Path(f"data/results/{ds_name}")
        res_dir.mkdir(parents=True, exist_ok=True)
        
        real_data = run_on_hpc_and_fetch(
            py_code=combined_wrapper_code,  # 传入我们的母体脚本
            sh_code=safe_loop_sh_code,      # 传入我们硬编码的防弹 Bash 脚本
            fetch_targets={
                "eval_result.json": str(res_dir / "eval_result.json"),
                "evaluation_curves.png": str(res_dir / "evaluation_curves.png"),
                "final_results_with_predictions.csv": str(res_dir / "final_results_with_predictions.csv"),
            },
            models_info=models_info,
            local_data_dir=str(ds_dir) 
        )
        
        if not real_data:
            print(f"!!! [Error] 数据集 {ds_name} 评测执行失败。")
            continue
            
        print(f"\n>>> 🧪 针对数据集 [{ds_name}] 的科学审判：")
        critic_response = _run_critic(real_data=real_data, save_directory=res_dir)
        print(critic_response)
        
        all_results_summary[ds_name] = {
            "outputs": real_data,
            "critic_md": str(res_dir / "critic_individual.md")
        }

    # 最终汇总
    print("\n========== [Phase 6] 主流程结束，保存最终摘要 ==========")
    workflow_summary = {
        "datasets_results": all_results_summary
    }
    summary_path = save_directory / "main_workflow_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(workflow_summary, f, ensure_ascii=False, indent=2)

    print(f"\n✨ >>> [Done] 所有 {len(dataset_dirs)} 份数据集独立评测完毕！主流程摘要已保存: {summary_path}")

if __name__ == "__main__":
    main()