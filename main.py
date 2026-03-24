import ast
import json
import os
from pathlib import Path
import re
from openai import OpenAI
from vanguard import run_vanguard_exploration
import pandas as pd
from agent import Agent
from config import MODEL_NAME, FIRST_STAGE_OBSERVATION_TXT, METRIC_WEIGHTS, validate_runtime_config
from prompts import CRITIC_PROMPT
from run_meeting import run_first_meeting, run_second_meeting
from workflow_utils import run_on_hpc_and_fetch
import queue
from concurrent.futures import ThreadPoolExecutor
# ===== 新增导入数据库管理器 =====
from database_manager import ingest_new_paper, get_target_models_for_eval
import sys
import time
import threading
from config import (
    SLURM_CPUS_PER_TASK,
    SLURM_GPUS,
    SLURM_PARTITION,
    HPC_TARGET_DIR,
    CONDA_SH_PATH,
    VLAB_ENV
)

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

    print("\n========== [Phase 0.6] 从静态注册表极速加载模型 ==========")
    registry_path = "data/local_registry.json"
    if not os.path.exists(registry_path):
        print(f"!!! [Fatal] 找不到注册表 {registry_path}！请先运行 prepare_models.py 扫描并下载模型！")
        return
        
    with open(registry_path, "r", encoding="utf-8") as f:
        all_models = json.load(f)

    # ---------------------------------------------------------
    # 🎯 在这里配置本次运行你想评测的模型名字 (可自由组合)
    # ---------------------------------------------------------
    target_model_names = ["AMP-Scanner-v2", "Macrel","amPEPpy","AI4AMP","AMPlify"] # <--- 在这里填入你想测的模型
    
    models_info = [m for m in all_models if m.get('model_name') in target_model_names]
    
    if not models_info:
        print(f"!!! [Fatal] 在注册表中没有找到目标模型: {target_model_names}")
        print("当前注册表包含的模型有:", [m['model_name'] for m in all_models])
        return
        
    print(f">>> 成功加载 {len(models_info)} 个模型配置 (含已挂载的目录树！)")
    for m in models_info:
        print(f"    - 模型: {m['model_name']} (指定环境: {m['env_name']})")


    print("\n========== [Phase 0.8] 检查 OpenAI 兼容环境变量 ==========")
    print(f"OPENAI_BASE_URL = {os.getenv('OPENAI_BASE_URL')}")
    print(f"OPENAI_API_KEY 已设置 = {'是' if os.getenv('OPENAI_API_KEY') else '否'}")

    save_directory = Path("data/vlab_discussions")
    save_directory.mkdir(parents=True, exist_ok=True)

   # =========================================================================
    # ⚡ [智能缓存系统] 拆分逻辑：缓存“脑力劳动”，强制“数据运行”
    # =========================================================================
    stage1_py_cache_path = save_directory / "stage1_code_cache.py"
    merged_obs_path = save_directory / "stage1_observation.txt"

    # 新增：记录上一次勘探的是哪个数据集
    last_ds_record_path = save_directory / "last_explored_dataset.txt"

    first_dataset_dir = dataset_dirs[0]
    current_ds_name = first_dataset_dir.name

    # 检查：代码存在、报告存在、且上一次测的也是这个数据集
    has_cache = (
        stage1_py_cache_path.exists() and 
        merged_obs_path.exists() and 
        last_ds_record_path.exists() and
        last_ds_record_path.read_text().strip() == current_ds_name
    )
    if has_cache:
        print(f"\n========== ⚡ 触发全量缓存：跳过针对 [{current_ds_name}] 的超算连接 ==========")
        with open(stage1_py_cache_path, "r", encoding="utf-8") as f:
            stage1_py_code = f.read()
    else:
        print(f"\n========== 🔍 缓存失效或数据集切换：正在为 [{current_ds_name}] 重新连接超算 ==========")
        
        # 1. 检查代码缓存（脑力劳动）
        if stage1_py_cache_path.exists():
            with open(stage1_py_cache_path, "r", encoding="utf-8") as f:
                stage1_py_code = f.read()
        else:
            # 执行 run_first_meeting 并保存 stage1_py_cache_path ... (此处省略旧逻辑)
            pass

    # --- 步骤 1：获取第一阶段 Python 代码 (只缓存这一步) ---
    if stage1_py_cache_path.exists():
        print("\n========== [Phase 1] ⚡ 发现代码缓存，跳过第一次会议 ==========")
        with open(stage1_py_cache_path, "r", encoding="utf-8") as f:
            stage1_py_code = f.read()
    else:
        print("\n========== [Phase 1] 第一次会议：生成模型运行代码 ==========")
        with WaitingSpinner(">>> [Meeting] MLOps 工程师与 PI 正在激烈讨论..."):
            first_result = run_first_meeting(models_info=models_info, save_dir=save_directory)

        stage1_py_code = first_result.get("py_code", "")
        if not stage1_py_code:
            print("!!! [Fatal] 第一次会议未能提取到代码。"); return

        ok, reason = _validate_generated_python(stage1_py_code)
        if not ok:
            print(f"!!! [Fatal] 代码语法错误: {reason}"); return

        with open(stage1_py_cache_path, "w", encoding="utf-8") as f:
            f.write(stage1_py_code)
        print(f">>> [Cache Saved] 运行脚本已缓存。")

    # --- 步骤 2：强制执行超算勘探 (针对当前运行环境，不设缓存) ---
    print("\n========== [Phase 2] 执行超算勘探，刷新标准化观测数据 ==========")
    # 每次运行 main.py 都强制去超算跑一次探路，确保拿到的是当前最真实的数据结构
    first_dataset_dir = dataset_dirs[0]
    print(f">>> [探路者] 使用首个数据集进行结构勘探: {first_dataset_dir.name}")
    
    import base64
    stage1_b64_phase2 = base64.b64encode(stage1_py_code.encode('utf-8')).decode('utf-8')
    
    # 🚨 关键：在超算端执行前，强制清空旧的 obs 文件
    exploration_wrapper = f"""import os, subprocess, base64, glob
for f in glob.glob('data/stage1_obs_*.txt') + ['data/stage1_observation.txt']:
    if os.path.exists(f): os.remove(f)

with open('ai_stage1_runner.py', 'wb') as f:
    f.write(base64.b64decode('{stage1_b64_phase2}'.encode('utf-8')))

print(">>> [探路者] 正在生成最新勘探报告...")
for i in range({len(models_info)}):
    env = os.environ.copy()
    env['SLURM_ARRAY_TASK_ID'] = str(i)
    # 强制给探路者分配 GPU-0，一视同仁
    env['CUDA_VISIBLE_DEVICES'] = "0"
    res = subprocess.run('export CUDA_VISIBLE_DEVICES=0 && python ai_stage1_runner.py', shell=True, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        with open(f'data/stage1_obs_{{i}}.txt', 'w', encoding='utf-8') as err_file:
            err_file.write(f"!!! 勘探崩溃 !!!\\n{{res.stderr}}")
"""

    stage1_fetch_targets = {
        f"data/stage1_obs_{i}.txt": str(save_directory / f"stage1_obs_{i}.txt") 
        for i in range(len(models_info))
    }
    
    stage1_real_outputs = run_on_hpc_and_fetch(
        py_code=exploration_wrapper, 
        sh_code="",  
        fetch_targets=stage1_fetch_targets,
        models_info=models_info,
        local_data_dir=str(first_dataset_dir),
        use_sbatch=True 
    )

    # 合并最新鲜的碎片文件
    with open(merged_obs_path, "w", encoding="utf-8") as fout:
        for i in range(len(models_info)):
            frag_path = save_directory / f"stage1_obs_{i}.txt"
            if frag_path.exists():
                with open(frag_path, "r", encoding="utf-8") as fin:
                    content = fin.read()
                    if "!!! 勘探崩溃 !!!" in content:
                        print(f"\n!!! [Fatal] 探路任务 {i} 在超算崩溃！"); import sys; sys.exit(1)
                    fout.write(content + "\n")
    
    print(">>> [Done] 勘探报告已刷新并合并。")
    # 3. 🚨 关键：成功合并报告后，记录下这次勘探的数据集名字
    last_ds_record_path.write_text(current_ds_name)
    print(f">>> [Success] 已完成对 {current_ds_name} 的结构锁定，下次运行将自动跳过。")
    # =========================================================================
    print("\n========== [Phase 3] 第二次会议：PI复核第一次输出后生成评测代码 ==========")
    first_dataset_dir = dataset_dirs[0]
    gt_sample_text = "[未找到 ground_truth.csv]"
    gt_file_path = first_dataset_dir / "ground_truth.csv"  # 改成 first_dataset_dir
    
    if gt_file_path.exists():
        try:
            # 只读前两行，足够架构师分析出表头和数据格式了
            gt_df_sample = pd.read_csv(gt_file_path, nrows=2)
            gt_sample_text = gt_df_sample.to_markdown(index=False)
        except Exception as e:
            gt_sample_text = f"[读取 ground_truth.csv 失败: {e}]"
            
    # 把真值表的样本强行追加到 stage1_output_dir 里的 observation 文件中
    # 或者你可以直接修改 SECOND_MEETING_APPENDIX_TEMPLATE，把这个 sample 传进去
    with WaitingSpinner(">>> [Meeting] 审稿人正在复盘超算勘探报告，编写 Pandas 数据清洗逻辑..."):
        second_result = run_second_meeting(
            models_info=models_info,
            stage1_output_dir=save_directory,
            save_dir=save_directory,
            gt_sample=gt_sample_text
        )
    stage2_py_code = second_result.get("py_code", "")
    stage2_sh_code = second_result.get("sh_code", "")
    stage2_py_path = second_result.get("py_path", "")
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

    print("\n========== [Phase 4 & Phase 5] 批量轮询评测与科学审判 ==========")
    import base64
    all_results_summary = {}
    
    stage1_b64 = base64.b64encode(stage1_py_code.encode('utf-8')).decode('utf-8')
    stage2_b64 = base64.b64encode(stage2_py_code.encode('utf-8')).decode('utf-8')

    combined_wrapper_code = f"""import os, subprocess, base64, glob, shutil
import queue
from concurrent.futures import ThreadPoolExecutor

# 🚨 【数据集隔离强化】：在开始任何预测前，清空所有潜在的旧干扰项
for old_file in glob.glob('data/stage1_obs_*.txt') + ['data/stage1_observation.txt', 'eval_result.json']:
    if os.path.exists(old_file): os.remove(old_file)
if os.path.exists('data/Macrel_out'): shutil.rmtree('data/Macrel_out', ignore_errors=True)
if os.path.exists('data/AMP-Scanner-v2_out'): shutil.rmtree('data/AMP-Scanner-v2_out', ignore_errors=True)

with open('stage1_runner.py', 'wb') as f:
    f.write(base64.b64decode('{stage1_b64}'.encode('utf-8')))
with open('stage2_eval.py', 'wb') as f:
    f.write(base64.b64decode('{stage2_b64}'.encode('utf-8')))

print("\\n" + "="*60)
print(">>> [HPC Runtime] 启动阶段 1: 节点内多卡并行火力全开...")

num_gpus = {SLURM_GPUS}
free_gpus = queue.Queue()
for g in range(num_gpus): 
    free_gpus.put(g)

def run_model(i):
    # 从队列中获取显卡号
    gpu_id = free_gpus.get()
    env = os.environ.copy()
    env['SLURM_ARRAY_TASK_ID'] = str(i)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f">>> 🚀 任务 [{{i}}] 获取到 GPU-{{gpu_id}}，全力预测中...")
    
    # 拼接完整的执行命令，确保显卡隔离生效
    full_cmd = f"export CUDA_VISIBLE_DEVICES={{gpu_id}} && python stage1_runner.py"
    res = subprocess.run(full_cmd, shell=True, env=env, capture_output=True, text=True)
    
    if res.returncode != 0:
        print(f"!!! 任务 [{{i}}] 发生致命崩溃:\\n{{res.stderr}}")
        
    print(f">>> 🏁 任务 [{{i}}] 完成！释放 GPU-{{gpu_id}}...")
    free_gpus.put(gpu_id)

# 确保所有线程安全执行并暴露错误
futures = []
with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    for i in range({len(models_info)}):
        futures.append(executor.submit(run_model, i))
for f in futures:
    f.result()

print("\\n>>> [HPC Runtime] 所有模型预测完毕，正在合并勘探报告...")
with open('data/stage1_observation.txt', 'w', encoding='utf-8') as fout:
    for obs_file in glob.glob('data/stage1_obs_*.txt'):
        with open(obs_file, 'r', encoding='utf-8') as fin:
            fout.write(fin.read() + '\\n')

print("\\n" + "="*60)
print(">>> [HPC Runtime] 启动阶段 2: 提取数据、算分并生成图表...")
res = subprocess.run('python stage2_eval.py', shell=True, capture_output=True, text=True)
if res.returncode != 0:
    print(f"!!! 算分阶段发生异常:\\n{{res.stderr}}")
    import sys
    sys.exit(1)
"""

    safe_loop_sh_code = f"""#!/bin/bash
#SBATCH -J amp_eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}  # <==== 动态拉取你 .env 里的 12 核
#SBATCH --gres=gpu:{SLURM_GPUS}              # <==== 🚨 核心：严格按照模板，向超算索要真实 GPU！拉取你 .env 里的 3 卡
#SBATCH --mem=40G                            # <==== 保留我们的防 OOM 装甲
#SBATCH -p {SLURM_PARTITION}
#SBATCH -o amp_eval_%j.out
#SBATCH -e amp_eval_%j.err

cd {HPC_TARGET_DIR}
source {CONDA_SH_PATH}
conda activate {VLAB_ENV}

# 🚨 【地毯式清场】防止上一轮残留数据污染
rm -rf data/*_out
rm -f data/stage1_obs_*.txt
rm -f data/stage1_observation.txt
rm -f eval_result.json evaluation_curves.png final_results_with_predictions.csv

python eval_script.py
echo "finish"
"""
    # ---------------- 替换 Phase 4 结束 ----------------

    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        print(f"\n=======================================================")
        print(f"   🚀 正在超算上独立评测数据集: [{ds_name}] ")
        
        res_dir = Path(f"data/results/{ds_name}")
        res_dir.mkdir(parents=True, exist_ok=True)
        
        real_data = run_on_hpc_and_fetch(
            py_code=combined_wrapper_code,
            sh_code=safe_loop_sh_code,  # <--- 我们完美的 4 卡脚本传进去了！
            fetch_targets={
                "eval_result.json": str(res_dir / "eval_result.json"),
                "evaluation_curves.png": str(res_dir / "evaluation_curves.png"),
                "final_results_with_predictions.csv": str(res_dir / "final_results_with_predictions.csv"),
            },
            models_info=models_info,
            local_data_dir=str(ds_dir),
            use_sbatch=True # <--- 改回 True，老老实实去排队拿 4 张卡！
        )
    # ---------------- 替换 Phase 4 结束 ----------------
        
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