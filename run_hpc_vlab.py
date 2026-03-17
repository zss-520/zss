import os
from dotenv import load_dotenv
import sys
import re
import json
import time
from pathlib import Path

import paramiko
import pandas as pd

# ==========================================
# 0. 路径与依赖导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from virtual_lab.agent import Agent
    from virtual_lab.run_meeting import run_meeting
except ModuleNotFoundError:
    print("!!! 警告: 在当前目录找不到 virtual_lab。尝试将上一级目录加入路径...")
    sys.path.append(os.path.dirname(current_dir))
    sys.path.append(os.path.join(os.path.dirname(current_dir), "src"))
    from virtual_lab.agent import Agent
    from virtual_lab.run_meeting import run_meeting


# ==========================================
# 1. 配置区
# ==========================================
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = "qwen3-coder-plus"

HPC_HOST = os.getenv("HPC_HOST")
HPC_PORT = int(os.getenv("HPC_PORT", "22"))
HPC_USER = os.getenv("HPC_USER")
HPC_PASS = os.getenv("HPC_PASS")
HPC_TARGET_DIR = os.getenv("HPC_TARGET_DIR")

CONDA_SH_PATH = os.getenv("CONDA_SH_PATH")
MACREL_ENV = os.getenv("MACREL_ENV")
AMPSCANNER_ENV = os.getenv("AMPSCANNER_ENV")
VLAB_ENV = os.getenv("VLAB_ENV")


def validate_runtime_config():
    required = {
        "OPENAI_API_KEY / DASHSCOPE_API_KEY": DASHSCOPE_API_KEY,
        "HPC_HOST": HPC_HOST,
        "HPC_USER": HPC_USER,
        "HPC_PASS": HPC_PASS,
        "CONDA_SH_PATH": CONDA_SH_PATH,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"缺少必要配置: {', '.join(missing)}")

    print(">>> [Config] 配置检查通过")
    print(f"    MODEL_NAME      = {MODEL_NAME}")
    print(f"    HPC_HOST        = {HPC_HOST}")
    print(f"    HPC_PORT        = {HPC_PORT}")
    print(f"    HPC_USER        = {HPC_USER}")
    print(f"    HPC_TARGET_DIR  = {HPC_TARGET_DIR}")
    print(f"    CONDA_SH_PATH   = {CONDA_SH_PATH}")
    print(f"    MACREL_ENV      = {MACREL_ENV}")
    print(f"    AMPSCANNER_ENV  = {AMPSCANNER_ENV}")
    print(f"    VLAB_ENV        = {VLAB_ENV}")


# ==========================================
# 2. 本地数据自动嗅探与预处理
# ==========================================
def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper().replace(" ", "")
    return seq


def is_valid_peptide_sequence(seq: str) -> bool:
    # 允许常见蛋白字符；如果你有更严格要求，可以进一步缩紧
    return bool(re.fullmatch(r"[A-Z\*\-]+", seq))


def parse_label_value(x):
    label_str = str(x).strip().lower()
    pos_set = {"1", "1.0", "amp", "positive", "true", "pos", "yes"}
    neg_set = {"0", "0.0", "non-amp", "negative", "false", "neg", "no", "decoy", "non"}

    if label_str in pos_set:
        return 1
    if label_str in neg_set:
        return 0

    # 数字兜底
    try:
        return 1 if float(label_str) > 0 else 0
    except Exception:
        return 0


def infer_label_from_filename(filename: str):
    lower_name = filename.lower()
    if "amp" in lower_name and "non" not in lower_name and "decoy" not in lower_name:
        return 1
    if "pos" in lower_name:
        return 1
    if "neg" in lower_name or "decoy" in lower_name or "non" in lower_name:
        return 0
    return None


def auto_prepare_local_data(data_dir="data"):
    print("\n========== [Phase 0] 本地数据自动嗅探与预处理 ==========")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"!!! 未找到 {data_dir} 目录，已自动创建。请将序列文件放入后重试。")
        return False

    all_records = []
    seq_counter = 1

    print(">>> 开始扫描数据目录...")
    for file in os.listdir(data_dir):
        if file in [
            "combined_test.fasta",
            "ground_truth.csv",
            "final_results_with_predictions.csv",
            "evaluation_curves.png",
            "eval_result.json",
        ]:
            continue

        file_path = os.path.join(data_dir, file)
        lower_name = file.lower()
        label = infer_label_from_filename(lower_name)

        print(f"[*] 发现文件: {file} ", end="")

        if file.endswith((".fa", ".fasta", ".txt")):
            if label is None:
                print("-> [警告] 无法通过文件名判断正负样本，已跳过。建议文件名中包含 pos/amp/neg/decoy/non")
                continue

            print(f"-> 判定为: {'🟢 正样本(Label=1)' if label == 1 else '🔴 负样本(Label=0)'}")

            with open(file_path, "r", encoding="utf-8") as f:
                current_id = ""
                current_seq = ""

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith(">"):
                        if current_seq:
                            current_seq = clean_sequence(current_seq)
                            if is_valid_peptide_sequence(current_seq):
                                all_records.append(
                                    {"id": current_id, "sequence": current_seq, "label": label}
                                )
                        current_id = line[1:].split()[0] if line[1:].strip() else f"{'pos' if label == 1 else 'neg'}_seq_{seq_counter}"
                        current_seq = ""
                    else:
                        if not current_id:
                            current_id = f"{'pos' if label == 1 else 'neg'}_seq_{seq_counter}"
                            seq_counter += 1
                        current_seq += line

                if current_seq:
                    current_seq = clean_sequence(current_seq)
                    if is_valid_peptide_sequence(current_seq):
                        all_records.append(
                            {"id": current_id, "sequence": current_seq, "label": label}
                        )

        elif file.endswith(".csv"):
            print("-> 判定为: 📊 混合表格文件")
            try:
                df = pd.read_csv(file_path)
                seq_col = [c for c in df.columns if "seq" in c.lower() or "peptide" in c.lower()]
                label_col = [c for c in df.columns if "label" in c.lower() or "class" in c.lower() or "target" in c.lower()]
                id_col = [c for c in df.columns if "id" in c.lower()]

                if seq_col and label_col:
                    seq_col = seq_col[0]
                    label_col = label_col[0]
                    id_col = id_col[0] if id_col else None

                    for _, row in df.iterrows():
                        seq = clean_sequence(row[seq_col])
                        if not seq or seq.lower() == "nan":
                            continue
                        if not is_valid_peptide_sequence(seq):
                            continue

                        seq_id = str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) else f"csv_seq_{seq_counter}"
                        lab = parse_label_value(row[label_col])

                        all_records.append({
                            "id": seq_id,
                            "sequence": seq,
                            "label": lab
                        })
                        seq_counter += 1
                else:
                    print("  -> [警告] 表格缺少 sequence 或 label 列，跳过。")
            except Exception as e:
                print(f"  -> [错误] 读取 CSV 失败: {e}")

        else:
            print("-> [跳过] 非支持文件类型")

    if not all_records:
        print("!!! 未提取到任何序列数据，流程终止。")
        return False

    df_records = pd.DataFrame(all_records)

    # 去掉空序列
    df_records["sequence"] = df_records["sequence"].astype(str).str.strip()
    df_records = df_records[df_records["sequence"] != ""].copy()

    # 重复 ID 检查
    dup_count = df_records["id"].duplicated().sum()
    if dup_count > 0:
        print(f"!!! [Warning] 检测到 {dup_count} 个重复序列 ID，将自动加后缀避免后续映射错位。")
        counts = {}
        new_ids = []
        for seq_id in df_records["id"]:
            if seq_id not in counts:
                counts[seq_id] = 0
                new_ids.append(seq_id)
            else:
                counts[seq_id] += 1
                new_ids.append(f"{seq_id}__dup{counts[seq_id]}")
        df_records["id"] = new_ids

    fasta_path = os.path.join(data_dir, "combined_test.fasta")
    csv_path = os.path.join(data_dir, "ground_truth.csv")

    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df_records.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")

    df_records[["id", "sequence", "label"]].to_csv(csv_path, index=False)

    print(f"\n[+] 数据预处理与合并完成！共合成 {len(df_records)} 条序列。")
    print(f"    🟢 正样本: {sum(df_records['label'] == 1)} 条")
    print(f"    🔴 负样本: {sum(df_records['label'] == 0)} 条")
    print(f"    📄 已生成模型测试库: {fasta_path}")
    print(f"    📊 已生成真值与合并表格: {csv_path}")
    return True


# ==========================================
# 3. Prompt 区
# ==========================================
TASK_DESC = f"""编写自动化评测脚本，对抗菌肽(AMP)预测模型 Macrel 和 AMP-Scanner-v2 进行评测并记录结果。
具体要求：
1. 【环境隔离要求】：在 Python 代码中调用这两个工具对 data/combined_test.fasta 进行预测时，必须在命令中激活对应的 conda 环境！
   - 运行 Macrel 时，请使用如下语法: `os.system('bash -c "source {CONDA_SH_PATH} && conda activate {MACREL_ENV} && macrel peptides --fasta data/combined_test.fasta --output data/macrel_out"')`
   - 运行 AMP-Scanner 时，请使用如下语法: `os.system('bash -c "source {CONDA_SH_PATH} && conda activate {AMPSCANNER_ENV} && amp_scanner_predict -i data/combined_test.fasta -o data/ampscanner_out.csv"')`
2. 预测完成后，读取 data/ground_truth.csv (包含 id, sequence, label 列)。
3. 解析两个模型的输出文件，根据序列 ID 与 ground_truth.csv 进行对齐映射。
4. 将提取出的预测概率和 0/1 预测结果追加到表格后面，新增列名为: `macrel_prob`, `macrel_pred`, `ampscanner_prob`, `ampscanner_pred`。
5. 将追加了预测结果的完整 DataFrame 保存为 `final_results_with_predictions.csv`。
6. 使用真实的 label 计算 ACC, Recall, MCC, AUROC, AUPRC 这 5 项指标。
7. 绘制合并的 ROC 和 PR 曲线保存为 `evaluation_curves.png`，核心指标 JSON 保存为 `eval_result.json`。"""

PI_PROMPT = """你是一位顶级的计算生物学 PI。当前的评测任务是：【{task_desc}】。
请引导 MLOps 工程师写出评测的 Python 脚本和 Slurm 提交脚本。特别提醒工程师：
1. 绝对不要直接运行工具命令，必须使用 `bash -c "source ... && conda activate ... && command"` 语法跨环境执行。
2. 合并预测结果到 CSV 时要注意 pandas 的 merge 或 map 对齐逻辑，确保最后输出带有 0/1 真实标签和两个模型预测结果的大表。"""

CODER_PROMPT = f"""你是一位精通超算的 MLOps 工程师。根据 PI 的要求编写评测代码。
【强制要求】：
1. 必须保存 `eval_result.json` (指标), `evaluation_curves.png` (图表) 和 `final_results_with_predictions.csv` (附带了预测结果的大表)。
2. 提供 `run_eval.sh` 使用 sbatch。在 Bash 脚本中执行 python 评测脚本前，必须先激活主流程环境：`source {CONDA_SH_PATH} && conda activate {VLAB_ENV}`。
请将 Python 代码严格放在 ```python 块中，Bash 放在 ```bash 块中。"""

CRITIC_PROMPT = """你是一位极其苛刻的独立审稿人。
【真实运行数据】：{real_data}
请根据指标对 Macrel 和 AMP-Scanner-v2 进行对比和无情点评，给出打分（格式：[Score]: X/10）。"""


# ==========================================
# 4. 代码提取与保存
# ==========================================
def extract_code(text):
    if not text:
        return "", ""

    py_codes = re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    sh_codes = re.findall(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)

    py_code = py_codes[-1].strip() if py_codes else ""
    sh_code = sh_codes[-1].strip() if sh_codes else ""
    return py_code, sh_code


def collect_strings_from_json(obj):
    texts = []
    if isinstance(obj, dict):
        for v in obj.values():
            texts.extend(collect_strings_from_json(v))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(collect_strings_from_json(item))
    elif isinstance(obj, str):
        texts.append(obj)
    return texts


def save_generated_code_from_meeting(meeting_history, save_directory, save_name="amp_eval_team"):
    """
    优先从 meeting_history 提取代码；
    如果提取不到，则从 save_dir/save_name.md 或 .json 中兜底读取；
    最后将提取出的 py/sh 保存到本地文件夹。
    """
    os.makedirs(save_directory, exist_ok=True)

    full_dialogue = ""
    if isinstance(meeting_history, str):
        full_dialogue = meeting_history
    elif isinstance(meeting_history, list):
        for msg in meeting_history:
            if isinstance(msg, dict):
                full_dialogue += msg.get("content", "") + "\n"
            elif hasattr(msg, "content"):
                full_dialogue += str(msg.content) + "\n"
            else:
                full_dialogue += str(msg) + "\n"

    py_code, sh_code = extract_code(full_dialogue)

    md_path = os.path.join(save_directory, f"{save_name}.md")
    json_path = os.path.join(save_directory, f"{save_name}.json")

    if (not py_code or not sh_code) and os.path.exists(md_path):
        print(f">>> [Fallback] 尝试从 Markdown 文件提取代码: {md_path}")
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        py_code2, sh_code2 = extract_code(md_text)
        py_code = py_code or py_code2
        sh_code = sh_code or sh_code2

    if (not py_code or not sh_code) and os.path.exists(json_path):
        print(f">>> [Fallback] 尝试从 JSON 文件提取代码: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            json_text = "\n".join(collect_strings_from_json(obj))
            py_code2, sh_code2 = extract_code(json_text)
            py_code = py_code or py_code2
            sh_code = sh_code or sh_code2
        except Exception as e:
            print(f"!!! [Warning] 读取 JSON 失败: {e}")

    py_save_path = os.path.join(save_directory, "generated_eval_script.py")
    sh_save_path = os.path.join(save_directory, "generated_run_eval.sh")

    if py_code:
        with open(py_save_path, "w", encoding="utf-8") as f:
            f.write(py_code + "\n")
        print(f">>> [OK] Python 代码已保存到: {py_save_path}")

    if sh_code:
        with open(sh_save_path, "w", encoding="utf-8") as f:
            f.write(sh_code + "\n")
        print(f">>> [OK] Bash 脚本已保存到: {sh_save_path}")

    return py_code, sh_code, py_save_path, sh_save_path


# ==========================================
# 5. Bash 修补器
# ==========================================
def inject_python_script_path(sh_code, py_path):
    """
    精确替换 run_eval.sh 中真正执行 python 脚本的那一行，
    支持 python / python3 / srun python / srun python3。
    """
    lines = sh_code.splitlines()
    new_lines = []
    replaced = False

    for line in lines:
        if not replaced:
            m = re.match(r"^(\s*(?:srun\s+)?python3?\s+)(\S+)(.*)$", line)
            if m:
                prefix, _, suffix = m.groups()
                new_lines.append(f"{prefix}{py_path}{suffix}")
                replaced = True
                continue
        new_lines.append(line)

    if not replaced:
        new_lines.append(f"python {py_path}")

    return "\n".join(new_lines)


def ensure_cd_to_workdir(sh_code, workdir):
    """
    确保 Slurm 脚本运行时切到目标目录，
    避免 Python 中的相对路径 data/... 失效。
    """
    lines = sh_code.splitlines()
    insert_idx = 0

    if lines and lines[0].startswith("#!"):
        insert_idx = 1

    while insert_idx < len(lines) and lines[insert_idx].lstrip().startswith("#SBATCH"):
        insert_idx += 1

    # 若已经有 cd 到该目录，就不重复插入
    for line in lines:
        if line.strip() == f"cd {workdir}":
            return sh_code

    lines.insert(insert_idx, f"cd {workdir}")
    return "\n".join(lines)


def append_safe_slurm_defaults(sh_code):
    """
    若 AI 没写日志参数，补充基础日志输出。
    """
    lines = sh_code.splitlines()
    has_output = any("#SBATCH" in line and "--output" in line for line in lines)
    has_error = any("#SBATCH" in line and "--error" in line for line in lines)

    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1

    while insert_idx < len(lines) and lines[insert_idx].lstrip().startswith("#SBATCH"):
        insert_idx += 1

    add_lines = []
    if not has_output:
        add_lines.append("#SBATCH --output=slurm-%j.out")
    if not has_error:
        add_lines.append("#SBATCH --error=slurm-%j.err")

    if add_lines:
        lines[insert_idx:insert_idx] = add_lines

    return "\n".join(lines)


# ==========================================
# 6. SSH 跨端执行器
# ==========================================
def read_remote_text(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    return out, err


def run_on_hpc_and_fetch(py_code, sh_code):
    print("\n>>> [SSH] 正在连接物理超算节点...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None

    try:
        ssh.connect(HPC_HOST, HPC_PORT, HPC_USER, HPC_PASS)
        sftp = ssh.open_sftp()

        # 创建远程目录
        read_remote_text(ssh, f"mkdir -p {HPC_TARGET_DIR}/data")

        # 上传数据
        print(">>> [SSH] 正在将统一处理好的数据推送至超算...")
        sftp.put("data/combined_test.fasta", f"{HPC_TARGET_DIR}/data/combined_test.fasta")
        sftp.put("data/ground_truth.csv", f"{HPC_TARGET_DIR}/data/ground_truth.csv")

        # 写入生成代码
        py_path = f"{HPC_TARGET_DIR}/eval_script.py"
        sh_path = f"{HPC_TARGET_DIR}/run_eval.sh"

        with sftp.file(py_path, "w") as f:
            f.write(py_code)

        sh_code = inject_python_script_path(sh_code, py_path)
        sh_code = ensure_cd_to_workdir(sh_code, HPC_TARGET_DIR)
        sh_code = append_safe_slurm_defaults(sh_code)

        with sftp.file(sh_path, "w") as f:
            f.write(sh_code)

        read_remote_text(ssh, f"chmod +x {sh_path}")

        print(">>> [SSH] AI 代码推流完毕，提交任务...")
        submit_out, submit_err = read_remote_text(ssh, f"cd {HPC_TARGET_DIR} && sbatch run_eval.sh")

        if "Submitted batch job" not in submit_out:
            print("!!! [Error] Slurm任务提交失败")
            print("--- stdout ---")
            print(submit_out)
            print("--- stderr ---")
            print(submit_err)
            return None

        match = re.search(r"Submitted batch job (\d+)", submit_out)
        if not match:
            print("!!! [Error] 无法从 sbatch 输出中解析 job id")
            print(submit_out)
            return None

        job_id = match.group(1)

        print(f">>> [Slurm] 等待计算节点完成 (Job ID: {job_id})", end="")
        while True:
            sq_out, _ = read_remote_text(ssh, f"squeue -j {job_id}")
            if job_id not in sq_out:
                print("\n>>> [Slurm] 任务完毕。")
                break
            print(".", end="", flush=True)
            time.sleep(15)

        print(">>> [SSH] 正在回收评测数据、图表和包含预测结果的总表...")
        real_data = None

        try:
            sftp.get(f"{HPC_TARGET_DIR}/eval_result.json", "./eval_result.json")
            with open("./eval_result.json", "r", encoding="utf-8") as f:
                real_data = json.load(f)
        except Exception as e:
            print(f"!!! [Error] 无法获取 eval_result.json: {e}")

        try:
            sftp.get(f"{HPC_TARGET_DIR}/evaluation_curves.png", "data/evaluation_curves.png")
            print(">>> [SSH] 📊 评测曲线图已保存至 data/evaluation_curves.png")
        except Exception as e:
            print(f">>> [Warn] 未取回 evaluation_curves.png: {e}")

        try:
            sftp.get(f"{HPC_TARGET_DIR}/final_results_with_predictions.csv", "data/final_results_with_predictions.csv")
            print(">>> [SSH] 📋 包含预测结果的最终合并表已保存至 data/final_results_with_predictions.csv")
        except Exception as e:
            print(f">>> [Warn] 未取回 final_results_with_predictions.csv: {e}")

        if real_data is None:
            print("!!! [Error] 核心结果 JSON 缺失，尝试读取远程日志排错")
            slurm_out, _ = read_remote_text(ssh, f"cat {HPC_TARGET_DIR}/slurm-{job_id}.out || true")
            slurm_err, _ = read_remote_text(ssh, f"cat {HPC_TARGET_DIR}/slurm-{job_id}.err || true")

            print("--- HPC slurm stdout (前 2000 字) ---")
            print(slurm_out[:2000] if slurm_out else "[空]")
            print("--- HPC slurm stderr (前 2000 字) ---")
            print(slurm_err[:2000] if slurm_err else "[空]")
            return None

        return real_data

    except Exception as e:
        print(f"!!! [Error] run_on_hpc_and_fetch 失败: {e}")
        return None

    finally:
        if sftp:
            try:
                sftp.close()
            except Exception:
                pass
        try:
            ssh.close()
        except Exception:
            pass


# ==========================================
# 7. 主流程
# ==========================================
def main():
    validate_runtime_config()

    if not auto_prepare_local_data("data"):
        return

    print("\n========== [Phase 1] 虚拟实验室闭门会议 (Team 模式) ==========")

    pi_agent = Agent(
        model=MODEL_NAME,
        title="PI",
        expertise="计算生物学",
        goal=PI_PROMPT.format(task_desc=TASK_DESC),
        role="计算生物学 PI",
    )

    coder_agent = Agent(
        model=MODEL_NAME,
        title="MLOps Coder",
        expertise="超算与代码编写",
        goal=CODER_PROMPT,
        role="超算 MLOps 工程师",
    )

    print(">>> 正在生成代码，PI 和 Coder 正在云端激烈讨论...")

    save_directory = Path("data/vlab_discussions")
    save_directory.mkdir(parents=True, exist_ok=True)

    meeting_history = run_meeting(
        meeting_type="team",
        agenda=f"请为任务【{TASK_DESC}】编写评测代码。",
        save_dir=save_directory,
        save_name="amp_eval_team",
        team_lead=pi_agent,
        team_members=[coder_agent],
        num_rounds=3,
    )

    py_code, sh_code, py_local_path, sh_local_path = save_generated_code_from_meeting(
        meeting_history=meeting_history,
        save_directory=str(save_directory),
        save_name="amp_eval_team",
    )

    if not py_code or not sh_code:
        print("!!! [Fatal] 未能提取到完整的 Python/Bash 代码。")
        print(f"    Python提取状态: {'成功' if py_code else '失败'}")
        print(f"    Bash提取状态: {'成功' if sh_code else '失败'}")
        print(f"    请检查会议记录文件: {os.path.join(str(save_directory), 'amp_eval_team.md')}")
        return

    print(f">>> [Local] Python脚本已保存: {py_local_path}")
    print(f">>> [Local] Bash脚本已保存: {sh_local_path}")

    print("\n========== [Phase 2] 打破虚拟壁垒，向超算进发 ==========")
    real_data = run_on_hpc_and_fetch(py_code, sh_code)
    if not real_data:
        return

    print("\n========== [Phase 3] 科学审判 (Individual 模式) ==========")

    critic_agent = Agent(
        model=MODEL_NAME,
        title="Critic",
        expertise="模型评测与结果分析",
        goal=CRITIC_PROMPT.format(real_data=json.dumps(real_data, indent=2, ensure_ascii=False)),
        role="严苛的独立审稿人",
    )

    print(">>> 正在将超算跑出的真实数据提交给审稿人...\n")

    critic_history = run_meeting(
        meeting_type="individual",
        agenda="这是我们最终在超算跑出的真实数据，请根据数据给出你的评价与打分。",
        save_dir=save_directory,
        save_name="critic_individual",
        team_lead=critic_agent,
        team_members=[],
        num_rounds=1,
    )

    critic_response = ""
    if isinstance(critic_history, str):
        critic_response = critic_history
    elif isinstance(critic_history, list):
        for msg in critic_history:
            if isinstance(msg, dict):
                critic_response += msg.get("content", "") + "\n"
            elif hasattr(msg, "content"):
                critic_response += str(msg.content) + "\n"
            else:
                critic_response += str(msg) + "\n"

    print("✨ >>> [审稿人 (Critic) 最终判决] <<< ✨\n" + critic_response)


if __name__ == "__main__":
    main()