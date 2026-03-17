import ast
import json
import os
import re
import time
from pathlib import Path

import paramiko

from config import (
    HPC_HOST,
    HPC_PORT,
    HPC_USER,
    HPC_PASS,
    HPC_TARGET_DIR,
    CONDA_SH_PATH,
    VLAB_ENV,
    SLURM_PARTITION,
    SLURM_JOB_NAME,
    SLURM_GPUS,
    SLURM_CPUS_PER_TASK,
    PIP_INDEX_URL,
    PIP_EXTRA_INDEX_URL,
)

IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "Bio": "biopython",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "psutil": "psutil",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "networkx": "networkx",
    "requests": "requests",
    "tqdm": "tqdm",
    "joblib": "joblib",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "sentence_transformers": "sentence-transformers",
    "plotly": "plotly",
    "statsmodels": "statsmodels",
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
}


def _score_python_block(code: str, stage: str = "generic") -> int:
    """
    stage-aware 打分：
    - stage1: 强烈偏向生成 model_outputs_for_eval.csv / model_outputs_summary.json 的脚本
    - stage2: 强烈偏向生成 eval_result.json / evaluation_curves.png / final_results_with_predictions.csv 的脚本
    """
    score = 0
    text = code.strip()
    if not text:
        return -(10**9)

    # 基础语法检查
    try:
        tree = ast.parse(text)
        score += 20
    except Exception:
        return -(10**6)

    # 完整脚本特征
    if re.search(r"^\s*import\s+\w+", text, re.MULTILINE):
        score += 30
    if re.search(r"^\s*from\s+\w+(\.\w+)*\s+import\s+", text, re.MULTILINE):
        score += 30
    if "def main(" in text:
        score += 60
    if 'if __name__ == "__main__":' in text or "if __name__ == '__main__':" in text:
        score += 80

    func_count = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    score += min(func_count * 15, 120)

    # 通用关键词
    common_keywords = [
        "ground_truth.csv",
        "macrel",
        "amp_scanner",
        "ampscanner",
        "pandas",
        "numpy",
        "subprocess",
        "shutil",
    ]
    for kw in common_keywords:
        if kw in text:
            score += 8

    # stage1 专用偏置
    if stage == "stage1":
        if "model_outputs_for_eval.csv" in text:
            score += 140
        if "model_outputs_summary.json" in text:
            score += 140
        if "macrel_prob" in text:
            score += 40
        if "ampscanner_prob" in text:
            score += 40
        if "macrel_pred" in text:
            score += 30
        if "ampscanner_pred" in text:
            score += 30

        if "eval_result.json" in text:
            score -= 220
        if "evaluation_curves.png" in text:
            score -= 180
        if "final_results_with_predictions.csv" in text:
            score -= 180
        if "roc_curve" in text or "precision_recall_curve" in text:
            score -= 140
        if "accuracy_score" in text or "roc_auc_score" in text or "average_precision_score" in text:
            score -= 140

    # stage2 专用偏置
    elif stage == "stage2":
        if "eval_result.json" in text:
            score += 140
        if "evaluation_curves.png" in text:
            score += 140
        if "final_results_with_predictions.csv" in text:
            score += 140
        if "roc_curve" in text or "precision_recall_curve" in text:
            score += 50
        if "accuracy_score" in text or "roc_auc_score" in text or "average_precision_score" in text:
            score += 50

        if "model_outputs_for_eval.csv" in text:
            score += 60
        if "model_outputs_summary.json" in text:
            score += 30

    else:
        task_keywords = [
            "eval_result.json",
            "evaluation_curves.png",
            "final_results_with_predictions.csv",
            "model_outputs_for_eval.csv",
            "model_outputs_summary.json",
            "matplotlib",
            "sklearn",
        ]
        for kw in task_keywords:
            if kw in text:
                score += 10

    # 额外惩罚：可疑中文标点污染
    if "，，" in text or "。。" in text or "；；" in text:
        score -= 120
    if "return " in text and "，" in text:
        score -= 200

    # 太短的片段扣分
    if len(text) < 300:
        score -= 80

    has_import = re.search(r"^\s*(import|from)\s+", text, re.MULTILINE) is not None
    has_main = "def main(" in text
    if not has_import:
        score -= 100
    if not has_main:
        score -= 60

    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if first_line.startswith("#"):
        score -= 10

    return score


def _select_best_python_block(py_codes: list[str], stage: str = "generic") -> str:
    if not py_codes:
        return ""

    scored = [(code, _score_python_block(code, stage=stage)) for code in py_codes]
    scored.sort(key=lambda x: x[1], reverse=True)

    print(">>> [Extract] Python 代码块评分（从高到低）:")
    for i, (_, s) in enumerate(scored[:5], 1):
        print(f"    #{i}: score={s}")

    return scored[0][0].strip()


def extract_code(text: str, stage: str = "generic") -> tuple[str, str]:
    if not text:
        return "", ""

    py_codes = re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    sh_codes = re.findall(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)

    py_code = _select_best_python_block(py_codes, stage=stage)
    sh_code = max(sh_codes, key=len).strip() if sh_codes else ""
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


def _meeting_to_text(meeting_history) -> str:
    full_dialogue = ""
    if isinstance(meeting_history, str):
        full_dialogue = meeting_history
    elif isinstance(meeting_history, list):
        for msg in meeting_history:
            if isinstance(msg, dict):
                full_dialogue += msg.get("content", "") or msg.get("message", "")
                full_dialogue += "\n"
            elif hasattr(msg, "content"):
                full_dialogue += str(msg.content) + "\n"
            else:
                full_dialogue += str(msg) + "\n"
    return full_dialogue


def save_generated_code_from_meeting(
    meeting_history,
    save_directory,
    py_filename: str = "generated_eval_script.py",
    sh_filename: str = "generated_run_eval.sh",
    meeting_dump_name: str = "meeting_dump",
):
    os.makedirs(save_directory, exist_ok=True)
    full_dialogue = _meeting_to_text(meeting_history)

    stage = "generic"
    dump_lower = meeting_dump_name.lower()
    if "stage1" in dump_lower or "first" in dump_lower:
        stage = "stage1"
    elif "stage2" in dump_lower or "second" in dump_lower:
        stage = "stage2"

    py_code, sh_code = extract_code(full_dialogue, stage=stage)

    md_path = os.path.join(save_directory, f"{meeting_dump_name}.md")
    json_path = os.path.join(save_directory, f"{meeting_dump_name}.json")

    if (not py_code or not sh_code) and os.path.exists(md_path):
        print(f">>> [Fallback] 尝试从 Markdown 文件提取代码: {md_path}")
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        py_code2, sh_code2 = extract_code(md_text, stage=stage)
        py_code = py_code or py_code2
        sh_code = sh_code or sh_code2

    if (not py_code or not sh_code) and os.path.exists(json_path):
        print(f">>> [Fallback] 尝试从 JSON 文件提取代码: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            json_text = "\n".join(collect_strings_from_json(obj))
            py_code2, sh_code2 = extract_code(json_text, stage=stage)
            py_code = py_code or py_code2
            sh_code = sh_code or sh_code2
        except Exception as e:
            print(f"!!! [Warning] 读取 JSON 失败: {e}")

    py_save_path = os.path.join(save_directory, py_filename)
    sh_save_path = os.path.join(save_directory, sh_filename)

    if py_code:
        with open(py_save_path, "w", encoding="utf-8") as f:
            f.write(py_code + "\n")
        print(f">>> [OK] Python 代码已保存到: {py_save_path}")

    if sh_code:
        with open(sh_save_path, "w", encoding="utf-8") as f:
            f.write(sh_code + "\n")
        print(f">>> [OK] Bash 脚本已保存到: {sh_save_path}")

    return py_code, sh_code, py_save_path, sh_save_path


def build_stage2_context_from_stage1_outputs(output_dir: str | Path, **kwargs) -> str:
    obs_path = Path(output_dir) / "stage1_observation.txt"
    if obs_path.exists():
        try:
            with open(obs_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"[读取 stage1_observation.txt 失败] {e}"
    return "[警告：未找到第一阶段的勘探报告 stage1_observation.txt，这通常意味着第一阶段运行失败]"


def read_remote_text(ssh, cmd: str, stream: bool = False) -> tuple[str, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    if stream:
        # 如果开启了推流模式，就一行一行实时打印超算的输出
        out_lines = []
        for line in iter(stdout.readline, ""):
            print(f"      [HPC] {line.strip()}")
            out_lines.append(line)
        out = "".join(out_lines)
        err = stderr.read().decode("utf-8", errors="ignore")
    else:
        out = stdout.read().decode("utf-8", errors="ignore")
        err = stderr.read().decode("utf-8", errors="ignore")
        
    return out, err


def _stdlib_modules() -> set[str]:
    stdlib = set()
    try:
        import sys
        stdlib = set(sys.stdlib_module_names)
    except Exception:
        pass

    stdlib.update(
        {
            "os", "sys", "re", "json", "math", "time", "datetime", "pathlib",
            "logging", "hashlib", "subprocess", "typing", "itertools", "functools",
            "collections", "random", "statistics", "csv", "glob", "shutil",
            "tempfile", "argparse", "traceback", "pickle", "gzip", "zipfile",
            "tarfile", "dataclasses", "inspect", "threading", "multiprocessing",
            "concurrent", "unittest", "doctest", "fractions", "decimal",
            "urllib", "http", "xml", "sqlite3", "base64",
        }
    )
    return stdlib


def infer_requirements_from_py_code(py_code: str) -> tuple[list[str], list[str]]:
    stdlib = _stdlib_modules()
    try:
        tree = ast.parse(py_code)
    except SyntaxError:
        return [], []

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported_roots.add(node.module.split(".")[0])

    filtered = sorted(
        {
            mod for mod in imported_roots
            if mod not in stdlib and not mod.startswith("_")
        }
    )
    pip_packages = sorted({IMPORT_TO_PIP.get(mod, mod) for mod in filtered})
    return filtered, pip_packages


def ensure_remote_eval_dependencies(ssh, py_code: str) -> None:
    imported_modules, pip_packages = infer_requirements_from_py_code(py_code)
    print(f">>> [Env] 从生成脚本中识别到第三方模块: {imported_modules}")
    print(f">>> [Env] 需要检查/安装的 pip 包: {pip_packages}")

    if not pip_packages:
        print(">>> [Env] 未识别到第三方 pip 依赖，跳过 login 节点安装。")
        return

    packages_json = json.dumps(pip_packages, ensure_ascii=False)
    import_name_map_json = json.dumps(
        {
            "scikit-learn": "sklearn",
            "opencv-python": "cv2",
            "Pillow": "PIL",
            "PyYAML": "yaml",
            "biopython": "Bio",
            "beautifulsoup4": "bs4",
            "pycryptodome": "Crypto",
            "sentence-transformers": "sentence_transformers",
        },
        ensure_ascii=False,
    )

    cmd = f'''cd {HPC_TARGET_DIR}
source {CONDA_SH_PATH}
conda activate {VLAB_ENV}

python - <<'PY'
import importlib
import json
import subprocess
import sys

packages = json.loads(r"""{packages_json}""")
import_name_map = json.loads(r"""{import_name_map_json}""")

missing = []
for pkg in packages:
    mod = import_name_map.get(pkg, pkg.replace('-', '_'))
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)

if missing:
    print('>>> Missing packages on login node:', ', '.join(missing))
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        '-i', r"{PIP_INDEX_URL}",
        '--extra-index-url', r"{PIP_EXTRA_INDEX_URL}",
        *missing
    ]
    print('>>> Running:', ' '.join(cmd))
    subprocess.check_call(cmd)
else:
    print('>>> All required packages already installed on login node.')
PY
'''
    out, err = read_remote_text(ssh, cmd)
    print(">>> [Env] login 节点依赖检查/安装输出：")
    print(out if out.strip() else "[stdout 空]")
    if err.strip():
        print(">>> [Env] login 节点依赖检查/安装 stderr：")
        print(err)
    if "CalledProcessError" in err or "ERROR:" in err or "Traceback" in err:
        raise RuntimeError("login 节点自动安装依赖失败")

# ==============================================================
# 新增模块：根据记忆库动态创建模型运行环境 (支持 Git 与 Zenodo 自动解析)
# ==============================================================
def setup_model_environments(ssh, models_info: list[dict]) -> None:
    if not models_info:
        return

    print("\n>>> [Auto-Env] 开始为提取到的模型动态构建专属 Conda 环境和代码库...")
    for model in models_info:
        env_name = model.get("env_name")
        if not env_name:
            continue

        repo_url = model.get("repo_url")
        # 容错处理：确保 None 变成空字符串，防止 bash 报错
        repo_url_str = repo_url if repo_url else ""
        python_version = model.get("python_version") or "3.9"
        dependencies = model.get("dependencies") or []

        pip_install_cmd = ""
        if dependencies:
            deps_str = " ".join(dependencies)
            pip_install_cmd = f"pip install -i {PIP_INDEX_URL} --extra-index-url {PIP_EXTRA_INDEX_URL} {deps_str}"

        # 编写 Bash 脚本：内嵌 Python 脚本处理 Zenodo API（带进度条）
        setup_script = f"""
        cd {HPC_TARGET_DIR}
        source {CONDA_SH_PATH}

        # 1. 动态获取源码 (智能区分 Git 和 Zenodo)
        REPO_URL="{repo_url_str}"
        if [ -n "$REPO_URL" ] && [ "$REPO_URL" != "null" ] && [ "$REPO_URL" != "None" ]; then
            if [[ "$REPO_URL" == *"zenodo.org/records/"* ]]; then
                echo ">>> [Zenodo] 识别到 Zenodo 链接，启动 API 自动下载与解压..."
                python - << 'EOF'
import urllib.request
import json
import os
import zipfile
import tarfile
import sys

# 直接利用外层 Python f-string 注入变量，完美避开 Bash 的解析陷阱
url = "{repo_url_str}"
record_id = url.rstrip('/').split('/')[-1]
api_url = f"https://zenodo.org/api/records/{{record_id}}"

def download_progress(block_num, block_size, total_size):
    # 显示下载进度条 (千万不要在这里用三引号，会截断外层的 f-string！)
    downloaded = block_num * block_size
    if total_size > 0:
        percent = int(downloaded * 100 / total_size)
        # 每 10% 打印一次进度，避免刷屏
        if percent % 10 == 0 and downloaded > 0:
            sys.stdout.write(f"\\r      -> 已下载 {{percent}}% ({{downloaded/1024/1024:.1f}} MB / {{total_size/1024/1024:.1f}} MB)")
            sys.stdout.flush()

try:
    print(f">>> [Zenodo] 正在请求 API: {{api_url}}")
    req = urllib.request.Request(api_url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    for f in data.get('files', []):
        download_url = f['links']['self']
        filename = f['key']
        if not os.path.exists(filename):
            print(f">>> [Zenodo] 开始下载 {{filename}} ...")
            urllib.request.urlretrieve(download_url, filename, reporthook=download_progress)
            print(f"\\n>>> [Zenodo] {{filename}} 下载完成！")
            
            # 自动解压
            if filename.endswith('.zip'):
                print(f">>> [Zenodo] 正在解压 {{filename}} ...")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall()
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                print(f">>> [Zenodo] 正在解压 {{filename}} ...")
                with tarfile.open(filename, 'r:gz') as tar_ref:
                    tar_ref.extractall()
        else:
            print(f">>> [Zenodo] 文件 {{filename}} 已存在，跳过下载。")
except Exception as e:
    print(f"!!! [Zenodo] 下载或解压失败: {{e}}")
EOF
            else:
                # 默认走 git clone
                repo_name=$(basename "$REPO_URL" .git)
                if [ ! -d "$repo_name" ]; then
                    echo ">>> [Git] Cloning $REPO_URL..."
                    git clone "$REPO_URL"
                else
                    echo ">>> [Git] Repo $repo_name already exists, skipping clone."
                fi
            fi
        fi

        # 2. 智能环境路由与创建 (动态区分 PT1, PT2, TF)
        if ! conda info --envs | grep -q "^[[:space:]]*{env_name}[[:space:]]"; then
            echo ">>> [Conda] 开始为 {env_name} 构建虚拟环境..."
            
            # 探测依赖中是否包含特定的框架要求
            if [[ "{deps_str}" == *"tensorflow"* ]]; then
                echo ">>> [Conda] 嗅探到 TensorFlow，正在克隆 base_tf 基座..."
                conda create -n {env_name} --clone base_tf -y
            elif [[ "{deps_str}" == *"torch==1."* ]]; then
                echo ">>> [Conda] 嗅探到旧版 PyTorch，正在克隆 base_pt1 基座..."
                conda create -n {env_name} --clone base_pt1 -y
            elif [[ "{deps_str}" == *"torch"* ]]; then
                echo ">>> [Conda] 嗅探到现代 PyTorch，正在克隆 base_pt2 基座..."
                conda create -n {env_name} --clone base_pt2 -y
            else
                echo ">>> [Conda] 未嗅探到巨无霸框架，创建轻量级纯净环境..."
                conda create -n {env_name} python={python_version} -y
            fi
        else
            echo ">>> [Conda] Environment '{env_name}' already exists, skipping creation."
        fi

        # 3. 覆盖安装专属依赖与版本微调
        # 这里哪怕基座里已经有了 numpy 1.24，只要文献要求 numpy==1.21
        # pip 就会瞬间把 1.24 换成 1.21，完美解决版本不一致问题！
        if [ -n "{pip_install_cmd}" ]; then
            echo ">>> [Pip] 正在微调依赖版本并安装特有包..."
            conda activate {env_name}
            {pip_install_cmd} || true
        fi

        # 3. 智能安装依赖 (优先使用仓库自带的配置文件)
        conda activate {env_name}
        
        # 记录当前在哪个目录 (通常是刚 clone 下来的仓库目录)
        if [ -n "$REPO_URL" ] && [ "$REPO_URL" != "null" ]; then
            repo_name=$(basename "$REPO_URL" .git)
            if [ -d "$repo_name" ]; then
                cd "$repo_name"
            fi
        fi

        echo ">>> [Pip] 正在侦测仓库原生依赖配置..."
        if [ -f "environment.yml" ]; then
            echo ">>> [Conda] 发现官方 environment.yml，正在基于官方配置更新环境..."
            conda env update -n {env_name} -f environment.yml
        elif [ -f "requirements.txt" ]; then
            echo ">>> [Pip] 发现官方 requirements.txt，正在优先安装官方依赖..."
            pip install -i {PIP_INDEX_URL} --extra-index-url {PIP_EXTRA_INDEX_URL} -r requirements.txt
            
            # 官方包装完后，补一下 AI 认为可能缺失的包 (以防万一)
            if [ -n "{pip_install_cmd}" ]; then
                echo ">>> [Pip] 补充安装 AI 提取的关键依赖..."
                {pip_install_cmd} || true  # 加上 || true 防止部分包冲突导致整个流程崩溃
            fi
        else
            echo ">>> [Pip] 未发现官方配置文件，完全依赖 AI 提取的包..."
            if [ -n "{pip_install_cmd}" ]; then
                {pip_install_cmd} || true
            else
                echo ">>> [Pip] 无需安装额外依赖。"
            fi
        fi
        
        # 退回到工作根目录
        cd {HPC_TARGET_DIR}
        """
        
        # 【关键修改】：开启 stream=True，实时查看超算干了啥！
        out, err = read_remote_text(ssh, setup_script, stream=True)
        
        print(f"    -> 模型 [{model['model_name']}] 环境就绪！")
        if "fatal:" in err or "CondaHTTPError" in err:
            print(f"    !!! [警告] 构建时出现潜在错误日志: \n{err}")

def build_standard_run_eval_sh(py_path: str, py_code: str) -> str:
    imported_modules, _ = infer_requirements_from_py_code(py_code)
    modules_json = json.dumps(imported_modules, ensure_ascii=False)

    return f'''#!/bin/bash
#SBATCH -J {SLURM_JOB_NAME}
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --gres=gpu:{SLURM_GPUS}
#SBATCH -p {SLURM_PARTITION}
#SBATCH -o {HPC_TARGET_DIR}/{SLURM_JOB_NAME}.%j.out
#SBATCH -e {HPC_TARGET_DIR}/{SLURM_JOB_NAME}.%j.err

set -e

cd {HPC_TARGET_DIR}
source {CONDA_SH_PATH}
conda activate {VLAB_ENV}

export MPLBACKEND=Agg

echo ">>> Python path: $(which python)"
python --version
pip --version || true

python - <<'PY'
import importlib
import json
import sys

modules = json.loads(r"""{modules_json}""")
missing = []
for mod in modules:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    print('>>> Missing packages in compute node environment:', ', '.join(missing))
    print('>>> Please install them on login node before job submission.')
    sys.exit(1)
else:
    print('>>> All required packages already installed.')
PY

python {py_path}

echo "finish"
'''


def run_on_hpc_and_fetch(
    py_code: str,
    sh_code: str,
    fetch_targets: dict[str, str] | None = None,
    models_info: list[dict] = None,  # <--- 新增这行
):
    print("\n>>> [SSH] 正在连接物理超算节点...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None
    fetch_targets = fetch_targets or {
        "eval_result.json": "./eval_result.json",
        "data/evaluation_curves.png": "data/evaluation_curves.png",
        "data/final_results_with_predictions.csv": "data/final_results_with_predictions.csv",
    }

    try:
        ssh.connect(HPC_HOST, HPC_PORT, HPC_USER, HPC_PASS)
        sftp = ssh.open_sftp()
        
        # ====== 新增：核弹级清理！防止幽灵文件干扰 ======
        read_remote_text(ssh, f"rm -rf {HPC_TARGET_DIR}/data/*")
        # ============================================
        
        read_remote_text(ssh, f"mkdir -p {HPC_TARGET_DIR}/data")

        print(">>> [SSH] 正在将统一处理好的数据推送至超算...")
        sftp.put("data/combined_test.fasta", f"{HPC_TARGET_DIR}/data/combined_test.fasta")
        sftp.put("data/ground_truth.csv", f"{HPC_TARGET_DIR}/data/ground_truth.csv")
        # ====== 核心新增：自动根据记忆库组装底层环境 ======
        if models_info:
            setup_model_environments(ssh, models_info)
        # ==================================================

        py_path = f"{HPC_TARGET_DIR}/eval_script.py"
        sh_path = f"{HPC_TARGET_DIR}/run_eval.sh"

        with sftp.file(py_path, "w") as f:
            f.write(py_code)

        print(">>> [Slurm] 使用固定模板生成 run_eval.sh")
        sh_code = build_standard_run_eval_sh(py_path, py_code)
        with sftp.file(sh_path, "w") as f:
            f.write(sh_code)

        read_remote_text(ssh, f"chmod +x {sh_path}")
        print(">>> [Env] 正在 login 节点检查并自动补齐评测依赖...")
        ensure_remote_eval_dependencies(ssh, py_code)

        print(">>> [SSH] AI 代码推流完毕，提交任务...")
        submit_out, submit_err = read_remote_text(ssh, f"cd {HPC_TARGET_DIR} && sbatch run_eval.sh")
        if "Submitted batch job" not in submit_out:
            print("!!! [Error] Slurm任务提交失败")
            print(submit_out)
            print(submit_err)
            return None

        match = re.search(r"Submitted batch job (\d+)", submit_out)
        if not match:
            print("!!! [Error] 无法从 sbatch 输出中解析 job id")
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

        fetched = {}
        for remote_name, local_path in fetch_targets.items():
            try_paths = [
                f"{HPC_TARGET_DIR}/{remote_name}",
                f"{HPC_TARGET_DIR}/data/{Path(remote_name).name}",
            ]
            got_it = False

            for remote_path in try_paths:
                try:
                    local_parent = Path(local_path).parent
                    local_parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote_path, local_path)
                    fetched[remote_name] = local_path
                    print(f">>> [SSH] 已取回 {remote_path} -> {local_path}")
                    got_it = True
                    break
                except Exception:
                    pass

            if not got_it:
                print(f">>> [Warn] 未取回 {remote_name}: 在根目录和 data/ 目录都未找到")

        if "eval_result.json" in fetched:
            try:
                with open(fetched["eval_result.json"], "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return fetched or None

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