# -*- coding: utf-8 -*-
"""
统一 HPC 模型部署器 v6.2.4。

新增能力：
1. 优先读取本地 data/models/{model}/README.md + requirements.txt + environment.yml，作为创建环境、下载权重、推断运行命令的依据。
2. requirements.txt 会先生成 requirements.hpc_auto_filtered.txt，自动跳过当前 Python 版本不可能安装的老包，例如 Python>=3.8 下的 tensorflow==1.14.0。
3. mini FASTA smoke test 失败后，读取 smoke log + README + requirements，调用 Agent 生成自愈计划；无 Agent 时也会用常见错误启发式修复，例如 No module named torch。
4. 自愈计划可安全执行 pip/conda 安装、README 下载命令、修改 inference_cmd_template，并自动重试 smoke test。
5. 只有 smoke_test_ok=true 才会把 data/local_registry.json 的 skip_env_setup 写为 true。
6. v6.2.2：过滤 Agent 误加的 TensorFlow 依赖、替换 NCBI BLAST FTP 下载、环境安装失败也进入自愈。
7. v6.2.4：过滤冲突的 CUDA 后缀 torch 依赖、避免 PyG 源码包编译冲突，并给 README conda install 自动加 -y。
"""
from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_md_loader import AgentMDLoader

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from config import (
    HPC_TARGET_DIR,
    CONDA_SH_PATH,
    PIP_INDEX_URL,
    PIP_EXTRA_INDEX_URL,
    DEFAULT_PYTHON_VERSION,
    HPC_REMOTE_REPO_ROOT,
    HPC_BASE_ENV_MAP,
)

ROOT = Path(__file__).resolve().parent
LOCAL_REGISTRY = ROOT / "data" / "local_registry.json"
SELF_HEAL_DIR = ROOT / "data" / "hpc_self_heal"
SELF_HEAL_PROMPT_DIR = ROOT / "agents" / "model_execution"
_SELF_HEAL_PROMPTS = AgentMDLoader(SELF_HEAL_PROMPT_DIR)

PIP_IMPORT_MAP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "Bio": "biopython",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "networkx": "networkx",
    "rdflib": "rdflib",
    "tensorboardX": "tensorboardX",
    "requests": "requests",
    "joblib": "joblib",
}

STDLIB_DEPENDENCIES = {
    "argparse", "collections", "csv", "datetime", "functools", "glob",
    "hashlib", "json", "logging", "math", "os", "pathlib", "pickle",
    "random", "re", "shlex", "shutil", "subprocess", "sys", "tempfile",
    "time", "typing", "urllib", "xml",
}
NORMALIZED_IMPORT_TO_PIP = {str(key).lower(): value for key, value in PIP_IMPORT_MAP.items()}


TF_RELATED_PACKAGES = {"tensorflow", "tensorflow-gpu", "tensorflow-estimator"}

TORCH_PRIMARY_PACKAGES = {"torch", "torchvision", "torchaudio"}
TORCH_GEOMETRIC_STACK = {
    "torch-geometric", "torch-geometric-temporal",
    "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv",
}


def _is_torch_cuda_local_spec(dep: str) -> bool:
    """True for pip specs like torch==1.11.0+cu113 that many indexes cannot resolve."""
    return bool(re.search(r"^(torch|torchvision|torchaudio)\s*==\s*[^\s;]+\+cu\d+", str(dep or "").strip().lower()))


def _genericize_torch_geometric(dep: str) -> str:
    """Avoid fragile old source-pinned PyG packages during generic onboarding.

    torch-geometric itself is usually installable as a Python wheel. Low-level
    extension packages often require version-specific wheels from
    data.pyg.org or local compilation, so they are handled by smoke/self-heal
    only when truly needed.
    """
    name = _dep_name(dep)
    if name == "torch-geometric":
        return "torch-geometric"
    return dep


def _dep_name(dep: Any) -> str:
    """Return normalized package name from a pip requirement line."""
    s = str(dep or "").strip()
    s = re.sub(r"\s+#.*$", "", s).strip()
    s = re.split(r"\s*(?:==|>=|<=|~=|!=|>|<|\[|;)\s*", s, maxsplit=1)[0]
    return s.strip().lower().replace("_", "-")


def _looks_pytorch_model(model: Dict[str, Any]) -> bool:
    deps_text = " ".join(str(x) for x in model.get("dependencies") or [])
    cmd_text = str(model.get("inference_cmd_template") or "")
    docs = _requirements_text(model) + "\n" + _repo_docs(model, max_chars=18000)
    all_text = (deps_text + "\n" + cmd_text + "\n" + docs).lower()
    return bool(re.search(r"\b(import\s+torch|from\s+torch|pytorch|torch==|torch>=|\btorch\b)", all_text))


def _filter_deps_for_model(model: Dict[str, Any], deps: List[str]) -> Tuple[List[str], List[str]]:
    """Filter agent-inferred dependencies before pip install.

    This is intentionally stricter than requirements.txt filtering. If README/code
    show a PyTorch model, old TensorFlow packages are treated as legacy training
    leftovers and are not installed during inference deployment. It also avoids
    common PyTorch/PyG dependency traps during automation: CUDA-local pip specs
    such as torch==1.11.0+cu113 and source-built PyG extension packages often
    conflict with the cluster base environment and should not make env setup fail.
    """
    pyver = str(model.get("python_version") or DEFAULT_PYTHON_VERSION or "3.9")
    m = re.match(r"^(\d+)\.(\d+)", pyver)
    py_tuple = (int(m.group(1)), int(m.group(2))) if m else (3, 9)
    is_torch = _looks_pytorch_model(model)
    base_env = str(_guess_base_env(model) or "").lower()
    base_has_torch = "pt" in base_env or "torch" in base_env

    kept: List[str] = []
    skipped: List[str] = []
    seen_names: set[str] = set()

    # First pass: normalize fragile PyG package name and remove known-bad specs.
    for dep in deps:
        d = str(dep or "").strip()
        if not d:
            continue
        d = _genericize_torch_geometric(d)
        name = _dep_name(d)
        low = d.lower()

        if name in STDLIB_DEPENDENCIES:
            skipped.append(d + "  # skipped: Python standard-library module")
            continue
        mapped_name = NORMALIZED_IMPORT_TO_PIP.get(name)
        if mapped_name and mapped_name.lower() != name:
            match = re.match(r"[A-Za-z0-9_.-]+", d)
            if match:
                d = mapped_name + d[match.end():]
                name = _dep_name(d)
                low = d.lower()

        if is_torch and name in TF_RELATED_PACKAGES:
            skipped.append(d + "  # skipped: PyTorch model; TensorFlow dependency treated as legacy/non-inference")
            continue
        if py_tuple >= (3, 8) and name in TF_RELATED_PACKAGES and re.search(r"==\s*1\.(14|15)(\.\d+)?\b", low):
            skipped.append(d + "  # skipped: TensorFlow 1.14/1.15 incompatible with Python >=3.8")
            continue

        if is_torch and name in TORCH_PRIMARY_PACKAGES:
            # base_pt* environments already provide a tested torch build. Installing
            # another torch wheel from a normal mirror is the most common failure.
            if base_has_torch:
                skipped.append(d + "  # skipped: base PyTorch env already provides torch stack")
                continue
            if _is_torch_cuda_local_spec(d):
                skipped.append(d + "  # skipped: CUDA-local pip spec; install torch via base env/official wheel only")
                continue

        if is_torch and name in TORCH_GEOMETRIC_STACK and name != "torch-geometric":
            skipped.append(d + "  # skipped: PyG compiled extension; defer to smoke/self-heal compatible install")
            continue

        # Name-level de-duplication prevents pairs such as numpy and numpy==1.21.6,
        # torch and torch==1.11.0+cu113 from entering one pip resolver invocation.
        if name in seen_names:
            skipped.append(d + "  # skipped: duplicate dependency name after normalization")
            continue
        seen_names.add(name)
        kept.append(d)

    return kept, skipped


def _rewrite_or_skip_setup_command(cmd: str) -> Optional[str]:
    """Rewrite fragile README download commands to safer HPC-friendly commands."""
    cmd = str(cmd or "").strip().strip("`")
    if not cmd:
        return None
    low = cmd.lower()

    # NCBI FTP downloads are often huge/slow on HPC login nodes. For BLAST,
    # prefer the cluster module/path if available, else conda/bioconda.
    if "ftp.ncbi.nlm.nih.gov" in low or "ncbi-blast" in low or "blast+/" in low:
        if os.getenv("HPC_ALLOW_NCBI_FTP_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "y"}:
            return cmd
        return (
            "if command -v blastp >/dev/null 2>&1; then "
            "echo '>>> [BLAST] existing blastp found: '$(command -v blastp); "
            "elif [ -x /blast/executables/blast+/2.12.0/bin/blastp ]; then "
            "export PATH=/blast/executables/blast+/2.12.0/bin:$PATH; echo '>>> [BLAST] using /blast/executables/blast+/2.12.0/bin'; "
            "else "
            "echo '>>> [BLAST] installing via conda/bioconda instead of slow NCBI FTP'; "
            "conda install -y -c bioconda -c conda-forge blast=2.12.0 || conda install -y -c bioconda -c conda-forge blast || true; "
            "fi"
        )

    # Generic FTP is too fragile for automation unless explicitly allowed.
    if "ftp://" in low and os.getenv("HPC_ALLOW_FTP_DOWNLOADS", "0").strip().lower() not in {"1", "true", "yes", "y"}:
        print(f"    ⚠️ 跳过 README FTP 下载命令，避免长时间卡住: {cmd[:160]}")
        return None

    # README conda install commands frequently omit -y and then block at
    # "Proceed ([y]/n)?" inside a non-interactive SSH session. Add -y safely.
    if re.match(r"^conda\s+install\b", low) and not re.search(r"(^|\s)-y(\s|$)", low) and "--yes" not in low:
        cmd = re.sub(r"^conda\s+install\b", "conda install -y", cmd, count=1, flags=re.I)

    return cmd


def _q(s: Any) -> str:
    return shlex.quote(str(s))


def _setup_cmd_timeout_seconds() -> int:
    try:
        return max(30, int(os.getenv("HPC_MODEL_SETUP_CMD_TIMEOUT_SECONDS", "900")))
    except Exception:
        return 900


def _wrap_env_commands_with_timeout(commands: List[str], label: str = "EnvCmd") -> List[str]:
    """Wrap README/agent setup commands so one hanging download never blocks the whole run.

    Each command is executed with GNU timeout when available. Failures are logged but
    do not abort the whole environment setup; the later smoke test decides readiness.
    Set HPC_SKIP_MODEL_ENV_COMMANDS=1 to skip README/agent setup/download commands.
    Set HPC_MODEL_SETUP_CMD_TIMEOUT_SECONDS=1800 to allow longer downloads.
    """
    if os.getenv("HPC_SKIP_MODEL_ENV_COMMANDS", "0").strip().lower() in {"1", "true", "yes", "y"}:
        return ["echo '>>> [EnvCmd] skipped because HPC_SKIP_MODEL_ENV_COMMANDS=1'"]
    timeout_s = _setup_cmd_timeout_seconds()
    wrapped: List[str] = []
    total = len(commands or [])
    for i, cmd in enumerate(commands or [], 1):
        cmd = str(cmd or "").strip()
        if not cmd:
            continue
        shown = cmd[:220].replace("\n", " ").replace("\r", " ")
        wrapped.append("echo " + _q(f">>> [{label} {i}/{total}] {shown}"))
        wrapped.append("set +e")
        wrapped.append(
            "if command -v timeout >/dev/null 2>&1; then "
            + f"timeout {timeout_s}s bash -lc {_q(cmd)}; "
            + "else "
            + f"bash -lc {_q(cmd)}; "
            + "fi"
        )
        wrapped.append("_envcmd_ec=$?")
        wrapped.append("set -e")
        wrapped.append(f"if [ $_envcmd_ec -eq 124 ]; then echo '>>> [{label}] command timed out after {timeout_s}s'; fi")
        wrapped.append(f"if [ $_envcmd_ec -ne 0 ]; then echo '>>> [{label}] command exited with code '$_envcmd_ec' and will be handled by smoke/self-heal'; fi")
    return wrapped


def _safe_name(text: Any, fallback: str = "model") -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9A-Z._-]+", "_", s).strip("_")
    return s[:80] or fallback


def _read_remote_text(ssh, cmd: str, stream: bool = False) -> Tuple[str, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    if stream:
        out_lines: List[str] = []
        for line in iter(stdout.readline, ""):
            print(f"      [HPC] {line.rstrip()}")
            out_lines.append(line)
        out = "".join(out_lines)
        err = stderr.read().decode("utf-8", errors="ignore")
    else:
        out = stdout.read().decode("utf-8", errors="ignore")
        err = stderr.read().decode("utf-8", errors="ignore")
    return out, err


def _sftp_mkdir_p(sftp, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def _sftp_put_dir(sftp, local_dir: Path, remote_dir: str) -> None:
    local_dir = Path(local_dir)
    _sftp_mkdir_p(sftp, remote_dir)
    for item in local_dir.iterdir():
        if item.name in {".git", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache", "mini_test_outputs"}:
            continue
        remote_path = posixpath.join(remote_dir, item.name)
        if item.is_dir():
            _sftp_put_dir(sftp, item, remote_path)
        elif item.is_file():
            _sftp_mkdir_p(sftp, posixpath.dirname(remote_path))
            sftp.put(str(item), remote_path)


def _load_registry() -> List[Dict[str, Any]]:
    if not LOCAL_REGISTRY.exists():
        return []
    try:
        data = json.loads(LOCAL_REGISTRY.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_registry(rows: List[Dict[str, Any]]) -> None:
    LOCAL_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_REGISTRY.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_registry_row(model_name: str, updates: Dict[str, Any]) -> None:
    rows = _load_registry()
    changed = False
    for row in rows:
        if str(row.get("model_name") or "") == str(model_name):
            row.update({k: v for k, v in updates.items() if v not in [None, ""]})
            changed = True
            break
    if changed:
        _save_registry(rows)


def _local_model_dir(model: Dict[str, Any]) -> Optional[Path]:
    local = model.get("local_model_dir") or ""
    if not local:
        return None
    p = Path(str(local))
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    allowed_root = (ROOT / "data" / "models").resolve()
    allow_external = os.getenv("ALLOW_EXTERNAL_MODEL_DIR", "0").strip().lower() in {"1", "true", "yes"}
    if not allow_external and p != allowed_root and allowed_root not in p.parents:
        print(f"    ⚠️ 拒绝 data/models 之外的 local_model_dir: {p}")
        return None
    return p if p.exists() and p.is_dir() else None


def _read_file_safe(path: Path, max_chars: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _repo_docs(model: Dict[str, Any], max_chars: int = 42000) -> str:
    repo = _local_model_dir(model)
    if not repo:
        return ""
    parts: List[str] = []
    priority_patterns = [
        "README.md", "README*", "readme*",
        "requirements.txt", "requirements*.txt",
        "environment.yml", "environment.yaml", "environment*.yml", "environment*.yaml",
        "setup.py", "pyproject.toml", "*.md",
    ]
    seen = set()
    for pat in priority_patterns:
        for fp in sorted(repo.glob(pat)):
            if fp in seen or not fp.is_file():
                continue
            seen.add(fp)
            parts.append(f"\n===== FILE: {fp.relative_to(repo)} =====\n{_read_file_safe(fp, 12000)}")
    scripts = []
    for fp in sorted(repo.rglob("*.py"))[:200]:
        rel = str(fp.relative_to(repo))
        if any(x in rel.lower() for x in ["__pycache__", ".venv", "site-packages"]):
            continue
        if re.search(r"(predict|infer|inference|test|eval|main|run)", fp.name, re.I):
            scripts.append(rel)
    if scripts:
        parts.append("\n===== PYTHON CANDIDATE SCRIPTS =====\n" + "\n".join(scripts[:80]))
    text = "\n".join(parts)
    return text[:max_chars]


def _requirements_text(model: Dict[str, Any]) -> str:
    repo = _local_model_dir(model)
    if not repo:
        return ""
    req = repo / "requirements.txt"
    return _read_file_safe(req, 20000) if req.exists() else ""


def _safe_env_commands(commands: Any) -> List[str]:
    out = []
    for cmd in commands or []:
        cmd = str(cmd or "").strip()
        if not cmd:
            continue
        low = cmd.lower()
        blocked = ["rm -rf", "rm -r ", "mkfs", ":(){", "shutdown", "reboot", "dd if=", "chmod -r 777", "sudo "]
        if any(b in low for b in blocked):
            print(f"    [WARN] 跳过危险 env_setup_command: {cmd}")
            continue
        allowed_prefix = ("pip ", "python -m pip", "conda ", "wget ", "curl ", "bash ", "sh ", "python ", "if ", "export ")
        if not low.startswith(allowed_prefix):
            print(f"    [WARN] 跳过非白名单 env_setup_command: {cmd}")
            continue
        out.append(cmd)
    return out


def _validated_inference_template(command: Any) -> str:
    """Reject Agent-generated shell payloads and require both runtime placeholders."""
    text = str(command or "").strip()
    lowered = text.lower()
    blocked = ("\n", "\r", "rm -rf", "rm -r ", "sudo ", "mkfs", "shutdown", "reboot", "dd if=", "`", "$(")
    if not text or any(token in lowered for token in blocked):
        return ""
    if "{fasta_path}" not in text or "{output_dir}" not in text:
        return ""
    return text


def _extract_readme_download_commands(model: Dict[str, Any]) -> List[str]:
    cmds: List[str] = []
    for c in model.get("readme_download_commands") or model.get("weight_download_commands") or []:
        if isinstance(c, str) and c.strip():
            rewritten = _rewrite_or_skip_setup_command(c.strip())
            if rewritten:
                cmds.append(rewritten)
    repo = _local_model_dir(model)
    if repo:
        readme = "\n".join(_read_file_safe(fp, 12000) for fp in sorted(repo.glob("README*"))[:3])
        for line in readme.splitlines():
            s = line.strip().strip("` ")
            if not s:
                continue
            if re.match(r"^(wget|curl\s+-L|python\s+.*download|bash\s+.*download|sh\s+.*download|conda\s+install)", s):
                if len(s) < 500:
                    rewritten = _rewrite_or_skip_setup_command(s)
                    if rewritten:
                        cmds.append(rewritten)
    # De-duplicate while preserving order.
    deduped: List[str] = []
    for c in cmds:
        if c not in deduped:
            deduped.append(c)
    return _safe_env_commands(deduped)


def _guess_base_env(model: Dict[str, Any]) -> Optional[str]:
    deps_text = " ".join([str(x) for x in model.get("dependencies") or []]).lower()
    cmd_text = str(model.get("inference_cmd_template") or "").lower()
    docs = (_requirements_text(model) + "\n" + _repo_docs(model, max_chars=12000)).lower()
    all_text = deps_text + " " + cmd_text + " " + docs
    base_map = HPC_BASE_ENV_MAP if isinstance(HPC_BASE_ENV_MAP, dict) else {}
    if re.search(r"\b(import\s+torch|from\s+torch|pytorch|torch==|torch>=|\btorch\b)", all_text):
        if "torch==1" in all_text or "pytorch==1" in all_text:
            vals = base_map.get("torch1") or []
            return vals[0] if vals else "base_pt1"
        vals = base_map.get("torch2") or []
        return vals[0] if vals else "base_pt2"
    if "tensorflow" in all_text or "keras" in all_text or "tf." in all_text:
        vals = base_map.get("tensorflow") or []
        return vals[0] if vals else "base_tf"
    vals = base_map.get("generic") or []
    return vals[0] if vals else None


def _remote_repo_dir(model: Dict[str, Any]) -> str:
    if model.get("remote_repo_dir"):
        return str(model["remote_repo_dir"]).rstrip("/")
    name = _safe_name(model.get("model_name") or model.get("repo_url") or "model")
    return posixpath.join(str(HPC_REMOTE_REPO_ROOT).rstrip("/"), name)


def _make_env_script(model: Dict[str, Any], remote_repo_dir: str) -> str:
    env_name = str(model.get("env_name") or f"env_{_safe_name(model.get('model_name'))}")
    pyver = str(model.get("python_version") or DEFAULT_PYTHON_VERSION or "3.9")
    raw_deps = [str(x).strip() for x in (model.get("dependencies") or []) if str(x).strip()]
    deps, skipped_deps = _filter_deps_for_model(model, raw_deps)
    deps_cmd = " ".join(_q(x) for x in deps)
    base_env = _guess_base_env(model)
    is_torch_model = _looks_pytorch_model(model)
    raw_env_cmds = _safe_env_commands(model.get("env_setup_commands")) + _extract_readme_download_commands(model)
    env_cmds = _wrap_env_commands_with_timeout(raw_env_cmds, label="README/Agent EnvCmd")
    env_cmd_block = "\n".join(env_cmds)

    if base_env:
        create_cmd = f"conda create -n {_q(env_name)} --clone {_q(base_env)} -y"
    else:
        create_cmd = f"conda create -n {_q(env_name)} python={_q(pyver)} -y"

    filter_block = rf'''
python - <<'FILTERPY'
import re, sys, pathlib
IS_TORCH_MODEL = {repr(is_torch_model)}
TF_RELATED = {'tensorflow', 'tensorflow-gpu', 'tensorflow-estimator'}
TORCH_PRIMARY = {'torch', 'torchvision', 'torchaudio'}
PYG_COMPILED = {'torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv'}
def dep_name(line):
    line = re.sub(r'\s+#.*$', '', str(line or '').strip())
    return re.split(r'\s*(?:==|>=|<=|~=|!=|>|<|\[|;)\s*', line, maxsplit=1)[0].strip().lower().replace('_','-')
def is_cuda_local_torch(line):
    return re.search(r'^(torch|torchvision|torchaudio)\s*==\s*[^\s;]+\+cu\d+', str(line or '').strip().lower()) is not None
req = pathlib.Path('requirements.txt')
out = pathlib.Path('requirements.hpc_auto_filtered.txt')
if req.exists():
    py = sys.version_info[:2]
    kept = []
    skipped = []
    for raw in req.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw.strip()
        low = line.lower()
        if not line or line.startswith('#'):
            continue
        name = dep_name(line)
        if IS_TORCH_MODEL and name in TF_RELATED:
            skipped.append(line + '  # skipped: PyTorch model; TensorFlow dependency treated as legacy/non-inference')
            continue
        if py >= (3, 8) and name in TF_RELATED and re.search(r'==\s*1\.(14|15)(\.\d+)?\b', low):
            skipped.append(line + '  # skipped: incompatible with Python >=3.8')
            continue
        if IS_TORCH_MODEL and name in TORCH_PRIMARY:
            skipped.append(line + '  # skipped: base PyTorch env already provides torch stack')
            continue
        if IS_TORCH_MODEL and is_cuda_local_torch(line):
            skipped.append(line + '  # skipped: CUDA-local torch spec unsupported by generic pip index')
            continue
        if IS_TORCH_MODEL and name in PYG_COMPILED:
            skipped.append(line + '  # skipped: PyG compiled extension; defer to smoke/self-heal compatible install')
            continue
        if IS_TORCH_MODEL and name == 'torch-geometric':
            raw = 'torch-geometric'
        kept.append(raw)
    out.write_text('\\n'.join(kept) + ('\\n' if kept else ''), encoding='utf-8')
    pathlib.Path('requirements.hpc_auto_skipped.txt').write_text('\\n'.join(skipped) + ('\\n' if skipped else ''), encoding='utf-8')
    print('>>> [ReqFilter] kept', len(kept), 'requirements; skipped', len(skipped))
    if skipped:
        print('>>> [ReqFilter] skipped lines:')
        print('\\n'.join(skipped))
FILTERPY
'''

    script = f"""
set -e
cd {_q(HPC_TARGET_DIR)}
source {_q(CONDA_SH_PATH)}
mkdir -p {_q(remote_repo_dir)}
mkdir -p {_q(HPC_TARGET_DIR)}/data

if conda env list | awk '{{print $1}}' | grep -qx {_q(env_name)}; then
    if [ "${{HPC_FORCE_RECREATE_MODEL_ENV:-0}}" = "1" ]; then
        echo '>>> [Conda] HPC_FORCE_RECREATE_MODEL_ENV=1, removing env {env_name}'
        conda env remove -n {_q(env_name)} -y || true
        echo '>>> [Conda] creating env {env_name}'
        {create_cmd}
    else
        echo '>>> [Conda] env {env_name} already exists'
    fi
else
    echo '>>> [Conda] creating env {env_name}'
    {create_cmd}
fi

conda activate {_q(env_name)}
python --version || true
pip --version || true

cd {_q(remote_repo_dir)}

{filter_block}

if [ -f environment.yml ]; then
    echo '>>> [Conda] found environment.yml, updating env...'
    conda env update -n {_q(env_name)} -f environment.yml || true
fi

if [ -f requirements.hpc_auto_filtered.txt ]; then
    echo '>>> [Pip] installing requirements.hpc_auto_filtered.txt...'
    python -m pip install -i {_q(PIP_INDEX_URL)} --extra-index-url {_q(PIP_EXTRA_INDEX_URL)} -r requirements.hpc_auto_filtered.txt || true
elif [ -f requirements.txt ]; then
    echo '>>> [Pip] installing requirements.txt...'
    python -m pip install -i {_q(PIP_INDEX_URL)} --extra-index-url {_q(PIP_EXTRA_INDEX_URL)} -r requirements.txt || true
fi

if [ -f setup.py ] || [ -f pyproject.toml ]; then
    echo '>>> [Pip] installing local package editable...'
    python -m pip install -i {_q(PIP_INDEX_URL)} --extra-index-url {_q(PIP_EXTRA_INDEX_URL)} -e . || true
fi
"""
    if skipped_deps:
        script += "\necho '>>> [DepFilter] skipped agent-inferred dependencies:'\n"
        for dep in skipped_deps:
            script += f"echo {_q(dep)}\n"
    if deps_cmd:
        script += f"""
echo '>>> [Pip] installing agent-inferred dependencies...'
python -m pip install -i {_q(PIP_INDEX_URL)} --extra-index-url {_q(PIP_EXTRA_INDEX_URL)} {deps_cmd} || true
"""
    if env_cmd_block:
        script += "\necho '>>> [README/Agent Env Commands] running model-specific setup/download commands...'\n"
        script += env_cmd_block + "\n"
    script += "\necho '>>> [Env] setup finished.'\n"
    return script


def _make_smoke_script(model: Dict[str, Any], remote_repo_dir: str) -> str:
    env_name = str(model.get("env_name") or f"env_{_safe_name(model.get('model_name'))}")
    model_name = str(model.get("model_name") or "model")
    cmd = _validated_inference_template(model.get("inference_cmd_template"))
    out_dir = posixpath.join(str(HPC_TARGET_DIR), "data", "mini_test_outputs", _safe_name(model_name))
    fasta = posixpath.join(str(HPC_TARGET_DIR), "data", "vlab_mini_test.fasta")
    cmd = cmd.replace("{fasta_path}", fasta).replace("{output_dir}", out_dir)
    if not cmd:
        return "echo 'No inference_cmd_template; skip smoke test'; exit 88"
    mini_content = os.getenv(
        "MINI_TEST_FASTA_CONTENT",
        ">mini_seq_1\\nACDEFGHIKLMNPQRSTVWY\\n>mini_seq_2\\nKLLKLLKLLKLL\\n",
    )
    return f"""
set -e
cd {_q(HPC_TARGET_DIR)}
source {_q(CONDA_SH_PATH)}
conda activate {_q(env_name)}
mkdir -p {_q(out_dir)}
printf %b {_q(mini_content)} > {_q(fasta)}
cd {_q(remote_repo_dir)}
echo '>>> [Smoke] running: {cmd}'
{cmd}
echo '>>> [Smoke] output tree:'
find {_q(out_dir)} -maxdepth 3 -type f -printf '%p %s bytes\\n' || true
echo '>>> [Smoke] done.'
"""


def _make_argparse_smoke_fallback_script(model: Dict[str, Any], remote_repo_dir: str) -> str:
    """Try a README/argparse-compatible smoke test when generic --input/--output failed."""
    env_name = str(model.get("env_name") or f"env_{_safe_name(model.get('model_name'))}")
    model_name = str(model.get("model_name") or "model")
    out_dir = posixpath.join(str(HPC_TARGET_DIR), "data", "mini_test_outputs", _safe_name(model_name))
    fasta = posixpath.join(str(HPC_TARGET_DIR), "data", "vlab_mini_test.fasta")
    mini_content = os.getenv(
        "MINI_TEST_FASTA_CONTENT",
        ">mini_seq_1\nACDEFGHIKLMNPQRSTVWY\n>mini_seq_2\nKLLKLLKLLKLL\n",
    )
    return f"""
set +e
cd {_q(HPC_TARGET_DIR)}
source {_q(CONDA_SH_PATH)}
conda activate {_q(env_name)}
mkdir -p {_q(out_dir)}
printf %b {_q(mini_content)} > {_q(fasta)}
cd {_q(remote_repo_dir)}
echo '>>> [SmokeFallback] --input/--output 不被脚本接受，开始读取 argparse/README 风格命令。'

if [ -f test.py ]; then
    python test.py -h > {_q(out_dir)}/test_help.txt 2>&1 || true
    echo '>>> [SmokeFallback] test.py -h:'
    head -n 120 {_q(out_dir)}/test_help.txt || true

    if grep -q -- '-pos_t' {_q(out_dir)}/test_help.txt && grep -q -- '-pos_npz' {_q(out_dir)}/test_help.txt && grep -q -- '-neg_t' {_q(out_dir)}/test_help.txt && grep -q -- '-neg_npz' {_q(out_dir)}/test_help.txt; then
        POS_T=$(find . -type f -iname '*pos*.pt' | head -n 1)
        if [ -z "$POS_T" ]; then POS_T=$(find . -type f -iname '*positive*.pt' | head -n 1); fi
        if [ -z "$POS_T" ]; then POS_T=$(find . -type f -iname '*pos*.pth' | head -n 1); fi
        NEG_T=$(find . -type f -iname '*neg*.pt' | head -n 1)
        if [ -z "$NEG_T" ]; then NEG_T=$(find . -type f -iname '*negative*.pt' | head -n 1); fi
        if [ -z "$NEG_T" ]; then NEG_T=$(find . -type f -iname '*neg*.pth' | head -n 1); fi
        POS_NPZ=$(find . -type f -iname '*pos*.npz' | head -n 1)
        if [ -z "$POS_NPZ" ]; then POS_NPZ=$(find . -type f -iname '*positive*.npz' | head -n 1); fi
        NEG_NPZ=$(find . -type f -iname '*neg*.npz' | head -n 1)
        if [ -z "$NEG_NPZ" ]; then NEG_NPZ=$(find . -type f -iname '*negative*.npz' | head -n 1); fi
        echo '>>> [SmokeFallback] detected files:'
        echo "    POS_T=$POS_T"
        echo "    POS_NPZ=$POS_NPZ"
        echo "    NEG_T=$NEG_T"
        echo "    NEG_NPZ=$NEG_NPZ"
        if [ -n "$POS_T" ] && [ -n "$POS_NPZ" ] && [ -n "$NEG_T" ] && [ -n "$NEG_NPZ" ]; then
            echo '>>> [SmokeFallback] running test.py with repository-native graph inputs.'
            python test.py -pos_t "$POS_T" -pos_npz "$POS_NPZ" -neg_t "$NEG_T" -neg_npz "$NEG_NPZ" -save {_q(out_dir)} -o {_q(out_dir)}/predictions.csv
            EC=$?
            echo '>>> [SmokeFallback] output tree:'
            find {_q(out_dir)} -maxdepth 3 -type f -printf '%p %s bytes\n' || true
            exit $EC
        else
            echo '>>> [SmokeFallback] 未找到 test.py 所需的 pos/neg .pt/.npz 文件。请确认 README 下载步骤是否已经下载/生成图特征文件。'
            echo '>>> [SmokeFallback] repository files preview:'
            find . -maxdepth 4 -type f | sed 's#^./##' | head -n 200
            exit 2
        fi
    fi
fi

echo '>>> [SmokeFallback] 没有识别到可自动构造的 repository-native smoke test。'
exit 3
"""


def _fatal_env_error(text: str) -> bool:
    low = text.lower()
    fatal = [
        "resolvepackagenotfound", "condahttperror", "traceback (most recent call last)",
        "could not find a version that satisfies the requirement",
        "no matching distribution found",
        "error: subprocess-exited-with-error",
        "failed building wheel",
    ]
    if "[reqfilter] skipped lines" in low and "tensorflow==1.14" in low and "no matching distribution" not in low:
        return False
    return any(x in low for x in fatal)


def _run_smoke(ssh, model: Dict[str, Any], remote_dir: str, smoke_log_path: str) -> Tuple[bool, str]:
    smoke_script = _make_smoke_script(model, remote_dir)
    cmd = f"({smoke_script}) > {_q(smoke_log_path)} 2>&1; echo $?"
    out, _ = _read_remote_text(ssh, cmd, stream=False)
    code = out.strip().splitlines()[-1] if out.strip() else "1"
    log, _ = _read_remote_text(ssh, f"tail -n 160 {_q(smoke_log_path)} || true")

    if code != "0" and "unrecognized arguments" in log.lower() and ("--input" in log or "--output" in log):
        print("    >>> [Smoke] 检测到脚本不支持通用 --input/--output，尝试 argparse/README 兼容 smoke fallback...")
        fallback_script = _make_argparse_smoke_fallback_script(model, remote_dir)
        cmd2 = f"({fallback_script}) >> {_q(smoke_log_path)} 2>&1; echo $?"
        out2, _ = _read_remote_text(ssh, cmd2, stream=False)
        code = out2.strip().splitlines()[-1] if out2.strip() else "1"
        log, _ = _read_remote_text(ssh, f"tail -n 220 {_q(smoke_log_path)} || true")

    print("    >>> [Smoke tail]")
    print(log)
    return code == "0", log


def _llm_client_and_model():
    try:
        from openai import OpenAI
    except Exception:
        return None, ""
    model = os.getenv("SELF_HEAL_LLM_MODEL") or os.getenv("ONBOARDING_LLM_MODEL") or os.getenv("MODEL_NAME") or "gpt-5.2"
    provider = os.getenv("SELF_HEAL_LLM_PROVIDER") or os.getenv("ONBOARDING_LLM_PROVIDER", "auto")
    provider = provider.lower()
    if provider == "dashscope" or (provider == "auto" and model.startswith("qwen")):
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            return None, model
        return OpenAI(api_key=key, base_url=os.getenv("DASHSCOPE_OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")), model
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None, model
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=key, base_url=base_url), model
    return OpenAI(api_key=key), model


def _parse_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _heuristic_self_heal_plan(model: Dict[str, Any], smoke_log: str, env_output: str = "") -> Dict[str, Any]:
    text = (smoke_log + "\n" + env_output)
    plan: Dict[str, Any] = {"diagnosis": "heuristic fallback", "pip_install": [], "conda_install": [], "env_setup_commands": [], "registry_updates": {}, "retry_smoke": True}
    mods = re.findall(r"(?:ModuleNotFoundError|ImportError):\s+No module named ['\"]([^'\"]+)['\"]", text)
    for mod in mods:
        root = mod.split(".")[0]
        pkg = PIP_IMPORT_MAP.get(root, root)
        if pkg not in plan["pip_install"]:
            plan["pip_install"].append(pkg)
    if "import torch" in text.lower() or "no module named 'torch'" in text.lower() or 'no module named "torch"' in text.lower():
        if "torch" not in plan["pip_install"]:
            plan["pip_install"].insert(0, "torch")
    if "no matching distribution found for tensorflow==1.14" in text.lower() or "tensorflow==1.14.0" in text.lower() or "the user requested tensorflow==1.14" in text.lower():
        plan["diagnosis"] += "; tensorflow 1.14 incompatible/currently conflicting, filtered/ignored for PyTorch inference unless README proves old TF is required"
        plan.setdefault("remove_requirement_patterns", []).extend(["tensorflow", "tensorflow-gpu", "tensorflow-estimator"])
        plan.setdefault("remove_dependency_patterns", []).extend(["tensorflow", "tensorflow-gpu", "tensorflow-estimator"])
    if "ftp.ncbi.nlm.nih.gov" in text.lower() or "ncbi-blast" in text.lower() or "control connection closed" in text.lower():
        plan["diagnosis"] += "; NCBI BLAST FTP download is unstable, replace with existing cluster BLAST or conda/bioconda install"
        blast_cmd = "if command -v blastp >/dev/null 2>&1; then echo '>>> [BLAST] existing blastp found'; elif [ -x /blast/executables/blast+/2.12.0/bin/blastp ]; then export PATH=/blast/executables/blast+/2.12.0/bin:$PATH; else conda install -y -c bioconda -c conda-forge blast=2.12.0 || conda install -y -c bioconda -c conda-forge blast || true; fi"
        if blast_cmd not in plan.setdefault("env_setup_commands", []):
            plan["env_setup_commands"].append(blast_cmd)
    if "unrecognized arguments" in text.lower():
        plan["diagnosis"] += "; inference arguments may be wrong; ask README-agent to rewrite inference_cmd_template"
    return plan


def _agent_self_heal_plan(model: Dict[str, Any], smoke_log: str, env_output: str = "") -> Dict[str, Any]:
    client, llm_model = _llm_client_and_model()
    docs = _repo_docs(model, max_chars=42000)
    req = _requirements_text(model)
    heuristic = _heuristic_self_heal_plan(model, smoke_log, env_output)
    if client is None:
        return validate_self_heal_plan(model, heuristic)
    system = _SELF_HEAL_PROMPTS.load_composed("self_heal_system")
    user = _SELF_HEAL_PROMPTS.render(
        "self_heal_task",
        {
            "registry_json": json.dumps(model, ensure_ascii=False, indent=2),
            "repository_docs": docs,
            "requirements_text": req[:20000],
            "environment_output": env_output[-12000:],
            "smoke_log": smoke_log[-16000:],
        },
        composed=True,
    )
    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
        )
        obj = _parse_json_from_text(resp.choices[0].message.content or "{}")
        if obj:
            for pkg in heuristic.get("pip_install", []):
                obj.setdefault("pip_install", [])
                if pkg not in obj["pip_install"]:
                    obj["pip_install"].append(pkg)
            return validate_self_heal_plan(model, obj)
    except Exception as e:
        print(f"    ⚠️ Agent 自愈计划生成失败，使用启发式：{e}")
    return validate_self_heal_plan(model, heuristic)


def validate_self_heal_plan(model: Dict[str, Any], plan: Any) -> Dict[str, Any]:
    """Convert an Agent suggestion into a bounded, code-enforced repair plan."""
    source = plan if isinstance(plan, dict) else {}
    pip_raw = [str(value).strip() for value in source.get("pip_install") or [] if str(value).strip()]
    pip_install, _ = _filter_deps_for_model(model, pip_raw)

    conda_install: List[str] = []
    for value in source.get("conda_install") or []:
        package = str(value or "").strip()
        if package and re.fullmatch(r"[A-Za-z0-9_.:+<>=!~-]+", package):
            conda_install.append(package)

    updates_in = source.get("registry_updates") if isinstance(source.get("registry_updates"), dict) else {}
    updates: Dict[str, Any] = {}
    version = str(updates_in.get("python_version") or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+){1,2}", version):
        updates["python_version"] = version
    deps_raw = updates_in.get("dependencies") if isinstance(updates_in.get("dependencies"), list) else []
    dependencies, _ = _filter_deps_for_model(model, [str(value).strip() for value in deps_raw if str(value).strip()])
    if dependencies:
        updates["dependencies"] = dependencies
    inference = _validated_inference_template(updates_in.get("inference_cmd_template"))
    if inference:
        updates["inference_cmd_template"] = inference

    patterns: List[str] = []
    for value in source.get("remove_requirement_patterns") or source.get("remove_dependency_patterns") or []:
        pattern = str(value or "").strip()
        if pattern and len(pattern) <= 120 and "\n" not in pattern and pattern not in patterns:
            patterns.append(pattern)

    return {
        "diagnosis": str(source.get("diagnosis") or "No Agent diagnosis supplied")[:4000],
        "pip_install": pip_install,
        "conda_install": conda_install[:30],
        "env_setup_commands": _safe_env_commands(source.get("env_setup_commands"))[:30],
        "registry_updates": updates,
        "remove_requirement_patterns": patterns[:50],
        "retry_smoke": bool(source.get("retry_smoke", True)),
    }


def _merge_model_updates(model: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(updates, dict):
        return model
    new = dict(model)
    allowed_keys = {"python_version", "dependencies", "inference_cmd_template"}
    for k, v in updates.items():
        if k not in allowed_keys:
            continue
        if v in [None, "", [], {}]:
            continue
        if k == "dependencies" and isinstance(v, list):
            old = [str(x) for x in new.get("dependencies") or []]
            for dep in v:
                dep = str(dep).strip()
                if dep and dep not in old:
                    old.append(dep)
            new["dependencies"] = old
        elif k == "inference_cmd_template":
            validated = _validated_inference_template(v)
            if validated:
                new[k] = validated
        elif k == "python_version" and re.fullmatch(r"\d+(?:\.\d+){1,2}", str(v)):
            new[k] = v
    return new




def _prune_dependencies(model: Dict[str, Any], patterns: List[str]) -> Dict[str, Any]:
    if not patterns:
        return model
    new = dict(model)
    deps = [str(x) for x in new.get("dependencies") or []]
    kept: List[str] = []
    for dep in deps:
        low = dep.lower()
        dep_name = _dep_name(dep)
        remove = False
        for pat in patterns:
            p = str(pat or "").lower().strip()
            if not p:
                continue
            if p in low or p == dep_name:
                remove = True
                break
        if not remove:
            kept.append(dep)
    new["dependencies"] = kept
    return new

def _apply_self_heal_plan(ssh, model: Dict[str, Any], remote_repo_dir: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    env_name = str(model.get("env_name") or f"env_{_safe_name(model.get('model_name'))}")
    print("    >>> [SelfHeal] plan:")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    SELF_HEAL_DIR.mkdir(parents=True, exist_ok=True)
    (SELF_HEAL_DIR / f"{_safe_name(model.get('model_name'))}_{int(time.time())}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    remove_patterns = [str(x).strip() for x in (plan.get("remove_dependency_patterns") or plan.get("remove_requirement_patterns") or []) if str(x).strip()]
    model = _prune_dependencies(model, remove_patterns)
    raw_pip_install = [str(x).strip() for x in plan.get("pip_install") or [] if str(x).strip()]
    pip_install, skipped_heal_deps = _filter_deps_for_model(model, raw_pip_install)
    if skipped_heal_deps:
        print("    >>> [SelfHeal DepFilter] skipped:")
        for dep in skipped_heal_deps:
            print("       " + dep)
    conda_install = [str(x).strip() for x in plan.get("conda_install") or [] if str(x).strip()]
    raw_extra_cmds = _safe_env_commands(plan.get("env_setup_commands"))
    extra_cmds = _wrap_env_commands_with_timeout(raw_extra_cmds, label="SelfHeal EnvCmd")

    blocks: List[str] = [
        f"cd {_q(HPC_TARGET_DIR)}",
        f"source {_q(CONDA_SH_PATH)}",
        f"conda activate {_q(env_name)}",
        f"cd {_q(remote_repo_dir)}",
    ]
    if conda_install:
        blocks.append("echo '>>> [SelfHeal] conda install packages...'")
        blocks.append("conda install -y " + " ".join(_q(x) for x in conda_install) + " || true")
    if pip_install:
        blocks.append("echo '>>> [SelfHeal] pip install packages...'")
        blocks.append("python -m pip install -i " + _q(PIP_INDEX_URL) + " --extra-index-url " + _q(PIP_EXTRA_INDEX_URL) + " " + " ".join(_q(x) for x in pip_install) + " || true")
    if extra_cmds:
        blocks.append("echo '>>> [SelfHeal] README/Agent commands...'")
        blocks.extend(extra_cmds)
    if len(blocks) > 4:
        out, err = _read_remote_text(ssh, "\n".join(blocks), stream=True)
        if err.strip():
            print(f"    [SelfHeal stderr] {err[-1500:]}")

    new_model = _merge_model_updates(model, plan.get("registry_updates") or {})
    if pip_install:
        new_model = _merge_model_updates(new_model, {"dependencies": pip_install})
    return new_model


def ensure_models_ready_on_hpc(ssh, models_info: List[Dict[str, Any]], mark_registry: bool = True, run_smoke_test: bool = True) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    max_heal_attempts = int(os.getenv("HPC_SELF_HEAL_MAX_ATTEMPTS", "2"))
    sftp = ssh.open_sftp()
    try:
        _read_remote_text(ssh, f"mkdir -p {_q(HPC_REMOTE_REPO_ROOT)} {_q(HPC_TARGET_DIR)}/data/mini_test_outputs")
        for original_model in models_info:
            model = dict(original_model)
            name = str(model.get("model_name") or "unknown_model")
            env_name = str(model.get("env_name") or f"env_{_safe_name(name)}")
            remote_dir = _remote_repo_dir(model)
            local_dir = _local_model_dir(model)

            print(f"\n>>> [HPC ModelOps] 准备模型: {name}")
            print(f"    env={env_name}")
            print(f"    remote_repo_dir={remote_dir}")
            docs = _repo_docs(model, max_chars=20000)
            if docs:
                print("    -> 已读取本地 README/requirements/environment，用于安装与自愈判断。")

            upload_ok = False
            if local_dir and local_dir.exists() and local_dir.is_dir():
                print(f"    -> 上传本地模型目录: {local_dir}")
                _sftp_put_dir(sftp, local_dir, remote_dir)
                upload_ok = True
            else:
                print(f"    ⚠️ 本地模型目录不存在或未设置: {local_dir}; 将尝试使用远端已有目录。")
                _read_remote_text(ssh, f"mkdir -p {_q(remote_dir)}")

            env_setup_ok = False
            smoke_ok = False
            smoke_log_path = posixpath.join(str(HPC_TARGET_DIR), "data", f"smoke_{_safe_name(name)}.log")
            matched_base_env = _guess_base_env(model)
            env_out = ""
            env_err = ""

            try:
                script = _make_env_script(model, remote_dir)
                out, err = _read_remote_text(ssh, script, stream=True)
                env_out, env_err = out, err
                if err.strip():
                    print(f"    [Env stderr] {err[-2000:]}")
                combined = out + "\n" + err
                env_setup_ok = not _fatal_env_error(combined)
            except Exception as e:
                print(f"    ❌ 环境创建失败: {e}")
                env_setup_ok = False

            if run_smoke_test and env_setup_ok:
                for attempt in range(0, max_heal_attempts + 1):
                    smoke_ok, smoke_log = _run_smoke(ssh, model, remote_dir, smoke_log_path)
                    if smoke_ok:
                        break
                    if attempt >= max_heal_attempts:
                        break
                    print(f"    ⚠️ smoke test 未通过，启动 Agent 自愈 {attempt+1}/{max_heal_attempts}...")
                    plan = _agent_self_heal_plan(model, smoke_log=smoke_log, env_output=env_out + "\n" + env_err)
                    if not plan.get("retry_smoke", True):
                        break
                    model = _apply_self_heal_plan(ssh, model, remote_dir, plan)
                    env_name = str(model.get("env_name") or env_name)
            elif not run_smoke_test:
                smoke_ok = True

            if mark_registry and env_setup_ok and smoke_ok:
                _update_registry_row(name, {
                    **model,
                    "skip_env_setup": True,
                    "remote_repo_dir": remote_dir,
                    "env_name": env_name,
                    "hpc_env_status": "ready",
                    "hpc_smoke_test": "passed" if run_smoke_test else "skipped",
                    "matched_base_env": matched_base_env,
                    "last_hpc_setup_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"    ✅ 已写回 registry: {name} skip_env_setup=true")
            else:
                _update_registry_row(name, {
                    **model,
                    "remote_repo_dir": remote_dir,
                    "env_name": env_name,
                    "hpc_env_status": "failed_or_incomplete",
                    "hpc_smoke_test": "failed" if run_smoke_test else "skipped",
                    "matched_base_env": matched_base_env,
                    "last_hpc_setup_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

            results.append({
                "model_name": name,
                "env_name": env_name,
                "upload_ok": upload_ok,
                "env_setup_ok": env_setup_ok,
                "smoke_test_ok": smoke_ok,
                "matched_base_env": matched_base_env,
                "remote_repo_dir": remote_dir,
                "smoke_test_log_path": smoke_log_path,
                "self_heal_attempts_allowed": max_heal_attempts,
            })
    finally:
        try:
            sftp.close()
        except Exception:
            pass
    return results
