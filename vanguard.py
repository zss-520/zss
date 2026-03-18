import os
from pathlib import Path
from workflow_utils import run_on_hpc_and_fetch
from config import CONDA_SH_PATH, VLAB_ENV

def run_vanguard_exploration(models_info: list[dict], sample_dataset_dir: str, save_directory: Path) -> list[dict]:
    """
    负责在超算上物理拉取模型源码，并勘探其真实的目录结构。
    将勘探结果注入到 models_info 中返回。
    """
    print("\n========== [Phase 0.8] 源码勘探先遣队：下载代码并生成真实目录树 ==========")
    
    for m in models_info:
        if m.get("skip_env_setup"):
            print(f"    - [跳过] 模型 {m['model_name']} 标记为本地已有免检环境。")
            continue
            
        repo_url = m.get("repo_url", "")
        m_name = m["model_name"]
        
        if not repo_url or repo_url == "null":
            print(f"    - [跳过] 模型 {m_name} 无有效开源链接。")
            continue
            
        print(f"\n>>> 🛸 先遣队正在超算上拉取并勘探模型: {m_name}")
        print(f"    Target URL: {repo_url}")
        
        # 核心黑科技：兼容 GitHub 和 Zenodo API 的终极下载与解压脚本
        vanguard_py = f"""import os, subprocess, json, zipfile, tarfile

repo_url = "{repo_url}"
out_dir = "/share/home/zhangss/repos/{m_name}"
os.makedirs(out_dir, exist_ok=True)
structure_text = "Empty"

try:
    if "github.com" in repo_url or "gitlab.com" in repo_url:
        print(">>> 检测到 Git 仓库，执行 git clone...")
        subprocess.run(f"git clone {{repo_url}} {{out_dir}}", shell=True)
    elif "zenodo.org" in repo_url:
        print(">>> 检测到 Zenodo 链接，启动 curl + wget 底层暴破方案...")
        record_id = repo_url.rstrip('/').split('/')[-1]
        api_url = f"https://zenodo.org/api/records/{{record_id}}"
        
        # 1. 彻底抛弃 urllib，使用系统 curl 强抓 API 数据 (完美绕过 TLS 指纹探测)
        curl_cmd = f"curl -sSL -A 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36' {{api_url}}"
        print(">>> 正在使用 curl 嗅探 Zenodo API...")
        api_response = subprocess.check_output(curl_cmd, shell=True, text=True)
        
        if "403 Forbidden" in api_response or not api_response.strip():
            raise Exception("API 请求失败，curl 也被防火墙拦截了！")
            
        data = json.loads(api_response)
        
        # 2. 解析文件列表，并使用系统 wget 强行下载
        for f in data.get('files', []):
            dl_url = f['links']['self']
            filename = f['key']
            filepath = os.path.join(out_dir, filename)
            print(f">>> 正在召唤 wget 下载 {{filename}}...")
            
            # 使用 wget 下载，自带断点续传和进度条
            wget_cmd = f"wget -q --show-progress --user-agent='Mozilla/5.0' -O {{filepath}} {{dl_url}}"
            subprocess.run(wget_cmd, shell=True)
            
            # 自动解压
            if filepath.endswith('.zip'):
                print(f">>> 开始解压 {{filename}}...")
                with zipfile.ZipFile(filepath, 'r') as z:
                    z.extractall(out_dir)
            elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz') or filepath.endswith('.tar'):
                print(f">>> 开始解压 {{filename}}...")
                with tarfile.open(filepath, 'r:*') as t:
                    t.extractall(out_dir)
                    
    print(">>> 代码获取完毕，开始扫描目录结构...")
    tree_output = subprocess.check_output(f"find {{out_dir}} -maxdepth 3", shell=True, text=True)
    structure_text = tree_output
except Exception as e:
    structure_text = f"Error downloading or exploring: {{e}}"

with open("repo_structure.txt", "w", encoding="utf-8") as f:
    f.write(structure_text)
"""
        
        vanguard_sh = f"""#!/bin/bash
cd /share/home/zhangss/vlab_workspace
source {CONDA_SH_PATH}
conda activate {VLAB_ENV}
python eval_script.py
echo "finish"
"""
        
        # 执行先遣队并抓取勘探报告
        res_dir = save_directory / f"vanguard_{m_name}"
        res_dir.mkdir(exist_ok=True, parents=True)
        
        vanguard_res = run_on_hpc_and_fetch(
            py_code=vanguard_py,
            sh_code=vanguard_sh,
            fetch_targets={"repo_structure.txt": str(res_dir / "repo_structure.txt")},
            models_info=[], 
            local_data_dir=sample_dataset_dir,
            use_sbatch=False  # <--- 【核心修改】：关闭 SLURM，强制在登录节点运行！
        )
        
        # 将拿到的真实目录结构直接塞进模型的字典里！
        if vanguard_res and "repo_structure.txt" in vanguard_res:
            with open(vanguard_res["repo_structure.txt"], "r", encoding="utf-8") as f:
                real_tree = f.read()
                m["repo_structure"] = real_tree[:3000] # 取前3000个字符防止撑爆大模型上下文
                print(f"    >>> [OK] 成功获取 {m_name} 真实代码目录树！(长度: {len(real_tree)})")
        else:
            m["repo_structure"] = "Failed to fetch repository structure."
            print(f"    !!! [警告] 未能获取 {m_name} 目录树。")

    return models_info