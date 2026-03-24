import os
from pathlib import Path
from workflow_utils import run_on_hpc_and_fetch
from config import CONDA_SH_PATH, VLAB_ENV, MODEL_NAME
from openai import OpenAI
from agent import Agent
from prompts import README_ANALYST_PROMPT

def run_vanguard_exploration(models_info: list[dict], sample_dataset_dir: str, save_directory: Path) -> list[dict]:
    """
    负责在超算上物理拉取模型源码，并勘探其真实的目录结构与 README。
    唤醒 README Analyst 自动推导执行命令，并将结果注入到 models_info 中返回。
    """
    print("\n========== [Phase 0.8] 源码勘探先遣队：拉取代码并自动推导执行命令 ==========")
    
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
        print(">>> 检测到 Zenodo 链接，启动底层抓取方案...")
        record_id = repo_url.rstrip('/').split('/')[-1]
        api_url = f"[https://zenodo.org/api/records/](https://zenodo.org/api/records/){{record_id}}"
        curl_cmd = f"curl -sSL -A 'Mozilla/5.0' {{api_url}}"
        api_response = subprocess.check_output(curl_cmd, shell=True, text=True)
        data = json.loads(api_response)
        
        for f in data.get('files', []):
            dl_url = f['links']['self']
            filename = f['key']
            filepath = os.path.join(out_dir, filename)
            print(f">>> 正在召唤 wget 下载 {{filename}}...")
            subprocess.run(f"wget -q --show-progress -O {{filepath}} {{dl_url}}", shell=True)
            if filepath.endswith('.zip'):
                with zipfile.ZipFile(filepath, 'r') as z: z.extractall(out_dir)
            elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz') or filepath.endswith('.tar'):
                with tarfile.open(filepath, 'r:*') as t: t.extractall(out_dir)
                    
    print(">>> 代码获取完毕，开始扫描目录结构并寻找 README...")
    tree_output = subprocess.check_output(f"find {{out_dir}} -maxdepth 3", shell=True, text=True)
    structure_text = tree_output
    
    # 👇 核心新增：强行搜查 README 并读取前 5000 字
    readme_text = ""
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.lower() in ['readme.md', 'readme', 'readme.txt', 'readme.rst']:
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as rf:
                        readme_text = rf.read()[:5000] 
                    break
                except Exception:
                    pass
        if readme_text: break

except Exception as e:
    structure_text = f"Error: {{e}}"
    readme_text = ""

with open("repo_structure.txt", "w", encoding="utf-8") as f:
    f.write(structure_text)
with open("readme_content.txt", "w", encoding="utf-8") as f:
    f.write(readme_text)
"""
        
        vanguard_sh = f"""#!/bin/bash
cd /share/home/zhangss/vlab_workspace
source {CONDA_SH_PATH}
conda activate {VLAB_ENV}
python eval_script.py
echo "finish"
"""
        res_dir = save_directory / f"vanguard_{m_name}"
        res_dir.mkdir(exist_ok=True, parents=True)
        
        # 🚨 获取这两个宝贵的情报文件
        vanguard_res = run_on_hpc_and_fetch(
            py_code=vanguard_py,
            sh_code=vanguard_sh,
            fetch_targets={
                "repo_structure.txt": str(res_dir / "repo_structure.txt"),
                "readme_content.txt": str(res_dir / "readme_content.txt") # <== 新增抓取目标
            },
            models_info=[], 
            local_data_dir=sample_dataset_dir,
            use_sbatch=False
        )
        
        if vanguard_res and "repo_structure.txt" in vanguard_res:
            with open(vanguard_res["repo_structure.txt"], "r", encoding="utf-8") as f:
                m["repo_structure"] = f.read()[:3000]
                
        # ==================================================
        # 🧠 唤醒大模型，阅读 README 并推导命令
        # ==================================================
        if vanguard_res and "readme_content.txt" in vanguard_res:
            with open(vanguard_res["readme_content.txt"], "r", encoding="utf-8") as f:
                readme_text = f.read().strip()
                
            if readme_text:
                print(f"    >>> [AI Analyst] 🕵️‍♂️ 侦探 Agent 正在阅读 {m_name} 的代码说明书...")
                client = OpenAI()
                analyst = Agent(
                    title="README Analyst", 
                    expertise="代码阅读与命令行解析", 
                    goal="提取推断命令", 
                    role="代码侦探", 
                    model=MODEL_NAME
                )
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": README_ANALYST_PROMPT},
                            {"role": "user", "content": f"请分析以下 README 内容并推导预测命令：\n\n{readme_text}"}
                        ],
                        temperature=0.1 # 必须极低，防止乱改命令
                    )
                    inferred_cmd = resp.choices[0].message.content.strip().replace("```bash", "").replace("```sh", "").replace("```", "").strip()
                    
                    if inferred_cmd and ("python" in inferred_cmd.lower() or "sh" in inferred_cmd.lower() or "./" in inferred_cmd):
                        # 完美推导！覆写掉原来的智障模板！
                        m["inference_cmd_template"] = inferred_cmd
                        print(f"    >>> [OK] 🎉 完美！自动推导出预测命令: \n        {inferred_cmd}")
                    else:
                        print(f"    !!! [Warn] AI 推导的命令貌似不合法: {inferred_cmd}")
                except Exception as e:
                    print(f"    !!! [Error] 调用大模型推导命令失败: {e}")
            else:
                print(f"    !!! [Warn] 仓库中未找到 README 文件，无法自动推导命令。")

    return models_info