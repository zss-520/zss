import json
import os
import shutil
import subprocess
import urllib.request
import re
from pathlib import Path
from openai import OpenAI
from tool_executor import ToolRegistry

from config import MODEL_NAME
from prompts import DOWNLOAD_GUIDE_PROMPT  # <--- 统一从这里引入提示词

def generate_llm_download_guide(item_name: str, item_type: str, url: str, description: str = "") -> str:
    """调用大模型，根据 URL 和描述动态生成下载指南"""
    client = OpenAI()
    
    # 动态格式化统一管理的 prompt，传入 description
    prompt = DOWNLOAD_GUIDE_PROMPT.format(
        item_type=item_type, 
        item_name=item_name, 
        url=url,
        description=description if description else "无特殊说明，请下载完整的公开数据集包"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "1. 请在浏览器中打开上述链接。\n2. 寻找与该项目相关的 Download、Data 或 Code 按钮进行下载。"

def smart_fetch_resource(item_name: str, item_type: str, url: str, target_dir_str: str, description: str = "") -> bool:
    """智能资源获取器：优先全自动下载 -> 网页勘探 -> 人工介入"""
    target_dir = Path(target_dir_str)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_abs_path = os.path.abspath(target_dir)
    
    is_auto_success = False
    
    # ==========================================
    # 阶段 1：尝试全自动获取 (Auto-Fetch)
    # ==========================================
    if "github.com" in url or "gitee.com" in url:
        print(f"    -> 🔍 识别为独立代码仓库，尝试自动执行 git clone...")
        
        clean_url = re.sub(r'/(tree|blob)/.*$', '', url)
        
        # 🚨 提取分支名 (针对类似 /tree/master 的链接)
        branch_name = None
        branch_match = re.search(r'/(tree|blob)/([^/]+)', url)
        if branch_match:
            branch_name = branch_match.group(2)
            
        clone_path = target_dir if item_type == "模型" else Path("data/downloads") / item_name.replace(" ", "_")
        
        if clone_path.exists() and any(clone_path.iterdir()):
            print("    ✅ 本地已存在该仓库，跳过下载。")
            is_auto_success = True
        else:
            clone_cmd = ["git", "clone"]
            if branch_name:
                print(f"    -> 🎯 锁定并强制拉取特定分支: {branch_name}")
                clone_cmd.extend(["-b", branch_name])
            clone_cmd.extend([clean_url, str(clone_path)])
            
            result = subprocess.run(
                clone_cmd, 
                capture_output=True, 
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            if result.returncode == 0:
                print("    ✅ 自动克隆成功！")
                is_auto_success = True
            else:
                err_msg = (result.stderr or result.stdout or "未知网络或Git错误").strip()[:50]
                print(f"    ⚠️ 自动克隆失败 (原因: {err_msg}...)。")

        # 数据集专属：提取纯数据集仓库里的文件
        if is_auto_success and item_type == "数据集":
            found_files = 0
            for ext in ["*.fasta", "*.fa", "*.csv", "*.tsv", "*.txt", "*.xlsx", "*.json"]:
                for file_path in clone_path.rglob(ext):
                    dest_path = target_dir / file_path.name
                    if not dest_path.exists():
                        shutil.copy2(file_path, dest_path)
                        found_files += 1
            print(f"    [Auto-Extract] 从 GitHub 仓库中提取了 {found_files} 个数据文件至 {target_dir}")
            if found_files == 0:
                is_auto_success = False
                print("    ⚠️ 仓库中未直接找到标准格式数据文件，转入后续流程。")

    elif url.lower().endswith(('.zip', '.tar.gz', '.csv', '.fasta', '.txt')):
        print(f"    -> 🔍 识别为直接下载链接，尝试自动 wget/urllib...")
        try:
            filename = url.split('/')[-1]
            # 伪装 User-Agent 防止被拒绝
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(target_dir / filename, 'wb') as out_file:
                    out_file.write(response.read())
            print("    ✅ 自动下载成功！")
            is_auto_success = True
        except urllib.error.HTTPError as e:
            print(f"    ⚠️ 自动下载失败 (HTTP {e.code}): 链接可能已死。")
        except Exception as e:
            print(f"    ⚠️ 自动下载失败 ({e})。")

    if is_auto_success:
        return True

    # ==========================================
    # 🚨 阶段 1.5：网页勘探者 (Web Scavenger) - 寻找搬家后的 GitHub
    # ==========================================
    if url.startswith("http") and "github.com" not in url.lower() and "zenodo.org" not in url.lower():
        print(f"    -> 🕵️ 正在勘探网页源代码，寻找是否已搬迁至 GitHub...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
                # 寻找形如 https://github.com/author/repo 的链接
                gh_match = re.search(r'https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+', html)
                if gh_match:
                    new_gh_url = gh_match.group(0)
                    print(f"    -> 💡 惊喜！在原网页中发现了搬迁后的 GitHub 链接: {new_gh_url}")
                    print(f"    -> 🔄 正在重定向，移交 Git 引擎重新处理...")
                    # 递归调用自身，走 GitHub 下载路线！
                    return smart_fetch_resource(item_name, item_type, new_gh_url, target_dir_str)
                else:
                    print("    -> ⚠️ 网页勘探完毕，未发现隐藏的 GitHub 链接。")
                    
        except urllib.error.HTTPError as e:
            print(f"    -> ❌ 网页访问失败 (HTTP {e.code}): 原链接已失效 (404/403) 或被移除。")
        except Exception as e:
            print(f"    -> ❌ 网页勘探异常: {e}")

    # ==========================================
    # 阶段 2：大模型动态指南 + 人工介入 (Human-in-the-loop)
    # ==========================================
    print("\n" + "🚨" * 20)
    print(f"🛑 [需人工介入] 无法自动获取 {item_type}: 【{item_name}】")
    print("🚨" * 20)
    
    print(f"🤖 正在呼叫大模型分析网页下载策略...")
    # 🚨 将 description 传给 AI
    guide = generate_llm_download_guide(item_name, item_type, url, description)
    
    print("\n👉 【AI 下载向导】:")
    print(guide)
    print(f"\n📂 请下载后将文件(或解压后的文件夹)放入此目录：\n   ---> {target_abs_path}")
    print("-" * 50)
    
    while True:
        ans = input(f">>> 📥 文件放好了吗？(输入 'y' 继续，'skip' 跳过该项): ").strip().lower()
        if ans == 'y':
            has_files = any(f.name for f in target_dir.iterdir() if not f.name.startswith('.'))
            if has_files:
                print("✅ 成功检测到人工存入的文件！继续自动化管线...\n")
                return True
            else:
                ans2 = input("⚠️ 目标文件夹里还是空的！你确定要强制继续吗？(y/n): ").strip().lower()
                if ans2 == 'y':
                    print("✅ 强制继续...\n")
                    return True
        elif ans == 'skip':
            print(f"⏭️ 已跳过 {item_name}。\n")
            return False
        else:
            print("❌ 输入无效，请输入 'y' 或 'skip'。")

# ==========================================
# 供主程序调用的批量拉取接口 (整合智能路由)
# ==========================================
def load_strategy(path: str = "data/benchmark_strategy.json") -> dict:
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def get_cloned_model_urls() -> dict:
    """提取情报知识库中已存在的所有模型 URL 及其本地克隆路径"""
    db_path = Path("data/model_knowledge_db.json")
    if not db_path.exists(): return {}
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            db = json.load(f)
        url_map = {}
        for m in db.get("models", []):
            url = m.get("repo_url", "")
            name = m.get("model_name", "Unknown").replace(" ", "_")
            if url and url != "无链接":
                clean_url = re.sub(r'/(tree|blob)/.*$', '', url.lower())
                url_map[clean_url] = Path(f"data/models/{name}")
        return url_map
    except Exception:
        return {}

def fetch_datasets():
    strategy = load_strategy()
    datasets = strategy.get("recommended_datasets", [])
    if not datasets: return
    
    print("\n========== [Fetcher] 智能数据集与 API 路由调度启动 ==========")
    cloned_models = get_cloned_model_urls()
    
    for ds in datasets:
        name = ds.get("dataset_name", "Unknown").replace(" ", "_")
        url = ds.get("download_url", "")
        # 🚨 提取 JSON 中的 description
        description = ds.get("description", "")
        target_dir = Path(f"data/datasets/{name}")
        
        print(f"\n>>> 🚀 [智能获取] 正在处理 数据集: {name} ...")
        print(f"    🔗 链接: {url}")
        
        if not url or url.lower() == "null" or url == "无链接":
            print("    -> ⚠️ 无有效链接，转入人机交互后备机制...")
            # 🚨 传入 description
            smart_fetch_resource(name, "数据集", url, str(target_dir), description)
            continue

        clean_url = re.sub(r'/(tree|blob)/.*$', '', url.lower())

        if clean_url in cloned_models and cloned_models[clean_url].exists():
            print("    -> 🔍 识别到该数据集与某个【预测模型】出自同一开源仓库！")
            print("    -> 📂 拦截 Git Clone！直接从本地模型源码中提取数据文件...")
            
            source_dir = cloned_models[clean_url]
            found_files = 0
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for ext in ["*.fasta", "*.fa", "*.csv", "*.tsv", "*.txt", "*.xlsx", "*.json"]:
                for file_path in source_dir.rglob(ext):
                    dest_path = target_dir / file_path.name
                    if not dest_path.exists():
                        shutil.copy2(file_path, dest_path)
                        found_files += 1
                        
            if found_files > 0:
                print(f"    ✅ 提取成功！从模型 {source_dir.name} 扒出了 {found_files} 个数据文件至 {target_dir}")
                continue
            else:
                print("    ⚠️ 本地模型源码中没有找到标准格式的数据文件，转入后备流程...")

        if "github.com" in url.lower() or "gitee.com" in url.lower():
            # 🚨 传入 description
            smart_fetch_resource(name, "数据集", url, str(target_dir), description)
            
        elif "zenodo.org" in url.lower():
            print("    -> 🔍 识别到 Zenodo 平台，正在移交 API 工具中心全自动处理...")
            success = ToolRegistry.execute("zenodo", target_url=url, output_dir=target_dir)
            if not success:
                print("    -> ⚠️ API 调用异常，转入人机交互后备机制...")
                # 🚨 传入 description
                smart_fetch_resource(name, "数据集", url, str(target_dir), description)
                
        else:
            # 🚨 传入 description
            smart_fetch_resource(name, "数据集", url, str(target_dir), description)
def fetch_models(selected_models: list):
    if not selected_models: return
    print("\n========== [Fetcher] 自动模型获取启动 ==========")
    for mod in selected_models:
        name = mod.get("model_name", "Unknown").replace(" ", "_")
        url = mod.get("repo_url", "")
        if not url or url.lower() == "null" or url == "无链接": continue
        
        print(f"\n>>> 🚀 [智能获取] 正在处理 模型: {name} ...")
        print(f"    🔗 链接: {url}")
        smart_fetch_resource(name, "模型", url, f"data/models/{name}")