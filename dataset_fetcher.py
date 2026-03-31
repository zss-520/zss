import json
import os
import shutil
import subprocess
from pathlib import Path

def load_strategy(path: str = "data/benchmark_strategy.json") -> dict:
    if not os.path.exists(path):
        print(f"!!! [Error] 未找到策略文件 {path}。请先运行 prepare_models.py")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_datasets():
    strategy = load_strategy()
    datasets = strategy.get("recommended_datasets", [])
    
    if not datasets:
        print(">>> [Info] 策略中没有推荐的数据集。")
        return

    base_dataset_dir = Path("data/datasets")
    download_cache_dir = Path("data/downloads")
    base_dataset_dir.mkdir(parents=True, exist_ok=True)
    download_cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== [Data Fetcher] 自动数据集拉取引擎启动 ==========")

    for ds in datasets:
        ds_name = ds.get("dataset_name", "Unknown_Dataset").replace(" ", "_")
        url = ds.get("download_url", "")
        role = ds.get("role", "unknown")
        
        print(f"\n>>> 正在处理 [{role}] 数据集: {ds_name}")
        print(f"    来源链接: {url}")

        target_dir = base_dataset_dir / ds_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 🟢 情况 1: 如果是 GitHub 链接，全自动拉取！
        if "github.com" in url:
            repo_name = url.strip("/").split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
                
            clone_path = download_cache_dir / repo_name
            
            if clone_path.exists():
                print(f"    [Skip] GitHub 仓库 {repo_name} 已存在缓存。")
            else:
                print(f"    [Git] 正在自动 Clone 仓库: {url} ...")
                try:
                    subprocess.run(["git", "clone", url, str(clone_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print("    [Git] Clone 成功！")
                except Exception as e:
                    print(f"    !!! [Error] Git Clone 失败: {e}")
                    continue

            # 智能扫描提取数据文件 (fasta, csv, txt)
            found_files = 0
            for ext in ["*.fasta", "*.fa", "*.csv", "*.txt"]:
                for file_path in clone_path.rglob(ext):
                    dest_path = target_dir / file_path.name
                    if not dest_path.exists():
                        shutil.copy2(file_path, dest_path)
                        found_files += 1
            
            print(f"    [Auto-Extract] 从 GitHub 仓库成功提取了 {found_files} 个潜在数据文件至 {target_dir}")

        # 🔴 情况 2: 如果是数据库官网，呼叫人类支援
        elif "http" in url:
            print(f"    [Manual Required] 这是一个 Web 数据库入口，机器无法稳定自动点击。")
            print(f"    👉 请手动在浏览器打开: {url}")
            print(f"    👉 下载原始文件后，将其放入此文件夹: {target_dir.absolute()}")
            
        else:
            print(f"    [Warning] 无效的下载链接。")

    print("\n========== [Data Fetcher] 拉取阶段结束 ==========")
    print(">>> 接下来，你可以运行 data_prep.py 来执行 CD-HIT 去重和长度清洗了！")

if __name__ == "__main__":
    fetch_datasets()