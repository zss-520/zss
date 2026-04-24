import os
import requests
import re
import subprocess
from pathlib import Path

class ToolRegistry:
    """
    统一 API 工具调用总线 (Tool Hub)
    未来所有新增的数据库 API、爬虫、系统命令等，全部作为静态方法注册在这里。
    """

    @staticmethod
    def download_zenodo_dataset(target_url: str, output_dir: Path) -> bool:
        """【工具 1】Zenodo API 自动化流式下载器"""
        record_id = None
        match = re.search(r'zenodo\.(\d+)', target_url)
        if match:
            record_id = match.group(1)
        elif target_url.isdigit():
            record_id = target_url
            
        if not record_id:
            print(f"❌ [API 工具] 无法解析 Zenodo Record ID: {target_url}")
            return False

        api_url = f"https://zenodo.org/api/records/{record_id}"
        print(f"    -> 📡 [API 工具] 正在请求 Zenodo 元数据: {api_url}")
        
        try:
            response = requests.get(api_url, timeout=15)
            response.raise_for_status()
            data = response.json()
            files = data.get("files", [])
            
            if not files:
                print("    -> ⚠️ [API 工具] Zenodo 记录无附件。")
                return False
                
            output_dir.mkdir(parents=True, exist_ok=True)
            for file_info in files:
                filename = file_info.get("key", "unknown_file")
                download_link = file_info.get("links", {}).get("self")
                if not download_link: continue
                
                file_path = output_dir / filename
                print(f"       下载中: {filename} ... ", end="", flush=True)
                with requests.get(download_link, stream=True, timeout=20) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print("✅")
            return True
        except Exception as e:
            print(f"    -> ❌ [API 工具] Zenodo 下载失败: {e}")
            return False

    @staticmethod
    def clone_github_repo(repo_url: str, output_dir: Path) -> bool:
        """【工具 2】自动化 Git Clone 工具"""
        if not repo_url.startswith("http"): return False
        print(f"    -> 📡 [API 工具] 正在执行 Git Clone: {repo_url}")
        try:
            subprocess.run(["git", "clone", repo_url, str(output_dir)], check=True, capture_output=True)
            print("       ✅ 源码拉取成功！")
            return True
        except subprocess.CalledProcessError as e:
            print(f"       ⚠️ Git Clone 失败: {e.stderr.decode()}")
            return False

    @staticmethod
    def query_bio_database(db_name: str, query_params: dict) -> dict:
        """【工具 3】预留坑位：未来用于查询 APD3, DBAASP, Uniprot 等生物学数据库 API"""
        print(f"    -> 📡 [API 工具] 模拟调用 {db_name} 数据库，参数: {query_params}")
        # 这里以后可以写类似 requests.post(f"https://api.{db_name}.org/search", json=query_params)
        return {"status": "success", "data": "API_RETURN_DATA_PLACEHOLDER"}

    # ==========================================
    # 核心路由中心 (Dispatcher)
    # ==========================================
    @classmethod
    def execute(cls, tool_name: str, **kwargs):
        """主叫路由：传入工具名和对应参数进行分发"""
        if tool_name == "zenodo":
            return cls.download_zenodo_dataset(**kwargs)
        elif tool_name == "github":
            return cls.clone_github_repo(**kwargs)
        elif tool_name == "bio_db":
            return cls.query_bio_database(**kwargs)
        else:
            print(f"❌ [路由报错] 未注册的 API 工具: {tool_name}")
            return False